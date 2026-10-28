"""Serializable restricted cubic spline: A is a literal prefix of B and C.

Four pooled-time knots (boundaries and 1/3, 2/3 quantiles) give three
nonconstant degrees of freedom: time plus two nonlinear terms. The nonlinear
terms are projected off [1, time] once, on the frozen estimation sample.
"""
from dataclasses import asdict, dataclass
import numpy as np
from src.block_A.secondary_comparison_models import PATTERNS


@dataclass(frozen=True)
class FrozenBasis:
    lower: float
    upper: float
    knots: tuple
    projection: tuple
    scales: tuple
    method: str = 'restricted_cubic_pooled_quantiles_1_3_2_3_projected'

    @classmethod
    def freeze(cls, time):
        t = np.asarray(time, dtype=float)
        lo, hi = float(t.min()), float(t.max())
        if not np.isfinite(t).all() or hi <= lo:
            raise ValueError('invalid spline bounds')
        z = (t-lo)/(hi-lo)
        knots = (0., *np.quantile(z, [1/3, 2/3]), 1.)
        if np.min(np.diff(knots)) < 1e-6:
            raise ValueError('nonunique_pooled_quantile_knots')
        raw = cls._raw(z, knots)
        projection = np.linalg.lstsq(np.column_stack([np.ones(len(z)), z]), raw, rcond=None)[0]
        h = raw-np.column_stack([np.ones(len(z)), z])@projection
        scales = np.sqrt(np.mean(h*h, axis=0))
        if np.min(scales) < 1e-10:
            raise ValueError('degenerate_nonlinear_basis')
        return cls(lo, hi, tuple(float(k) for k in knots), tuple(map(tuple, projection)), tuple(scales))

    @staticmethod
    def _raw(z, knots, derivative=0):
        z = np.asarray(z)
        k = np.asarray(knots)
        def power(a):
            return np.maximum(a, 0.)**(3-derivative)*(1., 3., 6.)[derivative]
        d = [(power(z-a)-power(z-k[-1]))/(k[-1]-a) for a in k[:-1]]
        return np.column_stack([d[j]-d[-1] for j in range(len(k)-2)])

    def nonlinear(self, time, derivative=0):
        t = np.atleast_1d(time).astype(float)
        z = (t-self.lower)/(self.upper-self.lower)
        linear = (np.column_stack([np.ones(len(z)), z]) if derivative == 0 else
                  np.column_stack([np.zeros(len(z)), np.ones(len(z))]) if derivative == 1 else
                  np.zeros((len(z), 2)))
        return ((self._raw(z, self.knots, derivative)-linear@np.asarray(self.projection)) /
                np.asarray(self.scales)/(self.upper-self.lower)**derivative)

    def matrix(self, patterns, time, model='C', derivative=0):
        t = np.atleast_1d(time).astype(float)
        p = np.asarray(patterns)
        if len(p) != len(t) or not set(p).issubset(PATTERNS):
            raise ValueError('invalid fixed group design')
        dummy = np.column_stack([p == g for g in PATTERNS[1:]]).astype(float)
        one = np.ones(len(t)) if derivative == 0 else np.zeros(len(t))
        trend = t if derivative == 0 else np.ones(len(t)) if derivative == 1 else np.zeros(len(t))
        a = np.column_stack([one, dummy*one[:, None], trend, dummy*trend[:, None]])
        if model == 'A':
            return a
        h = self.nonlinear(t, derivative)
        b = np.column_stack([a, h])
        if model == 'B':
            return b
        if model != 'C':
            raise ValueError('unknown model')
        return np.column_stack([b, *[dummy[:, j, None]*h for j in range(3)]])

    def row(self, pattern, time, model='C', derivative=0):
        return self.matrix([pattern], [time], model, derivative)[0]

    def as_dict(self):
        return dict(**asdict(self), knots_years=[self.lower+k*(self.upper-self.lower) for k in self.knots],
                    df_total=3, nonlinear_df=2, natural_boundary_constraint='second derivative zero at both boundaries')
