import numpy as np
import pytest
from src.block_A.temporal_spline_design import FrozenBasis
from temporal_spline_fixtures import evaluated


def test_literal_nested_design_frozen_prediction_serialization_and_natural_boundaries():
    ev, _, _ = evaluated()
    assert ev['gate']['eligible_C'], ev['gate']
    basis, data, matrices = ev['basis'], ev['data'], ev['matrices']
    np.testing.assert_array_equal(matrices['A'], matrices['B'][:, :8])
    np.testing.assert_array_equal(matrices['B'], matrices['C'][:, :10])
    restored = FrozenBasis(**{k: v for k, v in basis.as_dict().items() if k in FrozenBasis.__dataclass_fields__})
    np.testing.assert_array_equal(restored.matrix(data.observed_pattern, data.time), matrices['C'])
    np.testing.assert_allclose(basis.nonlinear([basis.lower, basis.upper], 2), 0, atol=1e-12)
    linear = np.column_stack([np.ones(len(data)), data.time])
    np.testing.assert_allclose(linear.T@basis.nonlinear(data.time), 0, atol=1e-10)
    assert [np.linalg.matrix_rank(matrices[m]) for m in 'ABC'] == [8, 10, 16]


def test_analytic_derivatives_match_finite_differences():
    ev, _, _ = evaluated()
    b = ev['basis']
    t = np.array([.14, .81, 1.4, 2.8])
    eps = 1e-5
    np.testing.assert_allclose(b.nonlinear(t, 1), (b.nonlinear(t+eps)-b.nonlinear(t-eps))/(2*eps), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(b.nonlinear(t, 2), (b.nonlinear(t+eps, 1)-b.nonlinear(t-eps, 1))/(2*eps), rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError):
        FrozenBasis.freeze([0., 0., 0., 1.])
