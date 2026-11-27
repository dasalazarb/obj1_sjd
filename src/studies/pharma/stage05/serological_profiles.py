"""As-of laboratory and serological origin profiles."""
from .clinical_profiles import interval_profiles


def serological_profiles(intervals, features):
    return interval_profiles(intervals, features, {'laboratory', 'serology'})
