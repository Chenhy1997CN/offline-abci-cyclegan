"""Trial-level utilities."""

from .grouping import group_trials_by_features
from .trial_goal import build_trial_goals, radial_scale_from_ratio, solve_trial_goal

__all__ = ["group_trials_by_features", "build_trial_goals", "radial_scale_from_ratio", "solve_trial_goal"]
