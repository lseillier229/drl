# algo/__init__.py
"""Collection d’algorithmes RL tabulaires.

Exporte directement les fonctions clefs :
    • Dynamic-Programming   –  policy_iteration, value_iteration, …
    • Monte-Carlo           –  mc_control_es, on/off-policy, …
Ainsi l’utilisateur peut simplement :
    >>> from algos import policy_iteration, mc_control_es
"""

from importlib.metadata import version as _v, PackageNotFoundError

# ---------------------------------------------------------------------------
#  Dynamic-Programming
# ---------------------------------------------------------------------------
from .dynamic_programming import (
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration,
    build_model_from_env,
)

# ---------------------------------------------------------------------------
#  Monte-Carlo
# ---------------------------------------------------------------------------
from .monte_carlo import (
    mc_prediction_first_visit,
    mc_control_es,
    on_policy_first_visit_mc_control,
    off_policy_mc_control,
)

__all__: list[str] = [
    # DP
    "policy_evaluation",
    "policy_improvement",
    "policy_iteration",
    "value_iteration",
    "build_model_from_env",
    # MC
    "mc_prediction_first_visit",
    "mc_control_es",
    "on_policy_first_visit_mc_control",
    "off_policy_mc_control",
]

# ---------------------------------------------------------------------------
#  Version helper (facultatif)
# ---------------------------------------------------------------------------
try:
    __version__: str = _v(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"
