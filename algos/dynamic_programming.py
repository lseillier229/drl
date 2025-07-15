# dynamic_programming.py
"""Dynamic‑Programming algorithms (tabular) for small MDPs.

The functions assume you already have a *model* of the environment,
expressed as two NumPy arrays:

    P  shape (S, A, S)   — transition probabilities P[s, a, s′]
    R  shape (S, A, S)   — expected immediate reward when moving s→s′ via a

…plus a discount factor γ.

This mirrors the matrices you ascribes in the notebook (``p`` / ``R``) and
plays nicely with your deterministic ``EnvStruct`` environments once you have
extracted such a model.  A minimal helper (`build_model_from_env`) is provided
for *deterministic* EnvStruct‑style envs.

Typical workflow
----------------
```
from envs.gridworld import GridWorld            # ← ton env maison
from algo.dynamic_programming import policy_iteration

P, R = build_model_from_env(GridWorld())        # tabular model
policy, V = policy_iteration(P, R, gamma=1.0)   # optimal π*
```

If you already know *your* `P`/`R`, just call the algorithms directly.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, List

__all__ = [
    "policy_evaluation",
    "policy_improvement",
    "policy_iteration",
    "value_iteration",
    "build_model_from_env",
]


def policy_evaluation(
        P: np.ndarray,
        R: np.ndarray,
        pi: np.ndarray,
        gamma: float = 1.0,
        theta: float = 1e-8) -> np.ndarray:
    """Iterative Policy Evaluation (IP; Sutton&Barto alg. 4.1).

    Parameters
    ----------
    P : (S, A, S) array
        Transition probabilities.
    R : (S, A, S) array
        Rewards associated with transitions.
    pi : (S, A) array
        Stochastic policy — each row sums to 1.
    gamma : float
        Discount factor.
    theta : float
        Convergence threshold.

    Returns
    -------
    V : (S,) array of state‑values under *pi*.
    """
    S, A, _ = P.shape
    V = np.zeros(S)
    while True:
        delta = 0.0
        for s in range(S):
            v = V[s]
            V[s] = sum(
                pi[s, a] * (P[s, a] @ (R[s, a] + gamma * V))  # dot over s′
                for a in range(A)
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


def policy_improvement(
        P: np.ndarray,
        R: np.ndarray,
        V: np.ndarray,
        gamma: float = 1.0,
) -> Tuple[np.ndarray, bool]:
    """Greedy policy improvement.

    Returns a *deterministic* improved policy and a boolean flag
    indicating if it is unchanged (i.e. stable).
    """
    S, A, _ = P.shape
    pi_new = np.zeros((S, A))
    stable = True

    for s in range(S):
        # Calculer Q-values pour cet état
        q_sa = np.array([P[s, a] @ (R[s, a] + gamma * V) for a in range(A)])

        # Trouver la meilleure action (gestion des ties)
        max_q = q_sa.max()
        best_actions = np.where(np.abs(q_sa - max_q) < 1e-10)[0]  # Toutes les actions optimales

        # Politique déterministique : choisir la première action optimale pour consistance
        a_best = best_actions[0]

        # Nouvelle politique déterministique
        pi_best = np.zeros(A)
        pi_best[a_best] = 1.0

        # Vérifier si la politique a changé (comparaison numérique robuste)
        if not np.allclose(pi_best, pi_new[s], atol=1e-10):
            stable = False

        pi_new[s] = pi_best

    return pi_new, stable


def policy_iteration(
    S: List[int],
    A: List[int],
    R: List[int],
    T: List[int],
    p: np.ndarray,
    theta: float = 0.00001,
    gamma: float = 0.999999,
):
  V = np.random.random((len(S),))
  V[T] = 0.0
  pi = np.array([np.random.choice(A) for s in S])
  pi[T] = 0

  while True:

    # Policy Evaluation
    while True:
      delta = 0.0

      for s in S:
        v = V[s]
        total = 0.0
        for s_p in S:
          for r_index in range(len(R)):
            r = R[r_index]
            total += p[s, pi[s], s_p, r_index] * (r + gamma * V[s_p])
        V[s] = total
        abs_diff = np.abs(v - V[s])
        delta = np.maximum(delta, abs_diff)

      if delta < theta:
        break

    # Policy Improvement

    policy_stable = True
    for s in S:
      old_action = pi[s]
      best_a = None
      best_a_score = -999999999.99999
      for a in A:
        score = 0.0
        for s_p in S:
          for r_index in range(len(R)):
            r = R[r_index]
            score += p[s, a, s_p, r_index] * (r + gamma * V[s_p])
        if best_a is None or score > best_a_score:
          best_a = a
          best_a_score = score
      if best_a != old_action:
        policy_stable = False
      pi[s] = best_a

    if policy_stable:
      break

  return pi, V


def value_iteration(
        P: np.ndarray,
        R: np.ndarray,
        gamma: float = 1.0,
        theta: float = 1e-8,
        max_iter: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Value‑Iteration (Bellman optimality)

    Returns (optimal_policy, optimal_value).
    """
    S, A, _ = P.shape
    V = np.zeros(S)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(S):
            v = V[s]
            q_sa = np.array([P[s, a] @ (R[s, a] + gamma * V) for a in range(A)])
            V[s] = q_sa.max()
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    # derive greedy policy
    pi = np.zeros((S, A))
    for s in range(S):
        q_sa = np.array([P[s, a] @ (R[s, a] + gamma * V) for a in range(A)])
        pi[s, q_sa.argmax()] = 1.0
    return pi, V


# ---------------------------------------------------------------------------
# Helper: build model matrices from *deterministic* EnvStruct environments
# ---------------------------------------------------------------------------

def build_model_from_env(env: "EnvStruct") -> Tuple[np.ndarray, np.ndarray]:
    """Extract *deterministic* P and R from an EnvStruct env.

    Works iff `env.step(a)` is deterministic and the reward equals the
    *change* in `score()`.  The environment must expose a public attribute
    or property `s` (current state id) so it can be reset to arbitrary
    states.  All your small environments (`LineWorld`, `GridWorld`, etc.)
    respect this pattern.
    """
    import copy
    S = env.num_states()
    A = env.num_actions()
    P = np.zeros((S, A, S))
    R = np.zeros((S, A, S))

    # brute‑force over every state/action by cloning env
    for s in range(S):
        for a in range(A):
            e = copy.deepcopy(env)
            # place env in state *s*
            if hasattr(e, "s"):
                e.s = s  # LineWorld
            elif hasattr(e, "pos"):
                # GridWorld: decode (row, col)
                w = int(np.sqrt(S))
                e.pos = (s // w, s % w)
            else:
                # Pour les autres environnements, on ne peut pas créer le modèle
                # On va juste mettre des valeurs par défaut
                P[s, a, s] = 1.0  # Self-loop
                R[s, a, s] = 0.0
                continue

            # Vérifier si l'état est terminal avant de faire step
            if e.is_game_over():
                # État terminal : pas de transition possible, reste dans le même état
                P[s, a, s] = 1.0  # Self-loop
                R[s, a, s] = 0.0  # Pas de récompense supplémentaire
                continue

            v_before = e.score()
            try:
                e.step(a)
                s_next = e.state()
                r = e.score() - v_before
                P[s, a, s_next] = 1.0
                R[s, a, s_next] = r
            except:
                # En cas d'erreur (comme l'ancienne exception "Youpi")
                # considérer que l'action n'a pas d'effet
                P[s, a, s] = 1.0
                R[s, a, s] = 0.0

    return P, R