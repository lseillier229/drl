# dynamic_programming.py
"""Dynamic‑Programming algorithms (tabular) for small MDPs."""
from __future__ import annotations
import numpy as np
from typing import Tuple, List
import copy

__all__ = [
    "policy_evaluation",
    "policy_improvement",
    "policy_iteration",
    "value_iteration",
    "build_model_from_env",
]


# ============================================================================
# Format 1 : Matrices P, R (compatible avec l'interface originale)
# ============================================================================

def policy_evaluation(
        P: np.ndarray,
        R: np.ndarray,
        pi: np.ndarray,
        gamma: float = 1.0,
        theta: float = 1e-8) -> np.ndarray:
    """Iterative Policy Evaluation."""
    S, A, _ = P.shape
    V = np.zeros(S)

    while True:
        delta = 0.0
        for s in range(S):
            v = V[s]
            # Nouvelle valeur selon la politique π
            new_v = 0.0
            for a in range(A):
                # Somme sur tous les états suivants possibles
                for s_next in range(S):
                    new_v += pi[s, a] * P[s, a, s_next] * (R[s, a, s_next] + gamma * V[s_next])

            V[s] = new_v
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V


def policy_improvement(
        P: np.ndarray,
        R: np.ndarray,
        V: np.ndarray,
        gamma: float = 1.0) -> Tuple[np.ndarray, bool]:
    """Greedy policy improvement."""
    S, A, _ = P.shape
    pi_old = np.zeros((S, A))
    pi_new = np.zeros((S, A))
    stable = True

    for s in range(S):
        # Calculer Q(s,a) pour toutes les actions
        q_values = np.zeros(A)
        for a in range(A):
            for s_next in range(S):
                q_values[a] += P[s, a, s_next] * (R[s, a, s_next] + gamma * V[s_next])

        # Politique greedy : choisir la meilleure action
        best_action = np.argmax(q_values)
        pi_new[s] = np.zeros(A)
        pi_new[s, best_action] = 1.0

    return pi_new, stable


# ============================================================================
# Format 2 : Listes S, A, R, T et tenseur p (utilisé dans le code étudiant)
# ============================================================================

def policy_iteration(
        *args,
        **kwargs
):
    """Policy Iteration - adapte automatiquement selon le format des arguments."""

    if len(args) == 2 and isinstance(args[0], np.ndarray):
        # Format 1 : P, R
        P, R = args
        gamma = kwargs.get('gamma', 1.0)
        theta = kwargs.get('theta', 1e-8)
        return _policy_iteration_format1(P, R, gamma, theta)

    elif len(args) >= 5:
        # Format 2 : S, A, R, T, p
        S, A, R, T, p = args[:5]
        theta = kwargs.get('theta', 0.001)
        gamma = kwargs.get('gamma', 0.9)
        return _policy_iteration_format2(S, A, R, T, p, theta, gamma)

    else:
        raise ValueError("Format d'arguments non reconnu pour policy_iteration")


def _policy_iteration_format1(P, R, gamma=1.0, theta=1e-8):
    """Policy Iteration au format matrices P, R."""
    S, A, _ = P.shape

    # Initialisation : politique aléatoire uniforme
    pi = np.ones((S, A)) / A

    while True:
        # Policy Evaluation
        V = policy_evaluation(P, R, pi, gamma, theta)

        # Policy Improvement
        pi_new, stable = policy_improvement(P, R, V, gamma)

        if stable or np.allclose(pi, pi_new):
            break

        pi = pi_new

    return pi, V


def _policy_iteration_format2(S, A, R, T, p, theta=0.00001, gamma=0.9):
    """Policy Iteration au format listes et tenseur p."""
    V = np.random.random((len(S),))
    V[T] = 0.0
    pi = np.array([np.random.choice(A) for s in S])
    for t in T:
        pi[t] = 0

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
            if s in T:
                continue

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


def value_iteration(*args, **kwargs):
    """Value Iteration - adapte automatiquement selon le format des arguments."""

    if len(args) == 2 and isinstance(args[0], np.ndarray):
        # Format 1 : P, R
        P, R = args
        gamma = kwargs.get('gamma', 1.0)
        theta = kwargs.get('theta', 1e-8)
        max_iter = kwargs.get('max_iter', 1000)
        return _value_iteration_format1(P, R, gamma, theta, max_iter)

    elif len(args) >= 5:
        # Format 2 : S, A, R, T, p
        S, A, R, T, p = args[:5]
        theta = kwargs.get('theta', 0.00001)
        gamma = kwargs.get('gamma', 0.9)
        return _value_iteration_format2(S, A, R, T, p, theta, gamma)

    else:
        raise ValueError("Format d'arguments non reconnu pour value_iteration")


def _value_iteration_format1(P, R, gamma=1.0, theta=1e-8, max_iter=1000):
    """Value Iteration au format matrices P, R."""
    S, A, _ = P.shape
    V = np.zeros(S)

    for iteration in range(max_iter):
        delta = 0.0

        for s in range(S):
            v = V[s]

            # Calculer la valeur maximale sur toutes les actions
            q_values = np.zeros(A)
            for a in range(A):
                for s_next in range(S):
                    q_values[a] += P[s, a, s_next] * (R[s, a, s_next] + gamma * V[s_next])

            V[s] = np.max(q_values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    # Extraire la politique optimale
    pi = np.zeros((S, A))
    for s in range(S):
        q_values = np.zeros(A)
        for a in range(A):
            for s_next in range(S):
                q_values[a] += P[s, a, s_next] * (R[s, a, s_next] + gamma * V[s_next])

        best_action = np.argmax(q_values)
        pi[s, best_action] = 1.0

    return pi, V


def _value_iteration_format2(S, A, R, T, p, theta=0.00001, gamma=0.9):
    """Value Iteration au format listes et tenseur p."""
    V = np.zeros(len(S))
    V[T] = 0.0

    while True:
        delta = 0.0
        for s in S:
            if s in T:
                continue

            Q_s_a = []
            for a in A:
                q = 0.0
                for s_prime in S:
                    for r_index in range(len(R)):
                        r = R[r_index]
                        q += p[s, a, s_prime, r_index] * (r + gamma * V[s_prime])
                Q_s_a.append(q)
            max_Q = max(Q_s_a)
            delta = max(delta, abs(max_Q - V[s]))
            V[s] = max_Q
        if delta <= theta:
            break

    return V


def build_model_from_env(env) -> Tuple:
    """
    Convertit un EnvStruct en modèle MDP.

    Retourne soit :
    - Format simple : (P, R) pour compatibilité avec l'ancienne API
    - Format étendu : (S, A, R_vals, T, p) pour la nouvelle API

    Par défaut, retourne le format étendu.
    """
    # ------------------- 1. Tailles de base ------------------------
    nS = env.num_states()
    nA = env.num_actions()
    S = list(range(nS))
    A = list(range(nA))

    # ------------------- 2. Helpers état ↔ indice ------------------
    def get_state(e):
        if hasattr(e, "state"):
            return e.state()
        if hasattr(e, "s"):
            return e.s
        if hasattr(e, "agent_pos"):
            r, c = e.agent_pos
            w = int(np.sqrt(nS))
            return r * w + c
        if hasattr(e, "pos"):
            r, c = e.pos
            w = int(np.sqrt(nS))
            return r * w + c
        raise AttributeError("Impossible de récupérer l'état courant.")

    def set_state(e, s_idx):
        if hasattr(e, "s"):
            e.s = s_idx
        elif hasattr(e, "agent_pos"):
            w = int(np.sqrt(nS))
            e.agent_pos = (s_idx // w, s_idx % w)
        elif hasattr(e, "pos"):
            w = int(np.sqrt(nS))
            e.pos = (s_idx // w, s_idx % w)
        elif hasattr(e, "set_state"):
            e.set_state(s_idx)
        else:
            # Pour les environnements qui ne supportent pas set_state
            # (comme RPS, MontyHall), on skip
            pass

    # ------------------- 3. Reconstruire P et R --------------------
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA, nS))

    for s in S:
        for a in A:
            copy_env = copy.deepcopy(env)
            copy_env.reset()

            try:
                set_state(copy_env, s)
            except:
                # Environnement ne supporte pas set_state
                continue

            # Vérifier si l'état est terminal
            if copy_env.is_game_over():
                P[s, a, s] = 1.0
                R[s, a, s] = 0.0
                continue

            score_before = copy_env.score()
            try:
                copy_env.step(a)
                s_next = get_state(copy_env)
                reward = copy_env.score() - score_before

                P[s, a, s_next] = 1.0
                R[s, a, s_next] = reward
            except Exception:
                # Action illégale ou erreur
                P[s, a, s] = 1.0
                R[s, a, s] = 0.0

    # ------------------- 4. Format étendu --------------------------
    # Valeurs de récompense uniques
    R_vals = sorted(list(set(float(r) for r in R.flatten())))
    if not R_vals:
        R_vals = [0.0]

    # Mapping récompense -> index
    r_idx_map = {r: i for i, r in enumerate(R_vals)}

    # Tenseur p(s,a,s',r)
    p = np.zeros((nS, nA, nS, len(R_vals)))
    for s in S:
        for a in A:
            for s_next in S:
                prob = P[s, a, s_next]
                if prob > 0:
                    r = float(R[s, a, s_next])
                    r_idx = r_idx_map[r]
                    p[s, a, s_next, r_idx] = prob

    # États terminaux
    T = []
    for s in S:
        # Un état est terminal si toutes les actions mènent à lui-même
        is_terminal = True
        for a in A:
            if P[s, a, s] != 1.0:
                is_terminal = False
                break
        if is_terminal:
            T.append(s)

    return S, A, R_vals, T, p