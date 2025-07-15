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
    pi_new = np.zeros((S, A))
    stable = True

    for s in range(S):
        # Action actuelle selon l'ancienne politique (pour vérifier la stabilité)
        old_action = pi_new[s].argmax()

        # Calculer Q(s,a) pour toutes les actions
        q_values = np.zeros(A)
        for a in range(A):
            for s_next in range(S):
                q_values[a] += P[s, a, s_next] * (R[s, a, s_next] + gamma * V[s_next])

        # Politique greedy : choisir la meilleure action
        best_action = np.argmax(q_values)
        pi_new[s] = np.zeros(A)
        pi_new[s, best_action] = 1.0

        # Vérifier si la politique a changé
        if best_action != old_action:
            stable = False

    return pi_new, stable


def policy_iteration(
        P: np.ndarray,
        R: np.ndarray,
        gamma: float = 1.0,
        theta: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Policy Iteration algorithm."""
    S, A, _ = P.shape

    # Initialisation : politique aléatoire uniforme
    pi = np.ones((S, A)) / A

    while True:
        # Policy Evaluation
        V = policy_evaluation(P, R, pi, gamma, theta)

        # Policy Improvement
        pi_new, stable = policy_improvement(P, R, V, gamma)

        if stable:
            break

        pi = pi_new

    return pi, V


def value_iteration(
        P: np.ndarray,
        R: np.ndarray,
        gamma: float = 1.0,
        theta: float = 1e-8,
        max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Value Iteration algorithm."""
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


def build_model_from_env(env: "EnvStruct") -> Tuple[np.ndarray, np.ndarray]:
    """Extract deterministic P and R from an EnvStruct env."""
    S = env.num_states()
    A = env.num_actions()
    P = np.zeros((S, A, S))
    R = np.zeros((S, A, S))

    # Helper to check if a state is terminal
    def is_terminal_state(env_copy, state):
        """Check if a state is terminal by setting the env to that state."""
        # Save current state
        original_state = get_state(env_copy)

        # Set to test state
        set_state(env_copy, state)
        is_term = env_copy.is_game_over()

        # Restore original state
        set_state(env_copy, original_state)

        return is_term

    # Helper functions to get/set state
    def get_state(e):
        if hasattr(e, 's'):
            return e.s
        elif hasattr(e, 'pos'):
            return e.pos
        elif hasattr(e, 'agent_pos'):
            return e.agent_pos
        elif hasattr(e, 'state'):
            return e.state()
        else:
            return None

    def set_state(e, state):
        if hasattr(e, 's'):
            e.s = state
        elif hasattr(e, 'pos'):
            if isinstance(state, int):
                # Convert state index to position for GridWorld
                w = int(np.sqrt(S))
                e.pos = (state // w, state % w)
            else:
                e.pos = state
        elif hasattr(e, 'agent_pos'):
            if isinstance(state, int):
                # Convert state index to position for GridWorld
                w = int(np.sqrt(S))
                e.agent_pos = (state // w, state % w)
            else:
                e.agent_pos = state

    # Build model for each (state, action) pair
    for s in range(S):
        # Check if this state is terminal using a fresh env copy
        env_test = copy.deepcopy(env)
        env_test.reset()

        # For grid-based environments
        if hasattr(env_test, 'agent_pos') or hasattr(env_test, 'pos'):
            w = int(np.sqrt(S))
            row, col = s // w, s % w
            set_state(env_test, (row, col))
        else:
            set_state(env_test, s)

        if env_test.is_game_over():
            # Terminal state: self-loop with 0 reward
            for a in range(A):
                P[s, a, s] = 1.0
                R[s, a, s] = 0.0
            continue

        # Non-terminal state: try each action
        for a in range(A):
            # Create fresh copy for this action
            e = copy.deepcopy(env)
            e.reset()

            # Set to state s
            if hasattr(e, 'agent_pos') or hasattr(e, 'pos'):
                w = int(np.sqrt(S))
                row, col = s // w, s % w
                set_state(e, (row, col))
            else:
                set_state(e, s)

            # Get score before action
            score_before = e.score()

            # Execute action
            try:
                e.step(a)

                # Get next state
                if hasattr(e, 'state'):
                    s_next = e.state()
                elif hasattr(e, 'agent_pos'):
                    r, c = e.agent_pos
                    s_next = r * int(np.sqrt(S)) + c
                elif hasattr(e, 'pos'):
                    r, c = e.pos
                    s_next = r * int(np.sqrt(S)) + c
                else:
                    s_next = e.s

                # Calculate reward
                reward = e.score() - score_before

                # Set transition
                P[s, a, s_next] = 1.0
                R[s, a, s_next] = reward

            except Exception as e:
                # If action fails, stay in same state with 0 reward
                P[s, a, s] = 1.0
                R[s, a, s] = 0.0

    return P, R