# monte_carlo.py
"""Tabular Monte-Carlo algorithms for small deterministic EnvStruct environments.

Toutes les fonctions nécessitent un environnement *env* conforme à l’interface
`EnvStruct` (reset, step, score, state, etc.) introduite dans ta librairie.

Fonctions fournies
------------------
* **mc_prediction_first_visit**        – estimation V^π par premières visites.
* **mc_control_es**                    – Monte-Carlo Exploring Starts (ES).
* **on_policy_first_visit_mc_control** – contrôle on-policy ε-greedy.
* **off_policy_mc_control**            – contrôle off-policy par IS pondérée.

Représentation
--------------
Les politiques π sont des tableaux NumPy shape (S, A) formant des lois de
probabilité (ligne = état).  Pour retourner une politique *déterministe*,
on place 1.0 sur l’action choisie.

Exemple
-------
```python
from envs.gridworld import GridWorld
from algo.monte_carlo import mc_control_es

pi_opt, Q = mc_control_es(GridWorld(), num_episodes=20000)
```
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Tuple, Callable

__all__ = [
    "mc_prediction_first_visit",
    "mc_control_es",
    "on_policy_first_visit_mc_control",
    "off_policy_mc_control",
]

# ---------------------------------------------------------------------------
#  Utils
# ---------------------------------------------------------------------------

def _random_policy(env) -> np.ndarray:
    """Uniform random policy."""
    S, A = env.num_states(), env.num_actions()
    return np.full((S, A), 1.0 / A)


def _greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Return deterministic greedy policy w.r.t Q."""
    S, A = Q.shape
    pi = np.zeros_like(Q)
    best = Q.argmax(axis=1)
    pi[np.arange(S), best] = 1.0
    return pi


def _epsilon_greedy_policy(Q: np.ndarray, epsilon: float) -> np.ndarray:
    S, A = Q.shape
    pi = np.full_like(Q, epsilon / A)
    best = Q.argmax(axis=1)
    pi[np.arange(S), best] += 1.0 - epsilon
    return pi


def _generate_episode(env, policy: np.ndarray):
    """Yield list of (state, action, reward) until termination."""
    episode = []
    env.reset()
    prev_score = env.score()
    while not env.is_game_over():
        s = env.state()
        a = np.random.choice(env.num_actions(), p=policy[s])
        env.step(a)
        r = env.score() - prev_score
        prev_score = env.score()
        episode.append((s, a, r))
    return episode

# ---------------------------------------------------------------------------
#  1)  First-visit Monte-Carlo Prediction
# ---------------------------------------------------------------------------

def mc_prediction_first_visit(
    env: "EnvStruct",
    policy: np.ndarray,
    num_episodes: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """Estime V^π via premières visites."""
    S = env.num_states()
    V = np.zeros(S)
    returns: list[list[float]] = [list() for _ in range(S)]  # stockage R

    for _ in range(num_episodes):
        episode = _generate_episode(env, policy)
        G = 0.0
        visited = set()
        # parcours inverse pour calculer G_t
        for t in reversed(range(len(episode))):
            s, _, r = episode[t]
            G = gamma * G + r
            if s not in visited:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visited.add(s)
    return V

# ---------------------------------------------------------------------------
# 2) Monte-Carlo Exploring-Starts Control (ES)
# ---------------------------------------------------------------------------

def mc_control_es(
    env: "EnvStruct",
    num_episodes: int,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo *Exploring Starts* (Algorithme 5.3 du SB).

    Hypothèse : chaque (état, action) peut être choisi comme départ.
    """
    S, A = env.num_states(), env.num_actions()
    Q = np.zeros((S, A))
    N = np.zeros((S, A))  # compteurs pour la moyenne
    pi = _random_policy(env)

    for _ in range(num_episodes):
        # 1) Exploring-start : on force un couple (s0, a0) aléatoire
        env.reset()
        # ---- nouvelle ligne : états non terminaux uniquement
        non_terminals = [s for s in range(S) if not env._is_terminal(s)]  # helper à ajouter si besoin
        s0 = np.random.choice(non_terminals)
        if hasattr(env, "s"):
            env.s = s0
        elif hasattr(env, "pos"):
            w = int(np.sqrt(S))
            env.pos = (s0 // w, s0 % w)
        a0 = np.random.randint(A)
        env.step(a0)
        episode = [(s0, a0, env.score())] + _generate_episode(env, pi)

        # 2) Retour cumulatif G et mise à jour première visite
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            if t == 0:
                # reward pour le premier pas (déjà inclus dans env.score)
                r = episode[t + 1][2] - episode[t][2] if len(episode) > 1 else 0.0
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]
        # 3) Amélioration greedy
        pi = _greedy_policy(Q)
    return pi, Q

# ---------------------------------------------------------------------------
# 3) On-policy First-Visit MC Control (ε-greedy)
# ---------------------------------------------------------------------------

def on_policy_first_visit_mc_control(
    env: "EnvStruct",
    num_episodes: int,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.999,
) -> Tuple[np.ndarray, np.ndarray]:
    """On-policy MC control avec ε-greedy décroissant."""
    S, A = env.num_states(), env.num_actions()
    Q = np.zeros((S, A))
    N = np.zeros((S, A))
    epsilon = epsilon_start

    for _ in range(num_episodes):
        pi = _epsilon_greedy_policy(Q, epsilon)
        episode = _generate_episode(env, pi)

        G = 0.0
        visited_sa = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited_sa:
                visited_sa.add((s, a))
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    final_pi = _greedy_policy(Q)
    return final_pi, Q

# ---------------------------------------------------------------------------
# 4) Off-policy MC Control (Importance Sampling pondérée)
# ---------------------------------------------------------------------------

def off_policy_mc_control(
    env: "EnvStruct",
    num_episodes: int,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Off-policy MC control (Weighted IS)
    *Target* policy est appris (greedy), *behavior* est uniforme.
    """
    S, A = env.num_states(), env.num_actions()
    Q = np.zeros((S, A))
    C = np.zeros((S, A))  # accumulateur IS
    target_pi = _greedy_policy(Q)     # initialement arbitraire (tout 1ère action)

    behavior_pi = _random_policy(env)

    for _ in range(num_episodes):
        episode = _generate_episode(env, behavior_pi)
        G = 0.0
        W = 1.0  # produit des ratios
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            C[s, a] += W
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
            # mettre à jour la politique cible au fur et à mesure
            target_pi = _greedy_policy(Q)
            if target_pi[s, a] == 0:
                break  # probabilité 0 → on peut sortir plus tôt
            W /= behavior_pi[s, a]
    return target_pi, Q
