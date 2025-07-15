# monte_carlo.py
"""Tabular Monte-Carlo algorithms for small deterministic EnvStruct environments."""
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
    """Generate episode following policy."""
    episode = []
    env.reset()

    while not env.is_game_over():
        s = env.state()
        a = np.random.choice(env.num_actions(), p=policy[s])

        # Enregistrer l'état et l'action AVANT le step
        prev_score = env.score()
        env.step(a)
        r = env.score() - prev_score

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
    returns: list[list[float]] = [list() for _ in range(S)]

    for _ in range(num_episodes):
        episode = _generate_episode(env, policy)
        G = 0.0
        visited = set()

        # Parcours inverse pour calculer les retours
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
    """Monte Carlo Exploring Starts."""
    S, A = env.num_states(), env.num_actions()
    Q = np.zeros((S, A))
    N = np.zeros((S, A))  # Compteurs de visites
    pi = _random_policy(env)

    for episode_idx in range(num_episodes):
        # Exploring start : choisir un état et une action au hasard
        env.reset()

        # Obtenir un état de départ valide
        start_state = np.random.randint(S)
        start_action = np.random.randint(A)

        # Pour certains environnements, on doit vérifier que l'état n'est pas terminal
        # On peut faire plusieurs tentatives
        max_attempts = 100
        for _ in range(max_attempts):
            env.reset()

            # Essayer de mettre l'environnement dans l'état start_state
            if hasattr(env, 's'):  # LineWorld
                env.s = start_state
            elif hasattr(env, 'agent_pos'):  # GridWorld
                w = int(np.sqrt(S))
                env.agent_pos = (start_state // w, start_state % w)
            elif hasattr(env, 'state') and hasattr(env, 'stage'):  # RPS, MontyHall
                # Pour ces environnements, on ne peut pas définir arbitrairement l'état
                # On génère un épisode normal
                episode = []
                prev_score = env.score()
                env.step(start_action)
                r = env.score() - prev_score
                episode.append((env.state(), start_action, r))
                episode.extend(_generate_episode(env, pi))
                break
            else:
                # Environnement non supporté pour ES, on fait un épisode normal
                episode = _generate_episode(env, pi)
                break

            # Vérifier si l'état est terminal
            if not env.is_game_over():
                # C'est bon, on peut commencer l'épisode
                prev_score = env.score()
                env.step(start_action)
                r = env.score() - prev_score
                episode = [(start_state, start_action, r)]
                episode.extend(_generate_episode(env, pi))
                break
            else:
                # État terminal, on réessaie avec un autre état
                start_state = np.random.randint(S)
        else:
            # Si on n'a pas réussi, on fait un épisode normal
            episode = _generate_episode(env, pi)

        # Calcul des retours et mise à jour
        G = 0.0
        visited_sa = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            if (s, a) not in visited_sa:
                visited_sa.add((s, a))
                N[s, a] += 1
                # Mise à jour incrémentale de la moyenne
                Q[s, a] += (G - Q[s, a]) / N[s, a]

        # Amélioration : politique greedy par rapport à Q
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

    for episode_idx in range(num_episodes):
        # Politique ε-greedy courante
        pi = _epsilon_greedy_policy(Q, epsilon)

        # Générer un épisode
        episode = _generate_episode(env, pi)

        # Calcul des retours
        G = 0.0
        visited_sa = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            if (s, a) not in visited_sa:
                visited_sa.add((s, a))
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]

        # Décroissance d'epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Retourner la politique greedy finale
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
    """Off-policy MC control avec Importance Sampling pondérée."""
    S, A = env.num_states(), env.num_actions()
    Q = np.zeros((S, A))
    C = np.zeros((S, A))  # Somme cumulée des poids d'importance

    # Politique cible (target) : initialement aléatoire, deviendra greedy
    target_pi = _random_policy(env)

    # Politique de comportement (behavior) : toujours aléatoire uniforme
    behavior_pi = _random_policy(env)

    for episode_idx in range(num_episodes):
        # Générer un épisode avec la politique de comportement
        episode = _generate_episode(env, behavior_pi)

        G = 0.0
        W = 1.0  # Poids d'importance

        # Parcours inverse de l'épisode
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            # Mise à jour avec importance sampling
            C[s, a] += W
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])

            # Mise à jour de la politique cible pour être greedy
            target_pi = _greedy_policy(Q)

            # Si la politique cible ne choisirait pas cette action, on arrête
            if target_pi[s, a] == 0:
                break

            # Mise à jour du poids d'importance
            W = W * (target_pi[s, a] / behavior_pi[s, a])

    return target_pi, Q