# temporal_difference.py
"""Temporal Difference Learning algorithms for tabular RL.

Implémente les algorithmes TD classiques :
- SARSA (on-policy)
- Q-Learning (off-policy)
- Expected SARSA (optionnel)

Ces algorithmes apprennent directement à partir d'épisodes sans nécessiter
un modèle complet de l'environnement, contrairement aux méthodes DP.

Exemple d'usage
---------------
```python
from envs.gridworld import GridWorld
from algos.temporal_difference import sarsa, q_learning

# Q-Learning
pi_opt, Q = q_learning(GridWorld(), num_episodes=10000)

# SARSA
pi_opt, Q = sarsa(GridWorld(), num_episodes=10000)
```
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

__all__ = [
    "sarsa",
    "q_learning", 
    "expected_sarsa",
]

# ---------------------------------------------------------------------------
#  Utils partagés
# ---------------------------------------------------------------------------

def _epsilon_greedy_action(Q: np.ndarray, state: int, epsilon: float) -> int:
    """Choisit une action selon une politique ε-greedy."""
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return Q[state].argmax()

def _greedy_policy_from_Q(Q: np.ndarray) -> np.ndarray:
    """Extrait la politique greedy déterministe de Q."""
    S, A = Q.shape
    pi = np.zeros((S, A))
    best_actions = Q.argmax(axis=1)
    pi[np.arange(S), best_actions] = 1.0
    return pi

# ---------------------------------------------------------------------------
# 1) SARSA (on-policy)
# ---------------------------------------------------------------------------

def sarsa(
    env: "EnvStruct",
    num_episodes: int,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
) -> Tuple[np.ndarray, np.ndarray]:
    """SARSA : State-Action-Reward-State-Action (on-policy TD control).
    
    Algorithme 6.4 du Sutton & Barto.
    
    Parameters
    ----------
    env : EnvStruct
        Environnement RL conforme à l'interface.
    num_episodes : int
        Nombre d'épisodes d'entraînement.
    alpha : float
        Taux d'apprentissage (learning rate).
    gamma : float
        Facteur de discount.
    epsilon_start : float
        Valeur initiale d'epsilon pour ε-greedy.
    epsilon_min : float
        Valeur minimale d'epsilon.
    epsilon_decay : float
        Facteur de décroissance d'epsilon.
        
    Returns
    -------
    policy : np.ndarray, shape (S, A)
        Politique greedy finale extraite de Q.
    Q : np.ndarray, shape (S, A)
        Fonction action-valeur apprise.
    """
    S, A = env.num_states(), env.num_actions()
    Q = np.zeros((S, A))
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        env.reset()
        s = env.state()
        a = _epsilon_greedy_action(Q, s, epsilon)
        
        while not env.is_game_over():
            # Exécuter l'action a dans l'état s
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score
            s_next = env.state()
            
            if env.is_game_over():
                # Mise à jour finale (pas d'action suivante)
                Q[s, a] += alpha * (r - Q[s, a])
                break
            else:
                # Choisir la prochaine action a_next selon ε-greedy
                a_next = _epsilon_greedy_action(Q, s_next, epsilon)
                # Mise à jour SARSA
                Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])
                # Transition
                s, a = s_next, a_next
        
        # Décroissance d'epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    policy = _greedy_policy_from_Q(Q)
    return policy, Q

# ---------------------------------------------------------------------------
# 2) Q-Learning (off-policy)
# ---------------------------------------------------------------------------

def q_learning(
    env: "EnvStruct",
    num_episodes: int,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
) -> Tuple[np.ndarray, np.ndarray]:
    """Q-Learning : off-policy TD control.
    
    Algorithme 6.5 du Sutton & Barto.
    La politique de comportement est ε-greedy, mais la mise à jour
    utilise max_a Q(s',a) (politique greedy).
    
    Parameters
    ----------
    env : EnvStruct
        Environnement RL conforme à l'interface.
    num_episodes : int
        Nombre d'épisodes d'entraînement.
    alpha : float
        Taux d'apprentissage.
    gamma : float
        Facteur de discount.
    epsilon_start : float
        Valeur initiale d'epsilon.
    epsilon_min : float
        Valeur minimale d'epsilon.
    epsilon_decay : float
        Facteur de décroissance d'epsilon.
        
    Returns
    -------
    policy : np.ndarray, shape (S, A)
        Politique greedy finale.
    Q : np.ndarray, shape (S, A)
        Fonction action-valeur apprise.
    """
    S, A = env.num_states(), env.num_actions()
    Q = np.zeros((S, A))
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        env.reset()
        s = env.state()
        
        while not env.is_game_over():
            # Choisir action selon ε-greedy (politique de comportement)
            a = _epsilon_greedy_action(Q, s, epsilon)
            
            # Exécuter l'action
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score
            s_next = env.state()
            
            if env.is_game_over():
                # Mise à jour finale
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                # Mise à jour Q-Learning : max_a Q(s', a)
                max_q_next = Q[s_next].max()
                Q[s, a] += alpha * (r + gamma * max_q_next - Q[s, a])
            
            s = s_next
        
        # Décroissance d'epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    policy = _greedy_policy_from_Q(Q)
    return policy, Q

# ---------------------------------------------------------------------------
# 3) Expected SARSA (optionnel)
# ---------------------------------------------------------------------------

def expected_sarsa(
    env: "EnvStruct",
    num_episodes: int,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expected SARSA : utilise l'espérance sur toutes les actions possibles.
    
    Au lieu d'utiliser Q(s', a') pour une action spécifique (SARSA) ou
    max_a Q(s', a) (Q-Learning), Expected SARSA utilise :
    E[Q(s', A')] = Σ_a π(a|s') Q(s', a)
    
    Plus stable que SARSA classique mais légèrement plus coûteux.
    
    Parameters
    ----------
    env : EnvStruct
        Environnement RL.
    num_episodes : int
        Nombre d'épisodes.
    alpha : float
        Taux d'apprentissage.
    gamma : float
        Facteur de discount.
    epsilon_start : float
        Epsilon initial pour ε-greedy.
    epsilon_min : float
        Epsilon minimal.
    epsilon_decay : float
        Décroissance d'epsilon.
        
    Returns
    -------
    policy : np.ndarray
        Politique greedy finale.
    Q : np.ndarray
        Fonction action-valeur.
    """
    S, A = env.num_states(), env.num_actions()
    Q = np.zeros((S, A))
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        env.reset()
        s = env.state()
        
        while not env.is_game_over():
            # Choisir action selon ε-greedy
            a = _epsilon_greedy_action(Q, s, epsilon)
            
            # Exécuter l'action
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score
            s_next = env.state()
            
            if env.is_game_over():
                # Mise à jour finale
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                # Calculer l'espérance selon la politique ε-greedy courante
                # Politique ε-greedy : ε/A pour toutes actions + (1-ε) pour la meilleure
                best_action = Q[s_next].argmax()
                expected_q = 0.0
                for a_prime in range(A):
                    if a_prime == best_action:
                        prob = (1 - epsilon) + epsilon / A
                    else:
                        prob = epsilon / A
                    expected_q += prob * Q[s_next, a_prime]
                
                # Mise à jour Expected SARSA
                Q[s, a] += alpha * (r + gamma * expected_q - Q[s, a])
            
            s = s_next
        
        # Décroissance d'epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    policy = _greedy_policy_from_Q(Q)
    return policy, Q