# planning.py
"""Planning algorithms that combine learning and planning.

Implémente les algorithmes de planification intégrée :
- Dyna-Q : Q-Learning + planning avec modèle appris
- Dyna-Q+ (optionnel) : Dyna-Q avec exploration encouragée

Ces algorithmes alternent entre :
1. Apprentissage direct depuis l'expérience réelle
2. Planification via un modèle appris de l'environnement

Exemple d'usage
---------------
```python
from envs.gridworld import GridWorld  
from algos.planning import dyna_q

pi_opt, Q, model = dyna_q(GridWorld(), num_episodes=1000, n_planning_steps=50)
```
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Set
import random

__all__ = [
    "dyna_q",
    "dyna_q_plus",
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

class SimpleModel:
    """Modèle déterministe simple pour les environnements du projet.
    
    Stocke les transitions observées sous forme (s, a) -> (s', r).
    Pour les environnements déterministes, chaque (s,a) n'a qu'une seule issue.
    """
    
    def __init__(self):
        # Dictionnaire : (state, action) -> (next_state, reward)
        self.transitions: Dict[Tuple[int, int], Tuple[int, float]] = {}
        # Set des couples (state, action) observés
        self.observed_sa: Set[Tuple[int, int]] = set()
    
    def update(self, state: int, action: int, next_state: int, reward: float):
        """Met à jour le modèle avec une transition observée."""
        self.transitions[(state, action)] = (next_state, reward)
        self.observed_sa.add((state, action))
    
    def sample(self) -> Tuple[int, int, int, float]:
        """Échantillonne une transition aléatoire du modèle.
        
        Returns
        -------
        state, action, next_state, reward : tuple
            Une transition (s, a, s', r) du modèle.
        """
        if not self.observed_sa:
            raise ValueError("Aucune transition dans le modèle")
        
        state, action = random.choice(list(self.observed_sa))
        next_state, reward = self.transitions[(state, action)]
        return state, action, next_state, reward
    
    def get_transition(self, state: int, action: int) -> Tuple[int, float]:
        """Récupère la transition pour (s, a) si elle existe."""
        return self.transitions.get((state, action), (None, None))

# ---------------------------------------------------------------------------
# 1) Dyna-Q
# ---------------------------------------------------------------------------

def dyna_q(
    env: "EnvStruct",
    num_episodes: int,
    n_planning_steps: int = 50,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
) -> Tuple[np.ndarray, np.ndarray, SimpleModel]:
    """Dyna-Q : Q-Learning intégré avec planification par modèle.
    
    Algorithme 8.2 du Sutton & Barto.
    
    À chaque étape :
    1. Exécute une action réelle et met à jour Q
    2. Met à jour le modèle avec la transition observée  
    3. Effectue n_planning_steps étapes de planification simulée
    
    Parameters
    ----------
    env : EnvStruct
        Environnement RL.
    num_episodes : int
        Nombre d'épisodes d'entraînement.
    n_planning_steps : int
        Nombre d'étapes de planification après chaque action réelle.
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
    policy : np.ndarray, shape (S, A)
        Politique greedy finale.
    Q : np.ndarray, shape (S, A)
        Fonction action-valeur apprise.
    model : SimpleModel
        Modèle appris de l'environnement.
    """
    S, A = env.num_states(), env.num_actions()
    Q = np.zeros((S, A))
    model = SimpleModel()
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        env.reset()
        s = env.state()
        
        while not env.is_game_over():
            # (a) Apprentissage direct : action réelle
            a = _epsilon_greedy_action(Q, s, epsilon)
            
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score
            s_next = env.state()
            
            # Mise à jour Q-Learning
            if env.is_game_over():
                target = r  # Pas d'état suivant
            else:
                target = r + gamma * Q[s_next].max()
            Q[s, a] += alpha * (target - Q[s, a])
            
            # Mise à jour du modèle
            model.update(s, a, s_next, r)
            
            # (b) Planification : n_planning_steps étapes simulées
            for _ in range(n_planning_steps):
                if not model.observed_sa:
                    break  # Pas encore de transitions dans le modèle
                
                # Échantillonner une transition du modèle
                s_sim, a_sim, s_next_sim, r_sim = model.sample()
                
                # Mise à jour Q via la transition simulée
                # Note : on suppose que les terminaux sont cohérents dans le modèle
                # Pour simplifier, on applique toujours la mise à jour complète
                target_sim = r_sim + gamma * Q[s_next_sim].max()
                Q[s_sim, a_sim] += alpha * (target_sim - Q[s_sim, a_sim])
            
            s = s_next
        
        # Décroissance d'epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    policy = _greedy_policy_from_Q(Q)
    return policy, Q, model

# ---------------------------------------------------------------------------
# 2) Dyna-Q+ (optionnel)
# ---------------------------------------------------------------------------

class ModelPlus:
    """Modèle étendu pour Dyna-Q+ qui suit le temps depuis la dernière visite."""
    
    def __init__(self, kappa: float = 1e-4):
        # Modèle de base
        self.transitions: Dict[Tuple[int, int], Tuple[int, float]] = {}
        self.observed_sa: Set[Tuple[int, int]] = set()
        
        # Extension Dyna-Q+ : suivi temporel
        self.last_visit: Dict[Tuple[int, int], int] = {}  # (s,a) -> dernier timestep
        self.timestep = 0
        self.kappa = kappa  # Bonus d'exploration
    
    def update(self, state: int, action: int, next_state: int, reward: float):
        """Met à jour le modèle avec timestamp."""
        self.timestep += 1
        self.transitions[(state, action)] = (next_state, reward)
        self.observed_sa.add((state, action))
        self.last_visit[(state, action)] = self.timestep
    
    def sample(self) -> Tuple[int, int, int, float]:
        """Échantillonne avec bonus d'exploration basé sur le temps."""
        if not self.observed_sa:
            raise ValueError("Aucune transition dans le modèle")
        
        state, action = random.choice(list(self.observed_sa))
        next_state, base_reward = self.transitions[(state, action)]
        
        # Bonus d'exploration : √(timestep - last_visit)
        time_since_visit = self.timestep - self.last_visit[(state, action)]
        exploration_bonus = self.kappa * np.sqrt(time_since_visit)
        
        # Récompense augmentée pour encourager l'exploration
        reward = base_reward + exploration_bonus
        
        return state, action, next_state, reward

def dyna_q_plus(
    env: "EnvStruct",
    num_episodes: int,
    n_planning_steps: int = 50,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    kappa: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, ModelPlus]:
    """Dyna-Q+ : Dyna-Q avec exploration encouragée.
    
    Extension de Dyna-Q qui ajoute un bonus d'exploration proportionnel
    au temps écoulé depuis la dernière visite d'un couple (état, action).
    Utile dans les environnements non-stationnaires.
    
    Parameters
    ----------
    env : EnvStruct
        Environnement RL.
    num_episodes : int
        Nombre d'épisodes.
    n_planning_steps : int
        Étapes de planification par action réelle.
    alpha : float
        Taux d'apprentissage.
    gamma : float
        Facteur de discount.
    epsilon_start : float
        Epsilon initial.
    epsilon_min : float
        Epsilon minimal.
    epsilon_decay : float
        Décroissance d'epsilon.
    kappa : float
        Coefficient du bonus d'exploration.
        
    Returns
    -------
    policy : np.ndarray
        Politique greedy finale.
    Q : np.ndarray
        Fonction action-valeur.
    model : ModelPlus
        Modèle étendu avec suivi temporel.
    """
    S, A = env.num_states(), env.num_actions()
    Q = np.zeros((S, A))
    model = ModelPlus(kappa=kappa)
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        env.reset()
        s = env.state()
        
        while not env.is_game_over():
            # (a) Apprentissage direct
            a = _epsilon_greedy_action(Q, s, epsilon)
            
            prev_score = env.score()
            env.step(a)
            r = env.score() - prev_score
            s_next = env.state()
            
            # Mise à jour Q-Learning (sans bonus pour l'expérience réelle)
            if env.is_game_over():
                target = r
            else:
                target = r + gamma * Q[s_next].max()
            Q[s, a] += alpha * (target - Q[s, a])
            
            # Mise à jour du modèle
            model.update(s, a, s_next, r)
            
            # (b) Planification avec bonus d'exploration
            for _ in range(n_planning_steps):
                if not model.observed_sa:
                    break
                
                # Échantillonner avec bonus d'exploration
                s_sim, a_sim, s_next_sim, r_sim_bonus = model.sample()
                
                # Mise à jour Q avec la récompense augmentée
                target_sim = r_sim_bonus + gamma * Q[s_next_sim].max()
                Q[s_sim, a_sim] += alpha * (target_sim - Q[s_sim, a_sim])
            
            s = s_next
        
        # Décroissance d'epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    policy = _greedy_policy_from_Q(Q)
    return policy, Q, model