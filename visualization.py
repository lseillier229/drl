#!/usr/bin/env python3
"""
Module de visualisation et d'interaction pour les environnements RL.

Fournit :
- Visualisation graphique des environnements
- Mode de jeu manuel (humain)
- Démo pas à pas des politiques apprises
- Sauvegarde/chargement des politiques
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import time

class EnvironmentVisualizer:
    """Visualiseur générique pour les environnements RL."""
    
    def __init__(self, env):
        self.env = env
        self.env_name = env.__class__.__name__
        
        # Mapping des touches pour le contrôle manuel
        self.key_mappings = {
            'LineWorld': {'left': 0, 'right': 1},
            'GridWorld': {'up': 0, 'right': 1, 'down': 2, 'left': 3},
            'RPS': {'1': 0, '2': 1, '3': 2},  # 1=Rock, 2=Paper, 3=Scissors
            'MontyHall1': {'1': 0, '2': 1, '3': 2},  # Portes 1, 2, 3
            'MontyHall2': {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}  # Portes 1-5
        }
        
        self.fig = None
        self.ax = None
        self.current_key = None
    
    def visualize_state(self, policy=None, value_function=None):
        """Visualise l'état actuel de l'environnement."""
        
        if self.env_name == 'LineWorld':
            self._visualize_lineworld(policy, value_function)
        elif self.env_name == 'GridWorld':
            self._visualize_gridworld(policy, value_function)
        elif self.env_name == 'RPS':
            self._visualize_rps()
        elif self.env_name in ['MontyHall1', 'MontyHall2']:
            self._visualize_montyhall()
        else:
            print(f"Visualisation non implémentée pour {self.env_name}")
    
    def _visualize_lineworld(self, policy=None, value_function=None):
        """Visualisation spécifique pour LineWorld."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 3))
        else:
            self.ax.clear()
        
        n_states = self.env.num_states()
        current_state = self.env.state()
        
        # Dessiner les états
        for i in range(n_states):
            color = 'lightblue'
            if i == 0:
                color = 'red'  # Terminal gauche
            elif i == n_states - 1:
                color = 'green'  # Terminal droit
            elif i == current_state:
                color = 'yellow'  # Position actuelle
            
            rect = patches.Rectangle((i, 0), 1, 1, linewidth=2, 
                                   edgecolor='black', facecolor=color)
            self.ax.add_patch(rect)
            
            # Afficher la valeur si disponible
            if value_function is not None:
                self.ax.text(i + 0.5, 0.5, f'{value_function[i]:.2f}', 
                           ha='center', va='center', fontsize=10)
            
            # Afficher la politique si disponible
            if policy is not None:
                if policy.ndim == 2:
                    action = np.argmax(policy[i])
                else:
                    action = policy[i]
                
                if i not in [0, n_states-1]:  # Pas d'action dans les états terminaux
                    arrow_dir = '←' if action == 0 else '→'
                    self.ax.text(i + 0.5, -0.3, arrow_dir, 
                               ha='center', va='center', fontsize=20, color='blue')
        
        self.ax.set_xlim(-0.5, n_states - 0.5)
        self.ax.set_ylim(-0.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'LineWorld - Score: {self.env.score():.1f}')
        self.ax.axis('off')
        
        plt.tight_layout()
    
    def _visualize_gridworld(self, policy=None, value_function=None):
        """Visualisation spécifique pour GridWorld."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        else:
            self.ax.clear()
        
        n_rows = self.env.n_rows
        n_cols = self.env.n_cols
        agent_pos = self.env.agent_pos
        
        # Dessiner la grille
        for i in range(n_rows):
            for j in range(n_cols):
                # Couleur selon le type de case
                color = 'white'
                if (i, j) == (0, 4):
                    color = 'red'  # Piège
                elif (i, j) == (4, 4):
                    color = 'green'  # Objectif
                elif (i, j) == agent_pos:
                    color = 'yellow'  # Agent
                
                rect = patches.Rectangle((j, n_rows - i - 1), 1, 1, 
                                       linewidth=2, edgecolor='black', facecolor=color)
                self.ax.add_patch(rect)
                
                state_idx = i * n_cols + j
                
                # Afficher la valeur si disponible
                if value_function is not None:
                    self.ax.text(j + 0.5, n_rows - i - 0.8, f'{value_function[state_idx]:.2f}', 
                               ha='center', va='center', fontsize=9)
                
                # Afficher la politique si disponible
                if policy is not None and (i, j) not in self.env.terminal_states:
                    if policy.ndim == 2:
                        action = np.argmax(policy[state_idx])
                    else:
                        action = int(policy[state_idx])
                    
                    # Flèches pour les actions
                    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
                    self.ax.text(j + 0.5, n_rows - i - 0.5, arrows[action], 
                               ha='center', va='center', fontsize=20, color='blue')
        
        self.ax.set_xlim(0, n_cols)
        self.ax.set_ylim(0, n_rows)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'GridWorld - Score: {self.env.score():.1f}')
        self.ax.axis('off')
        
        plt.tight_layout()
    
    def _visualize_rps(self):
        """Visualisation pour Rock-Paper-Scissors."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
        else:
            self.ax.clear()
        
        stage = self.env.stage
        my_prev = self.env.my_prev
        adv_prev = self.env.adv_prev
        score = self.env.score()
        
        symbols = {-1: "?", 0: "✊", 1: "✋", 2: "✌️"}
        names = {-1: "?", 0: "Rock", 1: "Paper", 2: "Scissors"}
        
        # Affichage selon le stage
        if stage == 0:
            self.ax.text(0.5, 0.7, "Round 1", ha='center', fontsize=20, weight='bold')
            self.ax.text(0.5, 0.5, "Choose: 1=Rock, 2=Paper, 3=Scissors", ha='center', fontsize=16)
        elif stage == 1:
            self.ax.text(0.5, 0.8, "Round 1 Results", ha='center', fontsize=20, weight='bold')
            self.ax.text(0.3, 0.6, f"You: {names[my_prev]}", ha='center', fontsize=30)
            self.ax.text(0.7, 0.6, f"Opponent: {names[adv_prev]}", ha='center', fontsize=30)
            self.ax.text(0.5, 0.4, "Round 2 - Opponent will copy your Round 1 move!", 
                        ha='center', fontsize=14, color='red')
            self.ax.text(0.5, 0.3, "Choose: 1=Rock, 2=Paper, 3=Scissors", ha='center', fontsize=16)
        else:
            self.ax.text(0.5, 0.8, "Game Over!", ha='center', fontsize=24, weight='bold')
            self.ax.text(0.5, 0.6, f"Final Score: {score:+.0f}", ha='center', fontsize=20)
            self.ax.text(0.5, 0.4, "Press 'r' to reset", ha='center', fontsize=16)
        
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        
        plt.tight_layout()
    
    def _visualize_montyhall(self):
        """Visualisation pour Monty Hall."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
        else:
            self.ax.clear()
        
        n_doors = self.env.n_doors
        stage = self.env.stage
        chosen = self.env.chosen
        opened_mask = self.env.opened_mask
        score = self.env.score()
        
        # Dessiner les portes
        door_width = 0.8 / n_doors
        for i in range(n_doors):
            x = 0.1 + i * (0.8 / n_doors) + door_width / 2
            
            # Couleur de la porte
            is_open = (opened_mask >> i) & 1
            if is_open:
                color = 'lightgray'
                self.ax.text(x, 0.5, "Chevre", ha='center', va='center', fontsize=40)
            elif i == chosen and chosen < n_doors:
                color = 'yellow'
            else:
                color = 'brown'
            
            rect = patches.Rectangle((x - door_width/2, 0.3), door_width, 0.4,
                                   linewidth=2, edgecolor='black', facecolor=color)
            self.ax.add_patch(rect)
            
            # Numéro de porte
            self.ax.text(x, 0.2, f'{i+1}', ha='center', fontsize=16, weight='bold')
        
        # Instructions
        if self.env.is_game_over():
            self.ax.text(0.5, 0.9, f"Game Over! Score: {score:.0f}", 
                        ha='center', fontsize=20, weight='bold',
                        color='green' if score > 0 else 'red')
            self.ax.text(0.5, 0.85, "Press 'r' to reset", ha='center', fontsize=14)
        else:
            self.ax.text(0.5, 0.9, f"Stage {stage+1}/{n_doors-1}", 
                        ha='center', fontsize=18, weight='bold')
            self.ax.text(0.5, 0.85, f"Choose a door (1-{n_doors})", ha='center', fontsize=14)
        
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.ax.set_title(f'{self.env_name} - Monty Hall Paradox')
        
        plt.tight_layout()
    
    def play_manual(self):
        """Mode de jeu manuel avec contrôle clavier."""
        print(f"\n=== MODE MANUEL - {self.env_name} ===")
        print("Contrôles :")
        
        if self.env_name in self.key_mappings:
            for key, action in self.key_mappings[self.env_name].items():
                action_names = {
                    'LineWorld': ['Gauche', 'Droite'],
                    'GridWorld': ['Haut', 'Droite', 'Bas', 'Gauche'],
                    'RPS': ['Pierre', 'Papier', 'Ciseaux'],
                    'MontyHall1': ['Porte 1', 'Porte 2', 'Porte 3'],
                    'MontyHall2': ['Porte 1', 'Porte 2', 'Porte 3', 'Porte 4', 'Porte 5']
                }
                if self.env_name in action_names:
                    print(f"  {key} : {action_names[self.env_name][action]}")
        
        print("  r : Reset")
        print("  q : Quitter")
        
        self.env.reset()
        
        def on_key_press(event):
            if event.key == 'q':
                plt.close()
                return
            elif event.key == 'r':
                self.env.reset()
                self.visualize_state()
                plt.draw()
                return
            
            # Traiter l'action
            if self.env_name in self.key_mappings and not self.env.is_game_over():
                if event.key in self.key_mappings[self.env_name]:
                    action = self.key_mappings[self.env_name][event.key]
                    self.env.step(action)
                    self.visualize_state()
                    plt.draw()
                    
                    if self.env.is_game_over():
                        print(f"\nPartie terminée ! Score final : {self.env.score()}")
        
        self.visualize_state()
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        plt.show()
    
    def demonstrate_policy(self, policy, delay=1.0, num_episodes=1):
        """Démontre une politique apprise pas à pas."""
        print(f"\n=== DÉMONSTRATION DE LA POLITIQUE ===")
        
        for episode in range(num_episodes):
            print(f"\nÉpisode {episode + 1}/{num_episodes}")
            self.env.reset()
            step = 0
            
            while not self.env.is_game_over():
                state = self.env.state()
                
                # Déterminer l'action selon la politique
                if policy.ndim == 2:
                    action = np.argmax(policy[state])
                else:
                    action = int(policy[state])
                
                print(f"  Step {step}: État={state}, Action={action}")
                
                # Visualiser l'état actuel
                self.visualize_state(policy)
                plt.pause(delay)
                
                # Exécuter l'action
                self.env.step(action)
                step += 1
            
            # État final
            self.visualize_state(policy)
            print(f"  Épisode terminé! Score: {self.env.score()}")
            plt.pause(delay * 2)
        
        plt.show()


class PolicyManager:
    """Gestionnaire pour sauvegarder/charger les politiques."""
    
    def __init__(self, save_dir="saved_policies"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_policy(self, policy_data: Dict[str, Any], filename: str):
        """Sauvegarde une politique et ses métadonnées."""
        filepath = self.save_dir / f"{filename}.pkl"
        
        # Ajouter un timestamp
        policy_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(filepath, 'wb') as f:
            pickle.dump(policy_data, f)
        
        print(f"Politique sauvegardée : {filepath}")
    
    def load_policy(self, filename: str) -> Dict[str, Any]:
        """Charge une politique sauvegardée."""
        filepath = self.save_dir / f"{filename}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Politique non trouvée : {filepath}")
        
        with open(filepath, 'rb') as f:
            policy_data = pickle.load(f)
        
        print(f"Politique chargée : {filepath}")
        print(f"   Algorithme : {policy_data.get('algorithm', 'Unknown')}")
        print(f"   Environnement : {policy_data.get('environment', 'Unknown')}")
        print(f"   Score : {policy_data.get('score', 'N/A')}")
        
        return policy_data
    
    def list_policies(self):
        """Liste toutes les politiques sauvegardées."""
        print("\n=== POLITIQUES SAUVEGARDÉES ===")
        
        policies = list(self.save_dir.glob("*.pkl"))
        
        if not policies:
            print("Aucune politique trouvée.")
            return
        
        for i, filepath in enumerate(policies):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            print(f"\n{i+1}. {filepath.stem}")
            print(f"   Algorithme : {data.get('algorithm', 'Unknown')}")
            print(f"   Environnement : {data.get('environment', 'Unknown')}")
            print(f"   Score : {data.get('score', 'N/A')}")
            print(f"   Date : {data.get('timestamp', 'Unknown')}")


def create_training_animation(env, algorithm_name, scores_history, interval=50):
    """Crée une animation de l'entraînement."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Graphique des scores
    ax1.set_xlim(0, len(scores_history))
    ax1.set_ylim(min(scores_history) - 0.1, max(scores_history) + 0.1)
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Entraînement {algorithm_name} sur {env.__class__.__name__}')
    ax1.grid(True)
    
    line, = ax1.plot([], [], 'b-')
    
    # Visualisation de l'environnement
    visualizer = EnvironmentVisualizer(env)
    
    def animate(frame):
        # Mettre à jour le graphique des scores
        line.set_data(range(frame), scores_history[:frame])
        
        # Afficher l'état actuel dans ax2
        # (Cette partie dépendra de l'implémentation spécifique)
        
        return line,
    
    anim = FuncAnimation(fig, animate, frames=len(scores_history), 
                        interval=interval, blit=True)
    
    return anim