#!/usr/bin/env python3
"""
Script principal de démonstration du projet RL.

Permet de :
- Entraîner n'importe quel algorithme sur n'importe quel environnement
- Visualiser les résultats
- Jouer manuellement
- Voir une démonstration pas à pas
- Sauvegarder/charger des politiques
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Imports des environnements
from envs import LineWorld, GridWorld, RPS, MontyHall1, MontyHall2

# Imports des algorithmes
from algos import (
    policy_iteration, value_iteration, build_model_from_env,
    mc_control_es, on_policy_first_visit_mc_control, off_policy_mc_control,
    sarsa, q_learning, expected_sarsa,
    dyna_q, dyna_q_plus
)

# Import du module de visualisation
from visualization import EnvironmentVisualizer, PolicyManager

class RLDemoSystem:
    """Système de démonstration pour le projet RL."""

    def __init__(self):
        # Registre des environnements
        self.environments = {
            'lineworld': LineWorld,
            'gridworld': GridWorld,
            'rps': RPS,
            'montyhall1': MontyHall1,
            'montyhall2': MontyHall2
        }

        # Registre des algorithmes
        self.algorithms = {
            # Dynamic Programming
            'policy_iteration': self._run_policy_iteration,
            'value_iteration': self._run_value_iteration,
            # Monte Carlo
            'mc_es': self._run_mc_es,
            'mc_on_policy': self._run_mc_on_policy,
            'mc_off_policy': self._run_mc_off_policy,
            # Temporal Difference
            'sarsa': self._run_sarsa,
            'q_learning': self._run_q_learning,
            'expected_sarsa': self._run_expected_sarsa,
            # Planning
            'dyna_q': self._run_dyna_q,
            'dyna_q_plus': self._run_dyna_q_plus
        }

        # Gestionnaire de politiques
        self.policy_manager = PolicyManager()

        # Paramètres par défaut
        self.default_params = {
            'num_episodes': 1000,
            'alpha': 0.1,
            'gamma': 1.0,
            'epsilon_start': 1.0,
            'epsilon_decay': 0.995,
            'n_planning_steps': 50
        }

    def train_and_demo(self, env_name: str, algo_name: str, **kwargs):
        """Entraîne un algorithme et fait une démonstration complète."""

        print(f"\n{'='*60}")
        print(f"ENTRAÎNEMENT : {algo_name.upper()} sur {env_name.upper()}")
        print(f"{'='*60}")

        # Créer l'environnement
        if env_name not in self.environments:
            print(f"❌ Environnement inconnu : {env_name}")
            return

        env_class = self.environments[env_name]
        env = env_class()

        # Vérifier l'algorithme
        if algo_name not in self.algorithms:
            print(f"❌ Algorithme inconnu : {algo_name}")
            return

        # Paramètres
        params = self.default_params.copy()
        params.update(kwargs)

        # Entraînement
        print(f"\n🚀 Début de l'entraînement...")
        start_time = time.time()

        try:
            result = self.algorithms[algo_name](env, params)
            training_time = time.time() - start_time

            print(f"✅ Entraînement terminé en {training_time:.2f} secondes")

            # Évaluation
            policy = result['policy']
            score = self._evaluate_policy(env, policy)
            print(f"📊 Score moyen sur 100 épisodes : {score:.3f}")

            # Sauvegarde
            save_data = {
                'algorithm': algo_name,
                'environment': env_name,
                'policy': policy,
                'score': score,
                'training_time': training_time,
                'parameters': params
            }

            if 'Q' in result:
                save_data['Q'] = result['Q']
            if 'V' in result:
                save_data['V'] = result['V']

            filename = f"{env_name}_{algo_name}_{int(time.time())}"
            self.policy_manager.save_policy(save_data, filename)

            # Menu d'options
            self._show_demo_menu(env, policy, result)

        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement : {e}")
            import traceback
            traceback.print_exc()

    def _show_demo_menu(self, env, policy, result):
        """Affiche le menu de démonstration."""

        while True:
            print(f"\n{'='*40}")
            print("OPTIONS DE DÉMONSTRATION")
            print(f"{'='*40}")
            print("1. Visualiser la politique")
            print("2. Démonstration pas à pas")
            print("3. Jouer manuellement")
            print("4. Statistiques détaillées")
            print("5. Quitter")

            choice = input("\nVotre choix (1-5) : ")

            if choice == '1':
                self._visualize_policy(env, policy, result)
            elif choice == '2':
                self._demo_step_by_step(env, policy)
            elif choice == '3':
                self._play_manual(env)
            elif choice == '4':
                self._show_statistics(env, policy, result)
            elif choice == '5':
                break
            else:
                print("❌ Choix invalide")

    def _visualize_policy(self, env, policy, result):
        """Visualise la politique apprise."""
        visualizer = EnvironmentVisualizer(env)

        # Réinitialiser l'environnement
        env.reset()

        # Obtenir la fonction de valeur si disponible
        value_function = result.get('V', None)
        if value_function is None and 'Q' in result:
            # Calculer V à partir de Q
            Q = result['Q']
            if policy.ndim == 2:
                # Politique stochastique
                value_function = np.sum(policy * Q, axis=1)
            else:
                # Politique déterministe
                value_function = Q[np.arange(len(policy)), policy.astype(int)]

        visualizer.visualize_state(policy, value_function)
        plt.show()

    def _demo_step_by_step(self, env, policy):
        """Démonstration pas à pas de la politique."""
        visualizer = EnvironmentVisualizer(env)

        try:
            delay = float(input("Délai entre les étapes (secondes) [1.0] : ") or "1.0")
            num_episodes = int(input("Nombre d'épisodes à montrer [3] : ") or "3")

            visualizer.demonstrate_policy(policy, delay, num_episodes)
        except ValueError:
            print("❌ Valeur invalide")

    def _play_manual(self, env):
        """Mode de jeu manuel."""
        visualizer = EnvironmentVisualizer(env)
        visualizer.play_manual()

    def _show_statistics(self, env, policy, result):
        """Affiche des statistiques détaillées."""
        print(f"\n{'='*40}")
        print("STATISTIQUES DÉTAILLÉES")
        print(f"{'='*40}")

        # Évaluation sur plusieurs épisodes
        scores = []
        lengths = []

        for _ in range(1000):
            env.reset()
            episode_length = 0

            while not env.is_game_over() and episode_length < 1000:
                state = env.state()
                if policy.ndim == 2:
                    action = np.random.choice(env.num_actions(), p=policy[state])
                else:
                    action = int(policy[state])
                env.step(action)
                episode_length += 1

            scores.append(env.score())
            lengths.append(episode_length)

        print(f"Score moyen : {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        print(f"Score min : {np.min(scores):.3f}")
        print(f"Score max : {np.max(scores):.3f}")
        print(f"Longueur moyenne des épisodes : {np.mean(lengths):.1f}")

        # Afficher la distribution des scores
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(scores), color='red', linestyle='--',
                   label=f'Moyenne: {np.mean(scores):.3f}')
        plt.xlabel('Score')
        plt.ylabel('Fréquence')
        plt.title('Distribution des scores sur 1000 épisodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def _evaluate_policy(self, env, policy,
                         num_episodes=100, max_steps=1_000):
        total = 0.0
        for _ in range(num_episodes):
            env.reset()
            steps = 0
            while not env.is_game_over() and steps < max_steps:
                s = env.state()
                a = (np.random.choice(env.num_actions(), p=policy[s])
                     if policy.ndim == 2 else int(policy[s]))
                env.step(a)
                steps += 1
            total += env.score()
        return total / num_episodes

    # ========== Wrappers pour les algorithmes ==========

    # main_demo.py  ── dans RLDemoSystem
    import numpy as np  # (déjà présent)

    def _run_policy_iteration(self, env, params):
        print("🔨 Construction du modèle de l'environnement…")
        S, A, R_vals, T, p = build_model_from_env(env)

        print("🧮 Exécution de Policy Iteration…")
        pi, V = policy_iteration(
            S, A, R_vals, T, p,
            theta=params.get("theta", 1e-8),
            gamma=params["gamma"],
        )
        return {"policy": pi, "V": V}

    def _run_value_iteration(self, env, params):
        print("🔨 Construction du modèle de l'environnement…")
        S, A, R_vals, T, p = build_model_from_env(env)

        print("🧮 Exécution de Value Iteration…")
        V = value_iteration(
            S, A, R_vals, T, p,
            theta=params.get("theta", 1e-8),
            gamma=params["gamma"],
        )

        # Greedy policy extraite de V
        pi = np.zeros(len(S), dtype=int)
        for s in S:
            q_best, a_best = -np.inf, 0
            for a in A:
                q_sa = sum(
                    p[s, a, s_p, r_idx] * (R_vals[r_idx] + params["gamma"] * V[s_p])
                    for s_p in S
                    for r_idx in range(len(R_vals))
                )
                if q_sa > q_best:
                    q_best, a_best = q_sa, a_best
                    a_best = a
            pi[s] = a_best

        return {"policy": pi, "V": V}

    def _run_mc_es(self, env, params):
        """Wrapper pour Monte Carlo ES."""
        print(f"🎲 Exécution de Monte Carlo ES ({params['num_episodes']} épisodes)...")
        pi, Q = mc_control_es(env, num_episodes=params['num_episodes'],
                             gamma=params['gamma'])
        return {'policy': pi, 'Q': Q}

    def _run_mc_on_policy(self, env, params):
        """Wrapper pour Monte Carlo On-Policy."""
        print(f"🎲 Exécution de MC On-Policy ({params['num_episodes']} épisodes)...")
        pi, Q = on_policy_first_visit_mc_control(
            env, num_episodes=params['num_episodes'], gamma=params['gamma'],
            epsilon_start=params['epsilon_start'], epsilon_decay=params['epsilon_decay']
        )
        return {'policy': pi, 'Q': Q}

    def _run_mc_off_policy(self, env, params):
        """Wrapper pour Monte Carlo Off-Policy."""
        print(f"🎲 Exécution de MC Off-Policy ({params['num_episodes']} épisodes)...")
        pi, Q = off_policy_mc_control(env, num_episodes=params['num_episodes'],
                                     gamma=params['gamma'])
        return {'policy': pi, 'Q': Q}

    def _run_sarsa(self, env, params):
        """Wrapper pour SARSA."""
        print(f"📈 Exécution de SARSA ({params['num_episodes']} épisodes)...")
        pi, Q = sarsa(env, num_episodes=params['num_episodes'],
                     alpha=params['alpha'], gamma=params['gamma'],
                     epsilon_start=params['epsilon_start'],
                     epsilon_decay=params['epsilon_decay'])
        return {'policy': pi, 'Q': Q}

    def _run_q_learning(self, env, params):
        """Wrapper pour Q-Learning."""
        print(f"📈 Exécution de Q-Learning ({params['num_episodes']} épisodes)...")
        pi, Q = q_learning(env, num_episodes=params['num_episodes'],
                          alpha=params['alpha'], gamma=params['gamma'],
                          epsilon_start=params['epsilon_start'],
                          epsilon_decay=params['epsilon_decay'])
        return {'policy': pi, 'Q': Q}

    def _run_expected_sarsa(self, env, params):
        """Wrapper pour Expected SARSA."""
        print(f"📈 Exécution de Expected SARSA ({params['num_episodes']} épisodes)...")
        pi, Q = expected_sarsa(env, num_episodes=params['num_episodes'],
                              alpha=params['alpha'], gamma=params['gamma'],
                              epsilon_start=params['epsilon_start'],
                              epsilon_decay=params['epsilon_decay'])
        return {'policy': pi, 'Q': Q}

    def _run_dyna_q(self, env, params):
        """Wrapper pour Dyna-Q."""
        print(f"🧠 Exécution de Dyna-Q ({params['num_episodes']} épisodes, "
              f"{params['n_planning_steps']} planning steps)...")
        pi, Q, model = dyna_q(env, num_episodes=params['num_episodes'],
                             n_planning_steps=params['n_planning_steps'],
                             alpha=params['alpha'], gamma=params['gamma'],
                             epsilon_start=params['epsilon_start'],
                             epsilon_decay=params['epsilon_decay'])
        return {'policy': pi, 'Q': Q, 'model': model}

    def _run_dyna_q_plus(self, env, params):
        """Wrapper pour Dyna-Q+."""
        print(f"🧠 Exécution de Dyna-Q+ ({params['num_episodes']} épisodes, "
              f"{params['n_planning_steps']} planning steps)...")
        pi, Q, model = dyna_q_plus(env, num_episodes=params['num_episodes'],
                                  n_planning_steps=params['n_planning_steps'],
                                  alpha=params['alpha'], gamma=params['gamma'],
                                  epsilon_start=params['epsilon_start'],
                                  epsilon_decay=params['epsilon_decay'],
                                  kappa=params.get('kappa', 1e-4))
        return {'policy': pi, 'Q': Q, 'model': model}


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Démonstration du projet RL')
    parser.add_argument('--env', choices=['lineworld', 'gridworld', 'rps',
                                         'montyhall1', 'montyhall2'],
                       help='Environnement à utiliser')
    parser.add_argument('--algo', choices=['policy_iteration', 'value_iteration',
                                          'mc_es', 'mc_on_policy', 'mc_off_policy',
                                          'sarsa', 'q_learning', 'expected_sarsa',
                                          'dyna_q', 'dyna_q_plus'],
                       help='Algorithme à utiliser')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Nombre d\'épisodes d\'entraînement')
    parser.add_argument('--interactive', action='store_true',
                       help='Mode interactif')

    args = parser.parse_args()

    demo = RLDemoSystem()

    if args.interactive or (not args.env or not args.algo):
        # Mode interactif
        print("\n" + "="*60)
        print("SYSTÈME DE DÉMONSTRATION - PROJET REINFORCEMENT LEARNING")
        print("="*60)

        while True:
            print("\n1. Entraîner un algorithme")
            print("2. Charger une politique sauvegardée")
            print("3. Jouer manuellement")
            print("4. Lister les politiques sauvegardées")
            print("5. Quitter")

            choice = input("\nVotre choix : ")

            if choice == '1':
                # Choisir environnement
                print("\nEnvironnements disponibles :")
                for i, env_name in enumerate(demo.environments.keys(), 1):
                    print(f"{i}. {env_name}")

                env_idx = int(input("Choisissez un environnement : ")) - 1
                env_name = list(demo.environments.keys())[env_idx]

                # Choisir algorithme
                print("\nAlgorithmes disponibles :")
                for i, algo_name in enumerate(demo.algorithms.keys(), 1):
                    print(f"{i}. {algo_name}")

                algo_idx = int(input("Choisissez un algorithme : ")) - 1
                algo_name = list(demo.algorithms.keys())[algo_idx]

                # Paramètres
                episodes = int(input(f"Nombre d'épisodes [{demo.default_params['num_episodes']}] : ")
                              or demo.default_params['num_episodes'])

                demo.train_and_demo(env_name, algo_name, num_episodes=episodes)

            elif choice == '2':
                # Charger une politique
                demo.policy_manager.list_policies()
                filename = input("\nNom du fichier (sans .pkl) : ")
                try:
                    policy_data = demo.policy_manager.load_policy(filename)
                    env = demo.environments[policy_data['environment']]()
                    demo._show_demo_menu(env, policy_data['policy'], policy_data)
                except Exception as e:
                    print(f"❌ Erreur : {e}")

            elif choice == '3':
                # Jouer manuellement
                print("\nEnvironnements disponibles :")
                for i, env_name in enumerate(demo.environments.keys(), 1):
                    print(f"{i}. {env_name}")

                env_idx = int(input("Choisissez un environnement : ")) - 1
                env_name = list(demo.environments.keys())[env_idx]
                env = demo.environments[env_name]()

                visualizer = EnvironmentVisualizer(env)
                visualizer.play_manual()

            elif choice == '4':
                demo.policy_manager.list_policies()

            elif choice == '5':
                print("\nAu revoir ! 👋")
                break

    else:
        # Mode ligne de commande
        demo.train_and_demo(args.env, args.algo, num_episodes=args.episodes)


if __name__ == "__main__":
    main()