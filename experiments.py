#!/usr/bin/env python3
"""
Framework d'expérimentation pour comparer les algorithmes RL.

Ce module permet de :
1. Tester tous les algorithmes sur tous les environnements
2. Étudier l'impact des hyperparamètres  
3. Analyser les performances et convergence
4. Générer des rapports automatiques

Usage:
    python experiments.py --env all --algo all --output results/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import argparse

# Imports des environnements
from envs import LineWorld, GridWorld, RPS, MontyHall1, MontyHall2

# Imports des algorithmes
from algos import (
    policy_iteration, value_iteration,
    mc_control_es, on_policy_first_visit_mc_control, off_policy_mc_control,
    sarsa, q_learning, expected_sarsa,
    dyna_q, dyna_q_plus,
    build_model_from_env
)

@dataclass
class ExperimentConfig:
    """Configuration d'une expérience."""
    env_name: str
    algo_name: str
    num_episodes: int = 1000
    num_runs: int = 10  # Répétitions pour moyenner les résultats
    hyperparams: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.hyperparams is None:
            self.hyperparams = {}

@dataclass 
class ExperimentResult:
    """Résultats d'une expérience."""
    config: ExperimentConfig
    final_score: float
    convergence_episode: int
    training_time: float
    policy: np.ndarray
    final_returns: List[float]  # Returns des derniers épisodes
    learning_curve: List[float]  # Performance au cours du temps
    hyperparams_used: Dict[str, Any]

class ExperimentRunner:
    """Gestionnaire d'expérimentations RL."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Registres des environnements et algorithmes
        self.environments = {
            "lineworld": LineWorld,
            "gridworld": GridWorld, 
            "rps": RPS,
            "montyhall1": MontyHall1,
            "montyhall2": MontyHall2,
        }
        
        self.algorithms = {

            "mc_es": self._run_mc_es,
            "mc_on_policy": self._run_mc_on_policy,
            "mc_off_policy": self._run_mc_off_policy,
            "sarsa": self._run_sarsa,
            "q_learning": self._run_q_learning,
            "expected_sarsa": self._run_expected_sarsa,
            "dyna_q": self._run_dyna_q,
            "dyna_q_plus": self._run_dyna_q_plus,
        }
        
        # Hyperparamètres par défaut pour chaque algorithme
        self.default_hyperparams = {
            "policy_iteration": {"gamma": 1.0, "theta": 1e-8},
            "value_iteration": {"gamma": 1.0, "theta": 1e-8},
            "mc_es": {"gamma": 1.0, "num_episodes": 5000},
            "mc_on_policy": {"gamma": 1.0, "num_episodes": 5000},
            "mc_off_policy": {"gamma": 1.0, "num_episodes": 5000},
            "sarsa": {"alpha": 0.1, "gamma": 1.0, "epsilon_start": 1.0},
            "q_learning": {"alpha": 0.1, "gamma": 1.0, "epsilon_start": 1.0},
            "expected_sarsa": {"alpha": 0.1, "gamma": 1.0, "epsilon_start": 1.0},
            "dyna_q": {"alpha": 0.1, "gamma": 1.0, "n_planning_steps": 50},
            "dyna_q_plus": {"alpha": 0.1, "gamma": 1.0, "n_planning_steps": 50, "kappa": 1e-4},
        }
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Exécute une expérience unique avec plusieurs runs."""
        print(f"Running {config.algo_name} on {config.env_name}...")
        
        all_scores = []
        all_convergence = []
        all_times = []
        all_curves = []
        final_policy = None
        
        # Hyperparamètres effectifs
        hyperparams = self.default_hyperparams.get(config.algo_name, {}).copy()
        hyperparams.update(config.hyperparams)
        
        for run in range(config.num_runs):
            start_time = time.time()
            
            # Exécution de l'algorithme
            result = self.algorithms[config.algo_name](
                config.env_name, hyperparams, config.num_episodes
            )
            
            end_time = time.time()
            
            # Évaluation de la politique résultante
            final_score = self._evaluate_policy(config.env_name, result["policy"])
            convergence = result.get("convergence_episode", config.num_episodes)
            learning_curve = result.get("learning_curve", [final_score])
            
            all_scores.append(final_score)
            all_convergence.append(convergence)
            all_times.append(end_time - start_time)
            all_curves.append(learning_curve)
            
            if run == 0:  # Garde la politique du premier run
                final_policy = result["policy"]
        
        # Statistiques agrégées
        mean_score = np.mean(all_scores)
        mean_convergence = np.mean(all_convergence)
        mean_time = np.mean(all_times)
        
        # Courbe d'apprentissage moyenne
        max_len = max(len(curve) for curve in all_curves)
        padded_curves = []
        for curve in all_curves:
            if len(curve) < max_len:
                # Extend avec la dernière valeur
                curve = curve + [curve[-1]] * (max_len - len(curve))
            padded_curves.append(curve)
        mean_learning_curve = np.mean(padded_curves, axis=0).tolist()
        
        return ExperimentResult(
            config=config,
            final_score=mean_score,
            convergence_episode=int(mean_convergence),
            training_time=mean_time,
            policy=final_policy,
            final_returns=all_scores,
            learning_curve=mean_learning_curve,
            hyperparams_used=hyperparams
        )
    
    def _evaluate_policy(self, env_name: str, policy: np.ndarray, num_episodes: int = 100) -> float:
        """Évalue une politique en simulant des épisodes."""
        env_class = self.environments[env_name]
        env = env_class()
        
        total_returns = []
        
        for _ in range(num_episodes):
            env.reset()
            episode_return = 0.0
            
            while not env.is_game_over():
                state = env.state()
                # Politique déterministe : argmax
                if policy.ndim == 2:
                    action = policy[state].argmax()
                else:
                    action = policy[state]
                
                prev_score = env.score()
                env.step(action)
                reward = env.score() - prev_score
                episode_return += reward
            
            total_returns.append(episode_return)
        
        return np.mean(total_returns)
    
    # ========== Wrappers des algorithmes ==========
    
    def _run_policy_iteration(self, env_name: str, params: dict, num_episodes: int) -> dict:
        env = self.environments[env_name]()
        P, R = build_model_from_env(env)
        policy, V = policy_iteration(P, R, **params)
        return {"policy": policy, "value_function": V}
    
    def _run_value_iteration(self, env_name: str, params: dict, num_episodes: int) -> dict:
        env = self.environments[env_name]()
        P, R = build_model_from_env(env)
        policy, V = value_iteration(P, R, **params)
        return {"policy": policy, "value_function": V}
    
    def _run_mc_es(self, env_name: str, params: dict, num_episodes: int) -> dict:
        env = self.environments[env_name]()
        policy, Q = mc_control_es(env, num_episodes, **params)
        return {"policy": policy, "Q": Q}
    
    def _run_mc_on_policy(self, env_name: str, params: dict, num_episodes: int) -> dict:
        env = self.environments[env_name]()
        policy, Q = on_policy_first_visit_mc_control(env, num_episodes, **params)
        return {"policy": policy, "Q": Q}
    
    def _run_mc_off_policy(self, env_name: str, params: dict, num_episodes: int) -> dict:
        env = self.environments[env_name]()
        policy, Q = off_policy_mc_control(env, num_episodes, **params)
        return {"policy": policy, "Q": Q}
    
    def _run_sarsa(self, env_name: str, params: dict, num_episodes: int) -> dict:
        env = self.environments[env_name]()
        policy, Q = sarsa(env, num_episodes, **params)
        return {"policy": policy, "Q": Q}
    
    def _run_q_learning(self, env_name: str, params: dict, num_episodes: int) -> dict:
        env = self.environments[env_name]()
        policy, Q = q_learning(env, num_episodes, **params)
        return {"policy": policy, "Q": Q}
    
    def _run_expected_sarsa(self, env_name: str, params: dict, num_episodes: int) -> dict:
        env = self.environments[env_name]()
        policy, Q = expected_sarsa(env, num_episodes, **params)
        return {"policy": policy, "Q": Q}
    
    def _run_dyna_q(self, env_name: str, params: dict, num_episodes: int) -> dict:
        env = self.environments[env_name]()
        policy, Q, model = dyna_q(env, num_episodes, **params)
        return {"policy": policy, "Q": Q, "model": model}
    
    def _run_dyna_q_plus(self, env_name: str, params: dict, num_episodes: int) -> dict:
        env = self.environments[env_name]()
        policy, Q, model = dyna_q_plus(env, num_episodes, **params)
        return {"policy": policy, "Q": Q, "model": model}
    
    def run_full_comparison(self, envs: List[str] = None, algos: List[str] = None) -> pd.DataFrame:
        """Exécute une comparaison complète algorithms × environments."""
        
        if envs is None:
            envs = list(self.environments.keys())
        if algos is None:
            algos = list(self.algorithms.keys())
        
        results = []
        
        for env_name in envs:
            for algo_name in algos:
                # Skip DP algorithms for complex environments
                if algo_name in ["policy_iteration", "value_iteration"] and env_name in ["montyhall2"]:
                    continue
                
                config = ExperimentConfig(
                    env_name=env_name,
                    algo_name=algo_name,
                    num_episodes=1000 if algo_name not in ["mc_es", "mc_on_policy", "mc_off_policy"] else 5000
                )
                
                try:
                    result = self.run_single_experiment(config)
                    results.append(result)
                except Exception as e:
                    print(f"Erreur avec {algo_name} sur {env_name}: {e}")
                    continue
        
        # Conversion en DataFrame pour analyse
        df_data = []
        for result in results:
            df_data.append({
                "Environment": result.config.env_name,
                "Algorithm": result.config.algo_name,
                "Final Score": result.final_score,
                "Convergence Episode": result.convergence_episode,
                "Training Time (s)": result.training_time,
                "Std Score": np.std(result.final_returns),
            })
        
        df = pd.DataFrame(df_data)
        
        # Sauvegarde
        df.to_csv(self.output_dir / "comparison_results.csv", index=False)
        
        return df, results
    
    def hyperparameter_study(self, env_name: str, algo_name: str, param_name: str, 
                           param_values: List[Any]) -> pd.DataFrame:
        """Étude de l'impact d'un hyperparamètre."""
        
        results = []
        
        for param_value in param_values:
            config = ExperimentConfig(
                env_name=env_name,
                algo_name=algo_name,
                hyperparams={param_name: param_value}
            )
            
            result = self.run_single_experiment(config)
            results.append({
                param_name: param_value,
                "Final Score": result.final_score,
                "Convergence Episode": result.convergence_episode,
                "Training Time": result.training_time,
                "Std Score": np.std(result.final_returns)
            })
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / f"hyperparam_{env_name}_{algo_name}_{param_name}.csv", index=False)
        
        return df

def generate_report(results_df: pd.DataFrame, results_list: List[ExperimentResult], 
                   output_dir: Path):
    """Génère un rapport d'analyse automatique."""
    
    # 1. Tableau de performance par environnement
    pivot_score = results_df.pivot(index="Algorithm", columns="Environment", values="Final Score")
    
    # 2. Graphiques de comparaison
    plt.figure(figsize=(15, 10))
    
    # Heatmap des performances
    plt.subplot(2, 2, 1)
    sns.heatmap(pivot_score, annot=True, fmt='.3f', cmap='viridis')
    plt.title("Performance des algorithmes par environnement")
    
    # Temps de convergence
    plt.subplot(2, 2, 2)
    pivot_conv = results_df.pivot(index="Algorithm", columns="Environment", values="Convergence Episode")
    sns.heatmap(pivot_conv, annot=True, fmt='.0f', cmap='plasma')
    plt.title("Épisodes jusqu'à convergence")
    
    # Temps d'entraînement
    plt.subplot(2, 2, 3)
    results_df.boxplot(column="Training Time (s)", by="Algorithm", ax=plt.gca())
    plt.title("Temps d'entraînement par algorithme")
    plt.xticks(rotation=45)
    
    # Performance globale
    plt.subplot(2, 2, 4)
    mean_scores = results_df.groupby("Algorithm")["Final Score"].mean().sort_values(ascending=False)
    mean_scores.plot(kind='bar')
    plt.title("Performance moyenne par algorithme")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Recommandations automatiques
    recommendations = []
    
    # Meilleur algorithme par environnement
    for env in pivot_score.columns:
        best_algo = pivot_score[env].idxmax()
        best_score = pivot_score[env].max()
        recommendations.append(f"**{env}**: {best_algo} (score: {best_score:.3f})")
    
    # Sauvegarde du rapport
    with open(output_dir / "analysis_report.md", "w") as f:
        f.write("# Rapport d'analyse des algorithmes RL\n\n")
        f.write("## Recommandations par environnement\n\n")
        for rec in recommendations:
            f.write(f"- {rec}\n")
        
        f.write("\n## Statistiques globales\n\n")
        f.write(f"- Nombre total d'expériences : {len(results_df)}\n")
        f.write(f"- Algorithme le plus rapide : {results_df.loc[results_df['Training Time (s)'].idxmin(), 'Algorithm']}\n")
        f.write(f"- Algorithme le plus performant : {mean_scores.index[0]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expérimentations RL")
    parser.add_argument("--env", nargs="+", default=["all"], help="Environnements à tester")
    parser.add_argument("--algo", nargs="+", default=["all"], help="Algorithmes à tester") 
    parser.add_argument("--output", default="results", help="Dossier de sortie")
    parser.add_argument("--hyperparam", action="store_true", help="Études d'hyperparamètres")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.output)
    
    # Expérimentation principale
    envs = list(runner.environments.keys()) if "all" in args.env else args.env
    algos = list(runner.algorithms.keys()) if "all" in args.algo else args.algo
    
    print("Démarrage des expérimentations...")
    df, results = runner.run_full_comparison(envs, algos)
    
    print("Génération du rapport...")
    generate_report(df, results, runner.output_dir)
    
    print(f"Résultats sauvegardés dans {runner.output_dir}")
    print("\nRésumé des performances:")
    print(df.groupby("Algorithm")["Final Score"].agg(["mean", "std"]).round(3))
