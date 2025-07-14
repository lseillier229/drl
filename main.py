from envs.lineworld import LineWorld
from envs.gridworld import GridWorld
from algos.monte_carlo import mc_control_es, off_policy_mc_control

pi_star, Q = off_policy_mc_control(GridWorld(), num_episodes=20_000, gamma=1.0)
print("Politique optimale (greedy) :", pi_star.argmax(axis=1))
