import gymnasium as gym
import nnabla_rl.algorithms.qrsac as qrsac
from nnabla_rl.environments.wrappers.gymnasium import Gymnasium2GymWrapper
import nnabla_rl.hooks as hooks
from SaveRewardsHook import EpisodeRewardLogger

def build_env(env_name, seed=None):
    env = gym.make(env_name, render_mode='human')
    env = Gymnasium2GymWrapper(env)
    env.seed(seed)
    return env

def main():
    env_name = "Humanoid-v5"
    env = build_env(env_name, seed=1002)

    results_folder = "./QRSAC/results/"
    reward_file = results_folder + env_name

    # initialize hooks for logging
    iteration_num_hook = hooks.IterationNumHook(timing=100)
    episode_reward_hook = EpisodeRewardLogger(reward_file, env)

    config = qrsac.QRSACConfig(
        batch_size=128,
        environment_steps=1,
        gradient_steps=1,
        initial_temperature=0.1,
        fix_temperature=True,
        start_timesteps=200,
        replay_buffer_size=int(1e6),
        num_steps=3,
        num_quantiles=16,
        kappa=1.0
    )
    QRSAC_algo = qrsac.QRSAC(env, config=config)
    QRSAC_algo.set_hooks(hooks=[iteration_num_hook, episode_reward_hook])

    QRSAC_algo.train(env, total_iterations=int(1e6))

    env.close()

if __name__ == "__main__":
    main()
