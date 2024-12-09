import gymnasium as gym
import nnabla_rl.algorithms.qrsac as qrsac
from nnabla_rl.environments.wrappers.gymnasium import Gymnasium2GymWrapper
import nnabla_rl.hooks as hooks
from SaveRewardsWriter import FileWriter

def build_env(env_name, seed=None):
    env = gym.make(env_name, render_mode='human')
    env = Gymnasium2GymWrapper(env)
    env.seed(seed)
    return env

def main():
    env_name = "Humanoid-v5"
    env = build_env(env_name, seed=1002)

    results_folder = "./QRSAC/results/"
    rewards_writer = FileWriter(outdir=results_folder, file_prefix=env_name)

    # initialize hooks for logging
    iteration_num_hook = hooks.IterationNumHook(timing=100)
    iteration_state_hook = hooks.IterationStateHook(writer=rewards_writer, timing=100)

    config = qrsac.QRSACConfig(
        batch_size=128,
        environment_steps=1,
        gradient_steps=3,
        initial_temperature=0.1,
        fix_temperature=False,
        start_timesteps=200,
        replay_buffer_size=int(1e6),
        num_steps=5,
        num_quantiles=16,
        kappa=1.0
    )
    QRSAC_algo = qrsac.QRSAC(env, config=config)
    QRSAC_algo.set_hooks(hooks=[iteration_num_hook, iteration_state_hook])

    QRSAC_algo.train(env, total_iterations=int(1e6))

    env.close()

if __name__ == "__main__":
    main()
