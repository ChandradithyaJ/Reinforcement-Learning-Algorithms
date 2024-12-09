import gymnasium as gym
import nnabla_rl.algorithms.qrsac as qrsac
from nnabla_rl.environments.wrappers.gymnasium import Gymnasium2GymWrapper
import nnabla_rl.hooks as hooks

def build_env(seed=None):
    env = gym.make('ALE/KungFuMaster-v5', render_mode='human')
    env = Gymnasium2GymWrapper(env)
    env.seed(seed)
    return env

def main():
    env = build_env(seed=1002)

    # initialize hooks for logging
    iteration_num_hook = hooks.IterationNumHook(timing=100)
    save_snapshot_hook = hooks.SaveSnapshotHook("./QRSAC/results/ALE-KungFuMasterv5-Results", timing=1000)

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
    QRSAC_algo.set_hooks(hooks=[iteration_num_hook, save_snapshot_hook])

    QRSAC_algo.train(env, total_iterations=int(1e6))

    env.close()

if __name__ == "__main__":
    main()
