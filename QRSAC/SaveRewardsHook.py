import nnabla_rl.hook as hooks
import pickle

class EpisodeRewardLogger(hooks.Hook):
    def __init__(self, reward_file, env):
        self.reward_file = reward_file
        self.episode_rewards = []
        self.env = env

    def on_hook_called(self, agent):
        episode_reward = agent.evaluator.evaluate(self.env)
        self.episode_rewards.append(episode_reward)
        pickle.dump(self.episode_rewards, open(self.reward_file, 'wb'))
