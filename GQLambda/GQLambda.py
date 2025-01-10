
"""
GQ(λ) Implementation for Tic-Tac-Toe

This implementation uses the GQ(λ) algorithm to train an agent to play Tic-Tac-Toe.
The agent learns to play against either a random opponent or a greedy opponent that
tries to win when possible and block wins when threatened.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pettingzoo.classic import tictactoe_v3

class Agent:
    """Base class for RL agents"""
    def __init__(self):
        pass
        
    def select_action(self, observation):
        """Select action based on observation"""
        raise NotImplementedError
        
    def update(self, *args, **kwargs):
        """Update agent's parameters"""
        raise NotImplementedError

class RandomAgent(Agent):
    """Agent that selects random legal actions"""
    def select_action(self, observation):
        action_mask = observation['action_mask']
        legal_actions = np.where(action_mask == 1)[0]
        return np.random.choice(legal_actions) if len(legal_actions) > 0 else None
        
    def update(self, *args, **kwargs):
        pass

class GreedyAgent(Agent):
    """Agent that plays greedily to win and block opponent wins"""
    def select_action(self, observation):
        action_mask = observation['action_mask']
        legal_actions = np.where(action_mask == 1)[0]
        if len(legal_actions) == 0:
            return None
            
        board = observation['observation']
        # Check for winning moves
        for action in legal_actions:
            row, col = action // 3, action % 3
            test_board = board.copy()
            test_board[row, col, 0] = 1
            if self._check_win(test_board[:,:,0]):
                return action
                
        # Check for blocking moves
        for action in legal_actions:
            row, col = action // 3, action % 3
            test_board = board.copy()
            test_board[row, col, 1] = 1
            if self._check_win(test_board[:,:,1]):
                return action
                
        return np.random.choice(legal_actions)
        
    def _check_win(self, board_plane):
        """Check if given board plane has a winning line"""
        # Check rows and columns
        for i in range(3):
            if np.all(board_plane[i,:] == 1) or np.all(board_plane[:,i] == 1):
                return True
        # Check diagonals
        return np.all(np.diag(board_plane) == 1) or np.all(np.diag(np.fliplr(board_plane)) == 1)
        
    def update(self, *args, **kwargs):
        pass

class GQLambdaAgent(Agent):
    """Implementation of GQ(λ) learning algorithm"""
    def __init__(self, n, alpha=0.001, eta=0.5):
        super().__init__()
        # Initialize parameters: feature size, value weights, critic weights, eligibility trace
        self.n = n
        self.theta = np.zeros(n)  # Value weights
        self.w = np.zeros(n)      # Critic weights
        self.e = np.zeros(n)      # Eligibility trace
        self.alpha = alpha        # Learning rate for value weights
        self.eta = eta           # Learning rate for critic weights
        
    def select_action(self, observation, epsilon=0.2):
        """Select action using epsilon-greedy policy"""
        action_mask = observation['action_mask']
        legal_actions = np.where(action_mask == 1)[0]
        if len(legal_actions) == 0:
            return None
            
        # Random action with probability epsilon
        if np.random.random() < epsilon:
            return np.random.choice(legal_actions)
            
        # Greedy action with probability 1-epsilon
        phi = self._get_features(observation)
        q_values = np.array([self._get_value(phi, a) for a in range(9)])
        q_values[action_mask == 0] = float('-inf')  # Mask illegal actions
        return np.argmax(q_values)
        
    def update(self, phi, phi_bar, lambda_, gamma, R, rho, I):
        """Update parameters based on transition"""
        # Calculate TD error
        delta = R + gamma * np.dot(self.theta, phi_bar) - np.dot(self.theta, phi)
        
        # Update eligibility trace
        self.e = rho * self.e + I * phi
        
        # Pre-compute dot products
        dot_w_e = np.dot(self.w, self.e)
        dot_w_phi = np.dot(self.w, phi)
        
        # Update value weights and critic weights
        self.theta += self.alpha * (delta * self.e - gamma * (1 - lambda_) * dot_w_e * phi_bar)
        self.w += self.alpha * self.eta * (delta * self.e - dot_w_phi * phi)
        
        # Decay eligibility trace
        self.e *= gamma * lambda_
        
    def _get_features(self, observation):
        """Convert board state to feature vector"""
        board = observation['observation']
        features = np.zeros(self.n)
        for i in range(3):
            for j in range(3):
                if board[i,j,0] == 1:
                    features[i*3 + j] = 1    # Player 1's marks
                elif board[i,j,1] == 1:
                    features[i*3 + j] = -1   # Player 2's marks
        return features
        
    def _get_value(self, phi, action):
        """Modified Q-value calculation to consider full board state"""
        action_phi = phi.copy()  # Consider full board state instead of just the action position
        action_phi[action] *= 2  # Give more weight to the selected action
        return np.dot(self.theta, action_phi)

class TicTacToeEnv:
    """Wrapper for PettingZoo Tic-Tac-Toe environment"""
    def __init__(self): 
        self.env = tictactoe_v3.env()
        
    def get_features(self, observation):
        """Convert board state to feature vector"""
        board, features = observation['observation'], np.zeros(9)
        for i in range(3):
            for j in range(3):
                if board[i,j,0] == 1: features[i*3 + j] = 1  # Player 1's marks
                elif board[i,j,1] == 1: features[i*3 + j] = -1  # Player 2's marks
        return features
    
    def get_value(self, agent, phi, action):
        """Get Q-value for state-action pair"""
        action_phi = np.zeros_like(phi)
        action_phi[action] = phi[action]
        return np.dot(agent.theta, action_phi)
    
    def epsilon_greedy_policy(self, agent, phi, action_mask, epsilon=0.1):
        """Select action using epsilon-greedy policy"""
        legal_actions = np.where(action_mask == 1)[0]
        if len(legal_actions) == 0: return None
        # Random action with probability epsilon
        if np.random.random() < epsilon: return np.random.choice(legal_actions)
        # Greedy action with probability 1-epsilon
        q_values = np.array([self.get_value(agent, phi, a) for a in range(9)])
        q_values[action_mask == 0] = float('-inf')  # Mask illegal actions
        return np.argmax(q_values)
    
    def opponent_policy(self, observation, opponent_greedy=True):
        """Opponent strategy: either greedy or random"""
        if opponent_greedy:
            return GreedyAgent().select_action(observation)
        else:
            return RandomAgent().select_action(observation)
    
    def check_win(self, board_plane):
        """Check if given board plane has a winning line"""
        # Check rows and columns
        for i in range(3):
            if np.all(board_plane[i,:] == 1) or np.all(board_plane[:,i] == 1): return True
        # Check diagonals
        return np.all(np.diag(board_plane) == 1) or np.all(np.diag(np.fliplr(board_plane)) == 1)

class Trainer:
    """Training manager for the GQ(λ) agent"""
    def __init__(self, opponent_greedy=True): 
        self.episode_rewards = []
        self.avg_rewards = []
        self.opponent_greedy = opponent_greedy
        self.total_episodes = 0
        
    def train(self, agent, env, n_episodes, lambda_=0.7, gamma=0.95):
        """Train agent for specified number of episodes"""
        start_episode = self.total_episodes
        for episode in tqdm(range(n_episodes)):
            env.env.reset()
            episode_reward, last_observation = 0, None
            
            for agent_id in env.env.agent_iter():
                observation, reward, termination, truncation, info = env.env.last()
                
                # Enhanced reward structure
                if termination:
                    board = observation['observation']
                    if env.check_win(board[:,:,0]):  # Player 1 wins
                        reward = reward_weights[0] if agent_id == 'player_1' else reward_weights[1]
                    elif env.check_win(board[:,:,1]):  # Player 2 wins 
                        reward = reward_weights[1] if agent_id == 'player_1' else reward_weights[0]
                    elif np.all(board[:,:,0] + board[:,:,1]):  # Draw
                        reward = reward_weights[2]  # Small positive reward for draw
                else:
                    # Small negative reward for each move to encourage faster winning
                    reward = reward_weights[3]
                
                episode_reward += reward
                
                if termination or truncation:
                    action = None
                    if last_observation is not None:
                        phi = env.get_features(last_observation)
                        agent.update(phi, np.zeros_like(phi), lambda_, gamma, reward, 1.0, 1.0)
                else:
                    if agent_id == 'player_1':  # Agent's turn
                        phi = env.get_features(observation)
                        action = env.epsilon_greedy_policy(agent, phi, observation['action_mask'])
                        if last_observation is not None:
                            phi_last = env.get_features(last_observation)
                            agent.update(phi_last, phi, lambda_, gamma, reward, 1.0, 1.0)
                        last_observation = observation.copy()
                    else:  # Opponent's turn
                        action = env.opponent_policy(observation, self.opponent_greedy)
                        
                env.env.step(action)
                
            self.episode_rewards.append(episode_reward/2) # average reward as 2 updates per episode
            self.avg_rewards.append(np.mean(self.episode_rewards[-100:]))
            self.total_episodes += 1
            
    def plot_learning_curve(self, plot_title="", fileappend=""):
        """Plot and save training progress"""
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards, alpha=0.6, label='Episode Rewards')
        plt.plot(self.avg_rewards, label='Moving Average (100 episodes)')
        plt.xlabel('Episode'), plt.ylabel('Reward')
        title = 'GQ(λ) Learning Performance'
        if plot_title:
            title += f' - {plot_title}'
        plt.title(title), plt.legend(), plt.grid(True)
        filename = f'learning_curve_{time.strftime("%Y%m%d_%H%M%S")}'
        if fileappend:
            filename += f'_{fileappend.replace(" ", "_")}'
        plt.savefig(f'GQLambda/results/{filename}.png')
        plt.close()

if __name__ == "__main__":
    # Initialize environment, agent and trainer
    env = TicTacToeEnv()
    agent = GQLambdaAgent(n=9)
    reward_weights = [10, -10, 0.5, -0.1]
    reward_string = f"w +{reward_weights[0]},l -{reward_weights[1]},d {reward_weights[2]},step {reward_weights[3]}"
    # Start with random opponent first, then switch to greedy


    trainer1 = Trainer(opponent_greedy=False)
    trainer1.train(agent, env, n_episodes=2500)
    trainer1.opponent_greedy = True
    trainer1.train(agent, env, n_episodes=2500)
    trainer1.plot_learning_curve(plot_title=f"Random then Greedy rewards: {reward_string}", fileappend="random2500_then_greedy2500")

    trainer2 = Trainer(opponent_greedy=True)
    trainer2.train(agent, env, n_episodes=2500)
    trainer2.opponent_greedy = False
    trainer2.train(agent, env, n_episodes=2500)
    trainer2.plot_learning_curve(plot_title=f"Greedy then Random rewards: {reward_string}", fileappend="greedy2500_then_random2500")

    trainer3 = Trainer(opponent_greedy=False)
    trainer3.train(agent, env, n_episodes=5000)
    trainer3.opponent_greedy = True
    trainer3.train(agent, env, n_episodes=5000)
    trainer3.plot_learning_curve(plot_title=f"Random then Greedy rewards: {reward_string}", fileappend="random5000_then_greedy5000")

    trainer4 = Trainer(opponent_greedy=True)
    trainer4.train(agent, env, n_episodes=5000)
    trainer4.opponent_greedy = False
    trainer4.train(agent, env, n_episodes=5000)
    trainer4.plot_learning_curve(plot_title=f"Greedy then Random rewards: {reward_string}", fileappend="greedy5000_then_random5000")

    trainer5 = Trainer(opponent_greedy=False)
    trainer5.train(agent, env, n_episodes=10000)
    trainer5.plot_learning_curve(plot_title=f"Random Only rewards: {reward_string}", fileappend="random10000")
    
    trainer6 = Trainer(opponent_greedy=True)
    trainer6.train(agent, env, n_episodes=10000)
    trainer6.plot_learning_curve(plot_title=f"Greedy Only rewards: {reward_string}", fileappend="greedy10000")