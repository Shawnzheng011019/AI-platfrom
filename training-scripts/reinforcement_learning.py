#!/usr/bin/env python3
"""
Reinforcement Learning Training Script
Supports DQN, PPO, and A3C algorithms
"""

import os
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import gym

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience tuple for DQN
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dims=[128, 128]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dims=[128, 128]):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        shared_layers = []
        input_dim = state_size
        
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(input_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Actor head
        self.actor = nn.Linear(input_dim, action_size)
        
        # Critic head
        self.critic = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        shared_features = self.shared_network(x)
        action_probs = F.softmax(self.actor(shared_features), dim=-1)
        state_value = self.critic(shared_features)
        return action_probs, state_value


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        
        # Hyperparameters
        self.lr = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.target_update = config.get('target_update_freq', 100)
        
        # Networks
        hidden_dims = config.get('hidden_dims', [128, 128])
        self.q_network = DQNNetwork(state_size, action_size, hidden_dims).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Replay buffer
        buffer_size = config.get('buffer_size', 10000)
        self.memory = ReplayBuffer(buffer_size)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return np.argmax(q_values.cpu().data.numpy())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        states = torch.FloatTensor(batch.state).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)
        dones = torch.BoolTensor(batch.done).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


class PPOAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        
        # Hyperparameters
        self.lr = config.get('learning_rate', 0.0003)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # Network
        hidden_dims = config.get('hidden_dims', [128, 128])
        self.network = PPONetwork(state_size, action_size, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, value = self.network(state)
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def remember(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * next_value * (1 - self.dones[step]) - self.values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])
            next_value = self.values[step]
        
        return returns, advantages
    
    def update(self, next_value):
        returns, advantages = self.compute_gae(next_value)
        
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        action_probs, values = self.network(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Clear storage
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
        return total_loss.item()


class RLTrainer:
    def __init__(self, config):
        self.config = config
        self.env_name = config.get('environment', 'CartPole-v1')
        self.algorithm = config.get('algorithm', 'dqn').lower()
        
        # Create environment
        self.env = gym.make(self.env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Create agent
        if self.algorithm == 'dqn':
            self.agent = DQNAgent(self.state_size, self.action_size, config)
        elif self.algorithm == 'ppo':
            self.agent = PPOAgent(self.state_size, self.action_size, config)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        logger.info(f"RL Trainer initialized: {self.algorithm.upper()} on {self.env_name}")
        
    def train(self):
        """Train the RL agent"""
        num_episodes = self.config.get('num_episodes', 1000)
        max_steps = self.config.get('max_steps_per_episode', 1000)
        
        scores = []
        losses = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            episode_loss = 0
            
            for step in range(max_steps):
                if self.algorithm == 'dqn':
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    if len(self.agent.memory) > self.agent.batch_size:
                        loss = self.agent.replay()
                        if loss is not None:
                            episode_loss += loss
                    
                    if step % self.agent.target_update == 0:
                        self.agent.update_target_network()
                        
                elif self.algorithm == 'ppo':
                    action, log_prob, value = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.remember(state, action, reward, log_prob, value, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # PPO update at end of episode
            if self.algorithm == 'ppo':
                next_value = 0 if done else self.agent.act(state)[2]
                loss = self.agent.update(next_value)
                episode_loss = loss
            
            scores.append(total_reward)
            losses.append(episode_loss)
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                logger.info(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {getattr(self.agent, 'epsilon', 'N/A')}")
        
        # Save model and training history
        if self.algorithm == 'dqn':
            torch.save(self.agent.q_network.state_dict(), os.path.join(self.config['output_dir'], 'model.pth'))
        elif self.algorithm == 'ppo':
            torch.save(self.agent.network.state_dict(), os.path.join(self.config['output_dir'], 'model.pth'))
        
        # Save training history
        history = {
            'scores': scores,
            'losses': losses,
            'algorithm': self.algorithm,
            'environment': self.env_name
        }
        
        with open(os.path.join(self.config['output_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training completed. Final average score: {np.mean(scores[-100:]):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train Reinforcement Learning Agent')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    trainer = RLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
