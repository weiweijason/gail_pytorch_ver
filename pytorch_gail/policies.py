# pytorch_gail/policies.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import gym


class ActorCriticPolicy(nn.Module):
    """
    基本的演員-評論家策略網路
    """
    def __init__(self, observation_space, action_space, hidden_size=64):
        super(ActorCriticPolicy, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        input_dim = int(np.prod(observation_space.shape))
        
        # 共享特徵提取器
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # 動作網路和值網路
        if isinstance(action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.actor = nn.Linear(hidden_size, action_space.n)
            self.critic = nn.Linear(hidden_size, 1)
        elif isinstance(action_space, gym.spaces.Box):
            self.is_discrete = False
            self.actor_mean = nn.Linear(hidden_size, action_space.shape[0])
            self.actor_log_std = nn.Parameter(
                torch.zeros(action_space.shape[0], dtype=torch.float32)
            )
            self.critic = nn.Linear(hidden_size, 1)
        else:
            raise ValueError(f"不支持的動作空間: {action_space}")
    
    def forward(self, obs):
        features = self.features(obs)
        
        if self.is_discrete:
            action_logits = self.actor(features)
            values = self.critic(features)
            return action_logits, values
        else:
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std)
            values = self.critic(features)
            return action_mean, action_std, values
    
    def act(self, obs, deterministic=False):
        # 確保輸入是批次形式
        is_single_obs = len(obs.shape) == 1
        if is_single_obs:
            obs = obs.unsqueeze(0)  # 將單一觀察值轉換為批次(batch)形式
            
        if self.is_discrete:
            action_logits, values = self.forward(obs)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=1)
            else:
                distribution = Categorical(logits=action_logits)
                action = distribution.sample()
                
            log_prob = F.log_softmax(action_logits, dim=1).gather(1, action.unsqueeze(1))
            log_prob = log_prob.squeeze(1)
            
        else:
            action_mean, action_std, values = self.forward(obs)
            
            if deterministic:
                action = action_mean
            else:
                distribution = Normal(action_mean, action_std)
                action = distribution.sample()
                
            log_prob = -0.5 * (
                ((action - action_mean) / (action_std + 1e-8)) ** 2 + 
                2 * self.actor_log_std + np.log(2 * np.pi)
            ).sum(dim=1)
        
        # 如果輸入是單一觀察值，則移除批次維度
        if is_single_obs:
            values = values.squeeze(0)
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            
        return values, action, log_prob
        
    def evaluate_actions(self, obs, actions):
        if self.is_discrete:
            action_logits, values = self.forward(obs)
            distribution = Categorical(logits=action_logits)
            
            log_prob = distribution.log_prob(actions.long().squeeze())
            entropy = distribution.entropy()
            
        else:
            action_mean, action_std, values = self.forward(obs)
            distribution = Normal(action_mean, action_std)
            
            log_prob = distribution.log_prob(actions).sum(1)
            entropy = distribution.entropy().sum(1)
            
        return values.squeeze(), log_prob, entropy