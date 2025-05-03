# pytorch_gail/adversary.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym


def logsigmoid(x):
    """
    PyTorch 版本的 logsigmoid 函數，等同於 tf.log(tf.sigmoid(x))
    """
    return -F.softplus(-x)


def logit_bernoulli_entropy(logits):
    """
    PyTorch 版本的 Bernoulli 熵計算
    """
    ent = (1.0 - torch.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class RunningMeanStd(nn.Module):
    """
    PyTorch 版本的 Running Mean Standard 計算
    """
    def __init__(self, shape=(), epsilon=1e-2):
        super(RunningMeanStd, self).__init__()
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))
        self.register_buffer("sum", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("sumsq", torch.zeros(shape, dtype=torch.float32) + epsilon)
        self.epsilon = epsilon
        self.shape = shape

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        new_count = self.count + batch_count

        self.sum += batch_mean * batch_count
        self.sumsq += (batch_var + batch_mean ** 2) * batch_count
        self.count = new_count

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return torch.sqrt(torch.maximum(
            torch.tensor(self.epsilon, device=self.count.device),
            self.sumsq / self.count - (self.sum / self.count) ** 2
        ))


class TransitionClassifier(nn.Module):
    """
    PyTorch 版本的 TransitionClassifier
    用於 GAIL 中的判別器
    """
    def __init__(self, observation_space, action_space, hidden_size, device='cuda', 
                 entcoeff=0.001, normalize=True):
        super(TransitionClassifier, self).__init__()
        
        self.device = device
        self.observation_shape = observation_space.shape
        self.entcoeff = entcoeff
        self.normalize = normalize
        
        if isinstance(action_space, gym.spaces.Box):
            # 連續動作空間
            self.discrete_actions = False
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = action_space.n
            self.discrete_actions = True
        else:
            raise ValueError("不支持的動作空間: {}".format(action_space))
            
        self.hidden_size = hidden_size
        
        # 創建網絡
        input_size = int(np.prod(self.observation_shape)) + (self.n_actions if not self.discrete_actions else self.n_actions)
        
        self.network = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        
        self.obs_rms = RunningMeanStd(shape=self.observation_shape) if normalize else None
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.to(device)

    def forward(self, obs, actions):
        # 處理觀測值
        if self.normalize:
            obs = (obs - self.obs_rms.mean) / self.obs_rms.std
        
        # 處理動作
        if self.discrete_actions:
            actions = F.one_hot(actions.long(), self.n_actions).float()
            
        # 合併觀測值和動作
        inputs = torch.cat([obs, actions], dim=1)
        
        return self.network(inputs)
        
    def get_reward(self, obs, actions):
        """
        使用觀測值和動作預測獎勵
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                
            if isinstance(actions, np.ndarray):
                actions_tensor = torch.FloatTensor(actions).to(self.device)
            else:
                actions_tensor = actions.to(self.device)
                
            if len(actions_tensor.shape) == 1:
                actions_tensor = actions_tensor.unsqueeze(0)
            elif len(actions_tensor.shape) == 0:
                # 單個離散動作
                actions_tensor = actions_tensor.unsqueeze(0)
                
            logits = self(obs_tensor, actions_tensor)
            reward = -torch.log(1 - torch.sigmoid(logits) + 1e-8)
            
            return reward.cpu().numpy()
            
    def train_discriminator(self, expert_obs, expert_actions, policy_obs, policy_actions):
        """
        訓練鑑別器區分專家和策略
        """
        # 轉換為PyTorch張量
        expert_obs = torch.FloatTensor(expert_obs).to(self.device)
        expert_actions = torch.FloatTensor(expert_actions).to(self.device)
        policy_obs = torch.FloatTensor(policy_obs).to(self.device)
        policy_actions = torch.FloatTensor(policy_actions).to(self.device)
        
        # 更新運行時統計
        if self.normalize:
            self.obs_rms.update(torch.cat([expert_obs, policy_obs], dim=0))
        
        # 計算專家和策略的 logits
        expert_logits = self(expert_obs, expert_actions)
        policy_logits = self(policy_obs, policy_actions)
        
        # 計算損失
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, 
            torch.ones(expert_logits.size()).to(self.device)
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, 
            torch.zeros(policy_logits.size()).to(self.device)
        )
        
        # 計算熵損失
        logits = torch.cat([policy_logits, expert_logits], dim=0)
        entropy = torch.mean(logit_bernoulli_entropy(logits))
        entropy_loss = -self.entcoeff * entropy
        
        # 總損失
        loss = expert_loss + policy_loss + entropy_loss
        
        # 計算準確率
        policy_acc = torch.mean((torch.sigmoid(policy_logits) < 0.5).float())
        expert_acc = torch.mean((torch.sigmoid(expert_logits) > 0.5).float())
        
        # 優化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'expert_loss': expert_loss.item(),
            'entropy': entropy.item(),
            'entropy_loss': entropy_loss.item(),
            'policy_acc': policy_acc.item(),
            'expert_acc': expert_acc.item(),
            'total_loss': loss.item()
        }