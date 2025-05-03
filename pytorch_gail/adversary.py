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
        
        # 檢查空批次問題
        if actions.shape[0] == 0:
            # 如果動作張量是空的，使用一個默認動作張量
            if self.discrete_actions:
                actions = torch.zeros(obs.shape[0], 1, device=self.device)
            else:
                actions = torch.zeros(obs.shape[0], self.n_actions, device=self.device)
        
        # 確保張量維度匹配
        # 檢查並修正維度不匹配問題
        if len(obs.shape) != len(actions.shape):
            # 如果維度數量不同
            if len(actions.shape) == 3 and len(obs.shape) == 2:
                # 3D動作張量 -> 2D
                actions = actions.view(actions.shape[0], -1)
            elif len(obs.shape) == 3 and len(actions.shape) == 2:
                # 如果觀察值是3D但動作是2D
                obs = obs.view(obs.shape[0], -1)
        
        # 處理動作
        if self.discrete_actions:
            # 確保離散動作是正確的長整型並轉為one-hot編碼
            try:
                actions = actions.reshape(-1).long()  # 確保是1D
                actions = F.one_hot(actions, self.n_actions).float()
            except RuntimeError as e:
                # 如果轉換失敗，創建默認的one-hot編碼
                print(f"動作轉換錯誤: {e}，使用默認動作")
                actions = torch.zeros(obs.shape[0], self.n_actions, device=self.device)
                actions[:, 0] = 1.0  # 默認選擇第一個動作
            
        # 合併觀測值和動作前確保兩者形狀兼容
        # 如果批次維度不匹配，但觀察值非空
        if obs.shape[0] != actions.shape[0]:
            if obs.shape[0] > 0 and actions.shape[0] > 0:
                # 嘗試調整批次大小
                min_batch = min(obs.shape[0], actions.shape[0])
                obs = obs[:min_batch]
                actions = actions[:min_batch]
            elif obs.shape[0] > 0 and actions.shape[0] == 0:
                # 如果動作是空的但觀察值不是，創建與觀察值匹配的動作
                if self.discrete_actions:
                    actions = torch.zeros(obs.shape[0], self.n_actions, device=self.device)
                    actions[:, 0] = 1.0  # 默認選擇第一個動作
                else:
                    actions = torch.zeros(obs.shape[0], self.n_actions, device=self.device)
            elif obs.shape[0] == 0:
                # 如果觀察值是空的，返回一個零張量
                return torch.zeros(1, 1, device=self.device)
        
        # 合併觀測值和動作
        try:
            inputs = torch.cat([obs, actions], dim=1)
            return self.network(inputs)
        except RuntimeError as e:
            print(f"合併張量錯誤: {e}, obs.shape={obs.shape}, actions.shape={actions.shape}")
            # 返回一個零張量作為後備
            return torch.zeros(max(1, obs.shape[0]), 1, device=self.device)
        
    def get_reward(self, obs, actions):
        """
        使用觀測值和動作預測獎勵
        """
        with torch.no_grad():
            # 將觀察值轉換為張量
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                
            # 將動作轉換為張量
            if isinstance(actions, np.ndarray):
                actions_tensor = torch.FloatTensor(actions).to(self.device)
            else:
                actions_tensor = actions.to(self.device)
                
            # 確保動作是 2D 張量 [batch_size, action_dim]
            if len(actions_tensor.shape) == 0:  # 單個標量值
                actions_tensor = actions_tensor.unsqueeze(0).unsqueeze(0)
            elif len(actions_tensor.shape) == 1:  # 一維向量 [action_dim] 或 [batch_size]
                actions_tensor = actions_tensor.unsqueeze(0) if actions_tensor.shape[0] != obs_tensor.shape[0] else actions_tensor.unsqueeze(1)
            elif len(actions_tensor.shape) == 3:  # 3D 張量 [batch_size, extra_dim, action_dim]
                # 壓縮 3D 到 2D
                actions_tensor = actions_tensor.view(actions_tensor.shape[0], -1)
            
            # 確保批次維度匹配
            if actions_tensor.shape[0] != obs_tensor.shape[0]:
                if actions_tensor.shape[0] == 1:
                    # 廣播動作到所有觀察值
                    actions_tensor = actions_tensor.expand(obs_tensor.shape[0], -1)
                elif obs_tensor.shape[0] == 1:
                    # 廣播觀察值到所有動作
                    obs_tensor = obs_tensor.expand(actions_tensor.shape[0], -1)
                else:
                    # 如果完全不匹配，打印詳細信息供調試
                    print(f"維度不匹配: obs_tensor.shape={obs_tensor.shape}, actions_tensor.shape={actions_tensor.shape}")
                    # 截斷到較短長度或進行其他調整
                    min_batch = min(obs_tensor.shape[0], actions_tensor.shape[0])
                    obs_tensor = obs_tensor[:min_batch]
                    actions_tensor = actions_tensor[:min_batch]
            
            # 進行預測
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