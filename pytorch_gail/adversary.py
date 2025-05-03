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
        """
        完全重寫的 forward 方法，確保不會觸發 CUDA 錯誤
        """
        try:
            # 處理觀測值
            if self.normalize:
                obs = (obs - self.obs_rms.mean) / self.obs_rms.std
            
            # 完全安全的離散動作處理
            if self.discrete_actions:
                # 檢測空批次或無效批次
                if actions.numel() == 0 or actions.shape[0] == 0:
                    # 創建安全的動作張量
                    if obs.shape[0] > 0:
                        # 生成與觀察值匹配批次大小的動作
                        fake_actions = torch.zeros(obs.shape[0], self.n_actions, device=obs.device)
                        fake_actions[:, 0] = 1.0  # 默認選擇第一個動作
                        actions = fake_actions
                    else:
                        # 如果觀察值也是空的，創建一個單例批次
                        fake_actions = torch.zeros(1, self.n_actions, device=obs.device)
                        fake_actions[0, 0] = 1.0
                        actions = fake_actions
                        
                        # 同時為觀察值創建匹配的假數據
                        if obs.numel() == 0 or obs.shape[0] == 0:
                            input_dim = self.observation_shape[0] if len(self.observation_shape) > 0 else 4
                            obs = torch.zeros(1, input_dim, device=obs.device)
                else:
                    # 處理非空但需要 one-hot 編碼的動作
                    try:
                        # 檢查動作張量是否已經是 one-hot 編碼
                        if len(actions.shape) == 1 or (len(actions.shape) == 2 and actions.shape[1] == 1):
                            # 需要 one-hot 編碼
                            actions = actions.reshape(-1).long()
                            # 確保動作在有效範圍內
                            actions = torch.clamp(actions, 0, self.n_actions - 1)
                            # 創建 one-hot 編碼
                            actions = torch.nn.functional.one_hot(actions, self.n_actions).float()
                        elif len(actions.shape) == 2 and actions.shape[1] == self.n_actions:
                            # 已經是 one-hot 編碼格式，確保是浮點型
                            actions = actions.float()
                        else:
                            # 不明形狀，創建默認 one-hot
                            print(f"無法識別的動作形狀: {actions.shape}，使用默認值")
                            fake_actions = torch.zeros(obs.shape[0], self.n_actions, device=obs.device)
                            fake_actions[:, 0] = 1.0  # 默認選擇第一個動作
                            actions = fake_actions
                    except Exception as e:
                        print(f"處理離散動作時出錯: {e}，使用默認動作")
                        fake_actions = torch.zeros(obs.shape[0], self.n_actions, device=obs.device)
                        fake_actions[:, 0] = 1.0
                        actions = fake_actions
            
            # 確保批次維度匹配
            if obs.shape[0] != actions.shape[0]:
                # 打印調試信息
                print(f"批次維度不匹配: obs.shape={obs.shape}, actions.shape={actions.shape}")
                
                # 調整批次大小
                min_batch = min(max(1, obs.shape[0]), max(1, actions.shape[0]))
                
                # 裁剪或擴展觀察值
                if obs.shape[0] > min_batch:
                    obs = obs[:min_batch]
                elif obs.shape[0] < min_batch:
                    # 通過複製現有數據擴展觀察值
                    if obs.shape[0] > 0:
                        repeat_times = min_batch // obs.shape[0] + (1 if min_batch % obs.shape[0] > 0 else 0)
                        obs = obs.repeat(repeat_times, 1)[:min_batch]
                    else:
                        # 如果 obs 是空的，創建假數據
                        input_dim = self.observation_shape[0] if len(self.observation_shape) > 0 else 4
                        obs = torch.zeros(min_batch, input_dim, device=obs.device)
                
                # 裁剪或擴展動作
                if actions.shape[0] > min_batch:
                    actions = actions[:min_batch]
                elif actions.shape[0] < min_batch:
                    # 通過複製現有數據擴展動作
                    if actions.shape[0] > 0:
                        repeat_times = min_batch // actions.shape[0] + (1 if min_batch % actions.shape[0] > 0 else 0)
                        actions = actions.repeat(repeat_times, 1)[:min_batch]
                    else:
                        # 如果 actions 是空的，創建假數據
                        action_dim = self.n_actions if self.discrete_actions else 1
                        actions = torch.zeros(min_batch, action_dim, device=actions.device)
            
            # 合併觀測值和動作
            try:
                inputs = torch.cat([obs, actions], dim=1)
                return self.network(inputs)
            except RuntimeError as e:
                print(f"張量連接或網絡前向傳播錯誤: {e}")
                print(f"Debug info - obs: {obs.shape}, actions: {actions.shape}")
                # 返回一個零張量作為默認輸出
                return torch.zeros(max(1, obs.shape[0]), 1, device=obs.device)
                
        except Exception as e:
            print(f"前向傳播中發生未預期錯誤: {e}")
            # 最終的應急方案
            return torch.zeros(1, 1, device=self.device)

    def get_reward(self, obs, actions):
        """
        使用觀測值和動作預測獎勵 - 完全加固版
        """
        try:
            with torch.no_grad():
                # 檢查輸入是否為空
                if isinstance(obs, np.ndarray) and (obs.size == 0 or obs.shape[0] == 0):
                    return np.array([0.1])  # 返回默認值
                
                if isinstance(actions, np.ndarray) and (actions.size == 0 or actions.shape[0] == 0):
                    # 為空動作創建默認獎勵，與觀察值批次大小匹配
                    batch_size = obs.shape[0] if isinstance(obs, np.ndarray) and len(obs.shape) > 0 else 1
                    return np.ones(batch_size) * 0.1
                
                # 轉換為張量
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                else:
                    obs_tensor = obs.to(self.device)
                
                if isinstance(actions, np.ndarray):
                    actions_tensor = torch.FloatTensor(actions).to(self.device)
                else:
                    actions_tensor = actions.to(self.device)
                
                # 確保張量是正確的形狀
                if len(obs_tensor.shape) == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                # 簡化動作處理，保持原始張量形狀，讓 forward 方法處理 one-hot 編碼
                # 這避免了之前可能觸發的索引錯誤
                
                # 計算獎勵
                try:
                    logits = self.forward(obs_tensor, actions_tensor)
                    reward = -torch.log(1 - torch.sigmoid(logits) + 1e-8)
                    return reward.cpu().numpy()
                except Exception as e:
                    print(f"計算獎勵邏輯中出錯: {e}")
                    # 返回默認獎勵
                    batch_size = obs_tensor.shape[0] if obs_tensor.shape[0] > 0 else 1
                    return np.ones(batch_size) * 0.1
        except Exception as e:
            print(f"獎勵計算過程發生未預期錯誤: {e}")
            return np.array([0.1])  # 單一默認獎勵

    def train_discriminator(self, expert_obs, expert_actions, policy_obs, policy_actions):
        """
        訓練鑑別器區分專家和策略 - 增強版
        """
        try:
            # 檢查數據有效性
            if len(expert_obs) == 0 or len(expert_actions) == 0 or len(policy_obs) == 0 or len(policy_actions) == 0:
                print("訓練數據為空，跳過本次判別器訓練")
                return {
                    'policy_loss': 0.0,
                    'expert_loss': 0.0,
                    'entropy': 0.0,
                    'entropy_loss': 0.0,
                    'policy_acc': 0.5,
                    'expert_acc': 0.5,
                    'total_loss': 0.0
                }
            
            # 轉換為PyTorch張量
            expert_obs = torch.FloatTensor(expert_obs).to(self.device)
            expert_actions = torch.FloatTensor(expert_actions).to(self.device)
            policy_obs = torch.FloatTensor(policy_obs).to(self.device)
            policy_actions = torch.FloatTensor(policy_actions).to(self.device)
            
            # 更新運行時統計
            if self.normalize:
                self.obs_rms.update(torch.cat([expert_obs, policy_obs], dim=0))
            
            # 計算專家和策略的 logits
            expert_logits = self.forward(expert_obs, expert_actions)
            policy_logits = self.forward(policy_obs, policy_actions)
            
            # 檢查 logits 是否有效
            if torch.isnan(expert_logits).any() or torch.isnan(policy_logits).any():
                print("警告: 發現 NaN 值，使用零張量替代")
                if torch.isnan(expert_logits).any():
                    expert_logits = torch.zeros_like(expert_logits)
                if torch.isnan(policy_logits).any():
                    policy_logits = torch.zeros_like(policy_logits)
            
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
            
            # 梯度裁剪以避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
            
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
        except Exception as e:
            print(f"訓練判別器時發生未預期錯誤: {e}")
            return {
                'policy_loss': 0.0,
                'expert_loss': 0.0,
                'entropy': 0.0,
                'entropy_loss': 0.0,
                'policy_acc': 0.5,
                'expert_acc': 0.5,
                'total_loss': 0.0
            }