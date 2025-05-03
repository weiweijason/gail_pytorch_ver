# pytorch_gail/ppo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions import Categorical, Normal


class PPO:
    """
    實現 Proximal Policy Optimization (PPO) 算法，用作GAIL的基礎RL方法
    """
    def __init__(self, policy, env, device='cuda', learning_rate=3e-4, 
                 n_steps=2048, batch_size=64, n_epochs=10, 
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, 
                 clip_range_vf=None, ent_coef=0.0, vf_coef=0.5, 
                 max_grad_norm=0.5, target_kl=None, n_envs=1):
        
        self.device = device
        self.policy = policy.to(device)
        self.env = env
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
    def collect_rollouts(self, env, reward_fn=None):
        """
        收集環境中的經驗
        
        :param env: 環境
        :param reward_fn: 可選的自定義獎勵函數
        """
        # 初始化緩衝區
        observations = []
        actions = []
        rewards = []
        values = []
        dones = []
        log_probs = []
        
        # 初始化環境
        obs = env.reset()
        
        # 確保觀察值是正確的形狀
        if isinstance(obs, (list, tuple)) and len(obs) == 0:
            raise ValueError("環境重置返回了空的觀察值。請確保環境正確初始化。")
        
        # 確保觀察值是numpy陣列並且有正確的維度和數據類型
        if isinstance(obs, np.ndarray) and obs.dtype == np.object_:
            # 將 object 類型轉換為 float32
            obs = np.array(obs, dtype=np.float32)
        else:
            obs = np.array(obs, dtype=np.float32).reshape(1, -1)[0] if not isinstance(obs, np.ndarray) or obs.shape == () else obs.astype(np.float32)
        
        done = False
        
        for _ in range(self.n_steps):
            # 轉換為張量
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            # 獲取策略動作
            with torch.no_grad():
                value, action, log_prob = self.policy.act(obs_tensor)
                
            # 執行動作
            next_obs, reward, done, info = env.step(action.cpu().numpy())
            
            # 確保next_obs也有正確的形狀和數據類型
            if isinstance(next_obs, np.ndarray) and next_obs.dtype == np.object_:
                next_obs = np.array(next_obs, dtype=np.float32)
            else:
                next_obs = np.array(next_obs, dtype=np.float32).reshape(1, -1)[0] if not isinstance(next_obs, np.ndarray) or next_obs.shape == () else next_obs.astype(np.float32)
            
            # 如果使用GAIL，使用判別器計算獎勵
            if reward_fn is not None:
                # 確保傳遞給reward_fn的觀察值是float32類型
                r_obs = obs.astype(np.float32) if isinstance(obs, np.ndarray) else np.array(obs, dtype=np.float32)
                r_action = action.cpu().numpy().astype(np.float32)
                reward = reward_fn(r_obs, r_action)
                
            # 儲存轉換
            observations.append(obs)
            actions.append(action.cpu().numpy())
            rewards.append(reward)
            values.append(value.cpu().numpy())
            dones.append(done)
            log_probs.append(log_prob.cpu().numpy())
            
            # 更新觀測值
            obs = next_obs
            
            # 如果回合結束，重置環境
            if done:
                obs = env.reset()
                # 確保重置後的觀察值也有正確的形狀和數據類型
                if isinstance(obs, np.ndarray) and obs.dtype == np.object_:
                    obs = np.array(obs, dtype=np.float32)
                else:
                    obs = np.array(obs, dtype=np.float32).reshape(1, -1)[0] if not isinstance(obs, np.ndarray) or obs.shape == () else obs.astype(np.float32)
                done = False
                
        # 計算優勢估計
        advantages = self._compute_advantages(
            rewards, values, dones
        )
        
        # 確保所有數據都是浮點型
        return {
            'observations': np.array(observations, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'values': np.array(values, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32),
            'log_probs': np.array(log_probs, dtype=np.float32),
            'advantages': advantages.astype(np.float32)
        }
        
    def _compute_advantages(self, rewards, values, dones):
        """
        計算廣義優勢估計 (GAE)
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = values[-1]
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * mask * last_advantage
            last_value = values[t]
            advantages[t] = last_advantage
            
        return advantages
        
    def train_on_batch(self, rollout):
        """
        在一批數據上訓練策略
        """
        observations = torch.FloatTensor(rollout['observations']).to(self.device)
        actions = torch.FloatTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)
        advantages = torch.FloatTensor(rollout['advantages']).to(self.device)
        returns = advantages + torch.FloatTensor(rollout['values']).to(self.device)
        
        # 標準化優勢
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 進行多個時期的訓練
        for _ in range(self.n_epochs):
            # 生成小批量
            indices = torch.randperm(len(observations))
            
            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]
                
                # 獲取小批量
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 評估動作和值
                values, log_probs, entropy = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                # 計算策略損失
                ratio = torch.exp(log_probs - batch_old_log_probs)
                policy_loss_1 = batch_advantages * ratio
                policy_loss_2 = batch_advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # 計算值損失
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = batch_old_values + torch.clamp(
                        values - batch_old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                # 計算熵損失
                entropy_loss = -entropy.mean()
                
                # 總損失
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # 優化
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()