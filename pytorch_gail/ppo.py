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
        
        # 處理環境觀察值的轉換
        try:
            # 檢查是否為列表並且包含數組和字典（特殊情況處理）
            if isinstance(obs, (list, tuple)) and any(isinstance(x, np.ndarray) for x in obs):
                # 尋找obs中的numpy數組元素並使用它
                for item in obs:
                    if isinstance(item, np.ndarray) and item.dtype != np.object_:
                        obs = item.astype(np.float32)
                        break
            # 如果是可直接展平的數組，將其轉換為浮點數
            elif isinstance(obs, np.ndarray):
                if obs.dtype == np.object_:
                    # 嘗試以遞迴方式處理複雜的觀察值結構
                    flat_obs = []
                    for item in obs.flatten():
                        if isinstance(item, (list, tuple, np.ndarray)):
                            flat_obs.extend([float(x) for x in item if not isinstance(x, dict)])
                        elif not isinstance(item, dict):  # 跳過字典類型
                            flat_obs.append(float(item))
                    obs = np.array(flat_obs, dtype=np.float32)
                else:
                    # 已經是數值陣列，直接轉換類型
                    obs = obs.astype(np.float32)
            else:
                # 嘗試直接將觀察值轉換為扁平浮點數數組
                try:
                    # 簡單情況：可以直接轉換
                    obs = np.array(obs, dtype=np.float32)
                except (ValueError, TypeError):
                    # 複雜情況：需要手動展平
                    flat_obs = []
                    
                    def flatten_recursive(item):
                        if isinstance(item, (list, tuple, np.ndarray)):
                            for subitem in item:
                                flatten_recursive(subitem)
                        elif not isinstance(item, dict):  # 跳過字典類型
                            flat_obs.append(float(item))
                    
                    flatten_recursive(obs)
                    obs = np.array(flat_obs, dtype=np.float32)
        except Exception as e:
            # 最後的應急方案：如果環境返回了難以處理的觀察值，嘗試模擬CartPole標準觀察值
            print(f"警告: 處理觀察值時出錯: {e}。嘗試使用替代方法...")
            try:
                # 檢測是否有標準CartPole觀察值
                if isinstance(obs, (list, tuple)) and len(obs) > 0:
                    for item in obs:
                        if isinstance(item, np.ndarray) and len(item) == 4:
                            obs = item.astype(np.float32)
                            break
                    else:
                        # 如果沒有找到適合的數組，創建一個零數組
                        obs = np.zeros(4, dtype=np.float32)
                else:
                    # 強制創建一個CartPole標準觀察格式
                    obs = np.zeros(4, dtype=np.float32)
            except Exception as e2:
                raise ValueError(f"無法將觀察值轉換為數值數組: {e}，且備選方案也失敗: {e2}。觀察值: {obs}")
        
        done = False
        
        for _ in range(self.n_steps):
            # 轉換為張量
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            # 獲取策略動作
            with torch.no_grad():
                value, action, log_prob = self.policy.act(obs_tensor)

            # 檢查動作是否在環境的動作空間內
            if not self.env.action_space.contains(action.cpu().numpy()):
                raise ValueError(f"動作 {action.cpu().numpy()} 超出環境的動作空間範圍 {self.env.action_space}")
                
            # 執行動作
            next_obs, reward, done, info = env.step(action.cpu().numpy())
            
            # 處理下一個觀察值的轉換，類似於上面的邏輯
            try:
                # 檢查是否為列表並且包含數組和字典（特殊情況處理）
                if isinstance(next_obs, (list, tuple)) and any(isinstance(x, np.ndarray) for x in next_obs):
                    # 尋找next_obs中的numpy數組元素並使用它
                    for item in next_obs:
                        if isinstance(item, np.ndarray) and item.dtype != np.object_:
                            next_obs = item.astype(np.float32)
                            break
                # 剩餘的處理邏輯與之前相同
                elif isinstance(next_obs, np.ndarray):
                    if next_obs.dtype == np.object_:
                        flat_next_obs = []
                        for item in next_obs.flatten():
                            if isinstance(item, (list, tuple, np.ndarray)):
                                flat_next_obs.extend([float(x) for x in item if not isinstance(x, dict)])
                            elif not isinstance(item, dict):  # 跳過字典類型
                                flat_next_obs.append(float(item))
                        next_obs = np.array(flat_next_obs, dtype=np.float32)
                    else:
                        next_obs = next_obs.astype(np.float32)
                else:
                    try:
                        next_obs = np.array(next_obs, dtype=np.float32)
                    except (ValueError, TypeError):
                        flat_next_obs = []
                        
                        def flatten_recursive(item):
                            if isinstance(item, (list, tuple, np.ndarray)):
                                for subitem in item:
                                    flatten_recursive(subitem)
                            elif not isinstance(item, dict):  # 跳過字典類型
                                flat_next_obs.append(float(item))
                        
                        flatten_recursive(next_obs)
                        next_obs = np.array(flat_next_obs, dtype=np.float32)
            except Exception as e:
                # 最後的應急方案：如果處理失敗，嘗試使用前一個觀察值
                print(f"警告: 處理下一個觀察值時出錯: {e}。嘗試使用替代方法...")
                try:
                    # 檢測是否有標準CartPole觀察值
                    if isinstance(next_obs, (list, tuple)) and len(next_obs) > 0:
                        for item in next_obs:
                            if isinstance(item, np.ndarray) and len(item) == 4:
                                next_obs = item.astype(np.float32)
                                break
                        else:
                            # 如果沒有找到適合的數組，使用前一個觀察值
                            next_obs = obs.copy()
                    else:
                        # 使用前一個觀察值
                        next_obs = obs.copy()
                except Exception as e2:
                    print(f"處理下一個觀察值的備選方案也失敗: {e2}，使用前一個觀察值")
                    next_obs = obs.copy()
            
            # 如果使用GAIL，使用判別器計算獎勵
            if reward_fn is not None:
                # 確保傳遞給 reward_fn 的參數格式正確
                reward = reward_fn(obs, action.cpu().numpy())

            # 檢查獎勵是否為有效數值
            if not np.isfinite(reward):
                raise ValueError(f"獎勵 {reward} 不是有效的數值。請檢查 reward_fn 的實現。")
                
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
                # 處理重置後的觀察值轉換，與上面初始化環境時的處理邏輯相同
                try:
                    # 檢查是否為列表並且包含數組和字典（特殊情況處理）
                    if isinstance(obs, (list, tuple)) and any(isinstance(x, np.ndarray) for x in obs):
                        # 尋找obs中的numpy數組元素並使用它
                        for item in obs:
                            if isinstance(item, np.ndarray) and item.dtype != np.object_:
                                obs = item.astype(np.float32)
                                break
                    elif isinstance(obs, np.ndarray):
                        if obs.dtype == np.object_:
                            flat_obs = []
                            for item in obs.flatten():
                                if isinstance(item, (list, tuple, np.ndarray)):
                                    flat_obs.extend([float(x) for x in item if not isinstance(x, dict)])
                                elif not isinstance(item, dict):  # 跳過字典類型
                                    flat_obs.append(float(item))
                            obs = np.array(flat_obs, dtype=np.float32)
                        else:
                            obs = obs.astype(np.float32)
                    else:
                        try:
                            obs = np.array(obs, dtype=np.float32)
                        except (ValueError, TypeError):
                            flat_obs = []
                            
                            def flatten_recursive(item):
                                if isinstance(item, (list, tuple, np.ndarray)):
                                    for subitem in item:
                                        flatten_recursive(subitem)
                                elif not isinstance(item, dict):  # 跳過字典類型
                                    flat_obs.append(float(item))
                            
                            flatten_recursive(obs)
                            obs = np.array(flat_obs, dtype=np.float32)
                except Exception as e:
                    # 最後的應急方案：如果環境返回了難以處理的觀察值，嘗試使用標準CartPole觀察值
                    print(f"警告: 處理重置後的觀察值時出錯: {e}。嘗試使用替代方法...")
                    try:
                        # 檢測是否有標準CartPole觀察值
                        if isinstance(obs, (list, tuple)) and len(obs) > 0:
                            for item in obs:
                                if isinstance(item, np.ndarray) and len(item) == 4:
                                    obs = item.astype(np.float32)
                                    break
                            else:
                                # 如果沒有找到適合的數組，創建一個零數組
                                obs = np.zeros(4, dtype=np.float32)
                        else:
                            # 強制創建一個CartPole標準觀察格式
                            obs = np.zeros(4, dtype=np.float32)
                    except Exception as e2:
                        raise ValueError(f"無法將重置後的觀察值轉換為數值數組: {e}，且備選方案也失敗: {e2}。觀察值: {obs}")
                
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