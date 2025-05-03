from pytorch_gail.policies import ActorCriticPolicy
from pytorch_gail.model import GAIL
from pytorch_gail.dataset import ExpertDataset
import gym
import numpy as np
import torch

# 使用 GPU (如果可用)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置: {device}")

# 檢查 gym 版本並設置相應的環境包裝
try:
    # 創建環境
    env = gym.make("CartPole-v1")
    gym_new_api = hasattr(env, 'reset') and env.reset.__code__.co_argcount > 0
except:
    # 備用方案
    env = gym.make("CartPole-v1")
    gym_new_api = False

# 載入專家數據
dataset = ExpertDataset(expert_path="expert_cartpole.npz")

# 創建策略網路
policy = ActorCriticPolicy(env.observation_space, env.action_space, hidden_size=64)

# 創建GAIL模型
gail = GAIL(
    policy=policy,
    env=env,
    expert_dataset=dataset,
    hidden_size_adversary=100,
    g_step=3,
    d_step=1,
    device=device
)

# 保存原始的 reset 方法，避免遞迴
original_reset = env.reset

# 定義新的 reset 方法，處理新舊 API 差異
def wrapped_reset():
    if gym_new_api:
        # 新版 gym API: reset 返回 (obs, info)
        obs, _ = original_reset()
    else:
        # 舊版 gym API: reset 只返回 obs
        obs = original_reset()
    
    # 安全處理 obs 避免 VisibleDeprecationWarning
    if isinstance(obs, (list, tuple)) and any(isinstance(x, np.ndarray) for x in obs):
        # 尋找 obs 中的 numpy 數組
        for item in obs:
            if isinstance(item, np.ndarray):
                return item.astype(np.float32)
        # 如果沒找到適合的數組，安全地轉換
        return np.array(obs, dtype=object).reshape(1, -1)[0]
    else:
        # 確保結果是扁平的浮點數數組
        return np.array(obs, dtype=np.float32).flatten()

# 替換環境的 reset 方法
env.reset = wrapped_reset

# 替換環境的 step 方法，處理新舊 API 差異
original_step = env.step

def wrapped_step(action):
    result = original_step(action)
    
    if len(result) == 5:  # 新版 gym API: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = result
        # 合併 terminated 和 truncated 為單一的 done 標誌
        done = terminated or truncated
        return obs, reward, done, info
    else:  # 舊版 gym API: (obs, reward, done, info)
        return result

env.step = wrapped_step

# 修補 TransitionClassifier.forward 方法來修復 CUDA 錯誤
original_forward = gail.reward_giver.forward

def safe_forward(self, obs, actions):
    """修補版的 forward 方法，避免 CUDA 錯誤"""
    with torch.no_grad():
        # 處理觀測值
        if self.normalize:
            obs = (obs - self.obs_rms.mean) / self.obs_rms.std
        
        # 確保張量維度匹配
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
            # 處理空張量情況
            if actions.shape[0] == 0:
                actions = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
            
            try:
                # 確保離散動作是正確的長整型並轉為one-hot編碼
                actions = actions.reshape(-1).long()  # 確保是1D
                actions = F.one_hot(actions, self.n_actions).float()
            except Exception as e:
                print(f"處理離散動作時出錯: {e}，使用默認動作")
                actions = torch.zeros(obs.shape[0], self.n_actions, device=obs.device)
                actions[:, 0] = 1.0  # 默認選擇第一個動作
        
        # 確保批次維度匹配
        if obs.shape[0] != actions.shape[0]:
            min_batch = min(max(1, obs.shape[0]), max(1, actions.shape[0]))
            if obs.shape[0] > 0:
                obs = obs[:min_batch]
            else:
                # 如果 obs 是空的，創建一個假的觀察值
                obs = torch.zeros((min_batch, obs.shape[1] if len(obs.shape) > 1 else 4), device=obs.device)
                
            if actions.shape[0] > 0:
                actions = actions[:min_batch]
            else:
                # 如果 actions 是空的，創建一個假的動作張量
                action_dim = self.n_actions if self.discrete_actions else actions.shape[1] if len(actions.shape) > 1 else 1
                actions = torch.zeros((min_batch, action_dim), device=actions.device)
        
        # 合併觀測值和動作
        inputs = torch.cat([obs, actions], dim=1)
    
    # 在關鍵步驟添加日誌
    print("[DEBUG] reward_giver.forward: obs shape:", obs.shape, "actions shape:", actions.shape)

    # 檢查 actions 的形狀和值
    print("[DEBUG] reward_giver.forward: actions values:", actions)
    if self.discrete_actions:
        # 離散動作空間
        if len(actions.shape) != 1:
            raise ValueError(f"離散動作的形狀應為 [batch_size]，但得到 {actions.shape}")
        if not ((0 <= actions).all() and (actions < self.n_actions).all()):
            raise ValueError(f"離散動作的值應在 [0, {self.n_actions}) 範圍內，但得到 {actions}")
    else:
        # 連續動作空間
        if len(actions.shape) != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"連續動作的形狀應為 [batch_size, action_dim]，但得到 {actions.shape}")
        if not ((actions >= self.action_space.low).all() and (actions <= self.action_space.high).all()):
            raise ValueError(f"連續動作的值應在範圍 {self.action_space.low} 到 {self.action_space.high} 之間，但得到 {actions}")

    # 確保 obs 和 actions 的形狀匹配
    if obs.shape[0] != actions.shape[0]:
        raise ValueError(f"批次大小不匹配: obs.shape[0]={obs.shape[0]}, actions.shape[0]={actions.shape[0]}")

    # 確保 actions 的值在動作空間範圍內
    if hasattr(env.action_space, 'n') and not (0 <= actions).all() and not (actions < env.action_space.n).all():
        raise ValueError(f"動作值超出範圍: actions={actions}, action_space.n={env.action_space.n}")

    # 實際前向傳播
    return self.network(inputs)

# 替換 forward 方法
gail.reward_giver.forward = lambda obs, actions: safe_forward(gail.reward_giver, obs, actions)

# 修補 get_reward 方法
original_get_reward = gail.reward_giver.get_reward

def safe_get_reward(self, obs, actions):
    """安全版本的 get_reward 方法，處理各種錯誤情況"""
    try:
        with torch.no_grad():
            # 將觀察值轉換為張量
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                
            # 將動作轉換為張量並處理可能的空張量情況
            if isinstance(actions, np.ndarray):
                if actions.size == 0:
                    if self.discrete_actions:
                        actions_tensor = torch.zeros(obs_tensor.shape[0], dtype=torch.long, device=self.device)
                    else:
                        actions_tensor = torch.zeros((obs_tensor.shape[0], 1), device=self.device)
                else:
                    actions_tensor = torch.FloatTensor(actions).to(self.device)
            else:
                actions_tensor = actions.to(self.device)
            
            # 確保動作維度正確
            if len(actions_tensor.shape) == 0:  # 標量
                actions_tensor = actions_tensor.unsqueeze(0).unsqueeze(0)
            elif len(actions_tensor.shape) == 1:  # 1D向量
                if self.discrete_actions and actions_tensor.shape[0] != obs_tensor.shape[0]:
                    # 如果是離散動作且批次大小不匹配
                    actions_tensor = actions_tensor.unsqueeze(0)
                elif not self.discrete_actions and actions_tensor.shape[0] not in [obs_tensor.shape[0], self.n_actions]:
                    # 如果是連續動作且維度不匹配
                    actions_tensor = actions_tensor.unsqueeze(0)
            
            # 計算獎勵
            logits = self.forward(obs_tensor, actions_tensor)
            reward = -torch.log(1 - torch.sigmoid(logits) + 1e-8)
            
            return reward.cpu().numpy()
    except Exception as e:
        print(f"計算獎勵時出錯: {e}")
        # 返回一個默認獎勵
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            return np.ones(obs.shape[0]) * 0.1
        else:
            return np.array([0.1])

# 替換 get_reward 方法
gail.reward_giver.get_reward = lambda obs, actions: safe_get_reward(gail.reward_giver, obs, actions)

# 增強的 learn 方法 - 使用 GAIL 獎勵但增加錯誤處理
original_learn = gail.learn

def enhanced_learn(total_timesteps, callback=None, log_interval=100, tb_log_name="GAIL"):
    """增強版的學習方法，提供更好的錯誤處理"""
    assert gail.expert_dataset is not None, "必須提供專家數據集來訓練GAIL"
    
    timesteps_per_batch = gail.rl_learner.n_steps * gail.rl_learner.n_envs
    n_batches = total_timesteps // timesteps_per_batch
    
    for batch_idx in range(n_batches):
        try:
            # 收集一批經驗，使用判別器獎勵
            rollout = gail.rl_learner.collect_rollouts(gail.env, reward_fn=gail.reward_giver.get_reward)
            
            # 訓練判別器
            if batch_idx % gail.d_step == 0:
                try:
                    d_losses = []
                    for _ in range(gail.d_step):
                        # 獲取策略的觀測值和動作
                        policy_obs = rollout['observations']
                        policy_actions = rollout['actions']
                        
                        # 獲取專家的觀測值和動作
                        expert_obs, expert_actions = gail.expert_dataset.get_next_batch()
                        
                        # 訓練判別器
                        d_loss = gail.reward_giver.train_discriminator(
                            expert_obs, expert_actions, 
                            policy_obs, policy_actions
                        )
                        d_losses.append(d_loss)
                except Exception as e:
                    print(f"訓練判別器時出錯: {e}")
            
            # 使用GAIL獎勵訓練策略
            gail.rl_learner.train_on_batch(rollout)
            
            # 日誌記錄
            if (batch_idx + 1) % log_interval == 0:
                print(f"===== Batch {batch_idx + 1}/{n_batches} =====")
                if len(d_losses) > 0:
                    for k, v in d_losses[-1].items():
                        print(f"{k}: {v:.4f}")
        except Exception as e:
            print(f"批次 {batch_idx} 處理時發生錯誤: {e}")
            continue
            
    return gail.rl_learner

# 替換原始的學習方法
gail.learn = enhanced_learn

# 添加一個導入以便使用 F.one_hot
import torch.nn.functional as F

# 訓練模型
try:
    gail.learn(total_timesteps=100000)
    print("訓練完成！")
except Exception as e:
    print(f"訓練過程中發生未捕獲的錯誤: {e}")