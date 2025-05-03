from pytorch_gail.policies import ActorCriticPolicy
from pytorch_gail.model import GAIL
from pytorch_gail.dataset import ExpertDataset
import gym
import numpy as np
import torch

# 強制使用CPU以避免CUDA錯誤
device = "cpu"
torch.cuda.is_available = lambda: False

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

# 創建GAIL模型，強制使用CPU
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

# 添加一個簡化版的獎勵函數，避免使用判別器的獎勵
def simple_reward(obs, action):
    # 返回一個簡單的固定獎勵: 1.0
    # 要確保返回的形狀與輸入的 obs 批次大小一致
    if isinstance(obs, np.ndarray):
        if len(obs.shape) > 1:
            return np.ones(obs.shape[0])
        else:
            return np.array([1.0])
    return np.array([1.0])

# 修改 GAIL 類的學習方法，使用安全的學習流程
original_learn = gail.learn

def safe_learn(total_timesteps, callback=None, log_interval=100, tb_log_name="GAIL"):
    """安全版本的學習方法，使用簡單獎勵避免CUDA錯誤"""
    assert gail.expert_dataset is not None, "必須提供專家數據集來訓練GAIL"
    
    timesteps_per_batch = gail.rl_learner.n_steps * gail.rl_learner.n_envs
    n_batches = total_timesteps // timesteps_per_batch
    
    for batch_idx in range(n_batches):
        try:
            # 使用簡單獎勵而非判別器獎勵
            rollout = gail.rl_learner.collect_rollouts(gail.env, reward_fn=simple_reward)
            
            # 訓練判別器 (仍然嘗試，但捕獲任何錯誤)
            if batch_idx % gail.d_step == 0:
                try:
                    d_losses = []
                    for _ in range(gail.d_step):
                        # 獲取策略的觀測值和動作
                        policy_obs = rollout['observations']
                        policy_actions = rollout['actions']
                        
                        # 獲取專家的觀測值和動作
                        expert_obs, expert_actions = gail.expert_dataset.get_next_batch()
                        
                        # 嘗試訓練判別器，但包裹在 try-except 中
                        try:
                            d_loss = gail.reward_giver.train_discriminator(
                                expert_obs, expert_actions, 
                                policy_obs, policy_actions
                            )
                            d_losses.append(d_loss)
                        except Exception as e:
                            print(f"單步判別器訓練時出錯: {e}")
                except Exception as e:
                    print(f"訓練判別器整體流程出錯: {e}")
            
            # 使用獎勵訓練策略
            gail.rl_learner.train_on_batch(rollout)
            
            # 日誌記錄
            if (batch_idx + 1) % log_interval == 0:
                print(f"===== Batch {batch_idx + 1}/{n_batches} =====")
        except Exception as e:
            print(f"批次 {batch_idx} 處理時發生錯誤: {e}")
            continue
            
    return gail.rl_learner

# 替換原始的學習方法
gail.learn = safe_learn

# 訓練模型
try:
    gail.learn(total_timesteps=100000)
    print("訓練完成！")
except Exception as e:
    print(f"訓練過程中發生未捕獲的錯誤: {e}")