from pytorch_gail.policies import ActorCriticPolicy
from pytorch_gail.model import GAIL
from pytorch_gail.dataset import ExpertDataset
import gym
import numpy as np

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
    device="cuda"
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

# 訓練模型
gail.learn(total_timesteps=100000)