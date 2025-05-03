from pytorch_gail.policies import ActorCriticPolicy
from pytorch_gail.model import GAIL
from pytorch_gail.dataset import ExpertDataset
import gym
import numpy as np

# 創建環境
env = gym.make("CartPole-v1")

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

# 定義新的 reset 方法
def wrapped_reset():
    obs = original_reset()
    return np.array(obs).reshape(1, -1)[0]

# 替換環境的 reset 方法
env.reset = wrapped_reset

# 訓練模型
gail.learn(total_timesteps=100000)