# pytorch_gail/main.py
import torch
import gym
from .policies import ActorCriticPolicy
from .model import GAIL
from .dataset import ExpertDataset


def train_gail(env_id, expert_path, total_timesteps=1000000, 
               log_interval=1000, device='cuda'):
    """
    訓練GAIL模型
    
    :param env_id: 環境ID
    :param expert_path: 專家數據路徑
    :param total_timesteps: 總時間步數
    :param log_interval: 日誌間隔
    :param device: 設備 ('cuda' 或 'cpu')
    """
    # 檢查CUDA是否可用
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU代替")
        device = 'cpu'
    elif device == 'cuda':
        # 顯示CUDA版本和GPU信息
        print(f"使用CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 創建環境
    env = gym.make(env_id)
    
    # 載入專家數據
    expert_dataset = ExpertDataset(expert_path=expert_path)
    
    # 創建策略
    policy = ActorCriticPolicy(env.observation_space, env.action_space)
    
    # 創建GAIL模型
    gail_model = GAIL(
        policy=policy,
        env=env,
        expert_dataset=expert_dataset,
        device=device
    )
    
    # 訓練模型
    gail_model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        tb_log_name="gail"
    )
    
    return gail_model