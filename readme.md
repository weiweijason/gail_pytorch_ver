# PyTorch-GAIL

此專案提供了一個基於PyTorch的生成對抗式模仿學習（Generative Adversarial Imitation Learning, GAIL）實現，用於代替Stable Baselines中基於TensorFlow的原始實現。

## 專案概述

GAIL (Generative Adversarial Imitation Learning) 是一種結合了生成對抗網路(GAN)和強化學習的模仿學習演算法。此實現將原有的TensorFlow版本重新改寫為PyTorch，並優化以支持CUDA 12.4和RTXA6000 GPU。

## 主要特點

- **基於PyTorch**: 使用PyTorch深度學習框架從頭實現
- **現代化改進**: 使用更現代的PPO (Proximal Policy Optimization) 替代TRPO作為基礎RL算法
- **CUDA 12.4支援**: 完全支援最新的CUDA 12.4版本
- **高性能GPU優化**: 針對RTXA6000 GPU進行優化
- **簡化API**: 提供更直觀、更易於使用的界面
- **模塊化設計**: 更容易自定義和擴展

## 專案結構

```
pytorch_gail/
│
├── adversary.py       # 判別器/獎勵函數實現
├── main.py            # 主程式和訓練函數
├── model.py           # GAIL模型實現
├── policies.py        # 策略網路實現
├── ppo.py             # PPO實現（代替TRPO）
└── dataset/
    ├── __init__.py
    ├── dataset.py     # 專家數據集處理
    └── expert_cartpole.npz  # 範例專家數據
```

## 安裝指南

### 安裝要求

- Python 3.7+
- PyTorch 2.0+
- NumPy
- Gym
- CUDA 12.4（用於GPU加速，推薦用於RTXA6000）

### 安裝步驟

1. 克隆此專案:
   ```bash
   git clone <repository-url>
   cd pytorch-gail
   ```

2. 安裝相依套件:
   ```bash
   pip install torch numpy gym
   ```

3. 安裝支援CUDA 12.4的PyTorch版本:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

## 使用方法

### 基本用法

```python
from pytorch_gail.main import train_gail

# 訓練GAIL模型
model = train_gail(
    env_id="CartPole-v1",  # 環境ID
    expert_path="pytorch_gail/dataset/expert_cartpole.npz",  # 專家數據路徑
    total_timesteps=100000,  # 總訓練步數
    device="cuda"  # 使用GPU訓練
)
```

### 自定義專家數據

您可以使用原始Stable Baselines的工具生成專家數據，或提供自己的專家數據集：

```python
from stable_baselines.gail.dataset.record_expert import generate_expert_traj

# 使用預訓練的模型生成專家數據
generate_expert_traj(model, 'expert_data.npz', n_episodes=10)
```

### 高級使用

```python
from pytorch_gail.policies import ActorCriticPolicy
from pytorch_gail.model import GAIL
from pytorch_gail.dataset import ExpertDataset
import gym

# 創建環境
env = gym.make("CartPole-v1")

# 載入專家數據
dataset = ExpertDataset(expert_path="expert_data.npz")

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

# 訓練模型
gail.learn(total_timesteps=100000)
```

## 與TensorFlow版本的區別

此PyTorch實現相比原始Stable Baselines的TensorFlow版本有以下改進：

1. **架構**: 使用了PyTorch的動態計算圖，更易於調試和理解
2. **基礎RL算法**: 使用PPO替代TRPO作為基礎RL方法，通常能提供更好的性能和穩定性
3. **代碼組織**: 更簡潔、更現代化的代碼結構
4. **靈活性**: 更容易自定義和擴展，PyTorch的模塊化設計使代碼更易讀
5. **性能**: 針對CUDA 12.4和現代GPU（如RTXA6000）優化，提供更佳性能

## CUDA和GPU配置

### CUDA 12.4 配置

確保您的環境已正確安裝CUDA 12.4：

```bash
# 檢查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 檢查CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

### RTXA6000 GPU 優化

此實現已針對RTXA6000 GPU進行優化，但您可能需要調整以下參數以獲得最佳性能：

```python
# 設定GPU記憶體分配
import torch
torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%的GPU記憶體
```

## 引用

如果您在研究中使用了此實現，請引用原始GAIL論文和此專案：

```
@article{ho2016generative,
  title={Generative adversarial imitation learning},
  author={Ho, Jonathan and Ermon, Stefano},
  journal={Advances in neural information processing systems},
  volume={29},
  year={2016}
}
```

## 授權

此專案遵循與Stable Baselines相同的MIT授權。