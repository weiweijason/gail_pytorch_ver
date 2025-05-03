# pytorch_gail/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

from .ppo import PPO  # 假設我們使用PPO代替TRPO
from .adversary import TransitionClassifier


class GAIL:
    """
    PyTorch 版本的 Generative Adversarial Imitation Learning (GAIL)
    """
    def __init__(self, policy, env, expert_dataset=None,
                 hidden_size_adversary=100, adversary_entcoeff=1e-3,
                 g_step=3, d_step=1, d_stepsize=3e-4, 
                 device='cuda', **kwargs):
        
        self.env = env
        self.policy = policy  # 策略網絡
        self.device = device
        
        # 不使用TRPO，而是使用更現代的PPO算法作為基礎RL方法
        self.rl_learner = PPO(policy, env, device=device, **kwargs)
        
        self.using_gail = True
        self.expert_dataset = expert_dataset
        self.g_step = g_step  # 生成器（策略）步數
        self.d_step = d_step  # 判別器步數
        self.d_stepsize = d_stepsize  # 判別器學習率
        self.hidden_size_adversary = hidden_size_adversary  # 判別器隱藏層大小
        self.adversary_entcoeff = adversary_entcoeff  # 判別器熵係數
        
        # 建立判別器
        self.reward_giver = TransitionClassifier(
            env.observation_space,
            env.action_space,
            hidden_size_adversary,
            device=device,
            entcoeff=adversary_entcoeff,
            normalize=True
        )
        
        # 設置數據加載器
        if self.expert_dataset is not None:
            self._initialize_dataset()
            
    def _initialize_dataset(self):
        """初始化專家數據加載器"""
        batch_size = self.rl_learner.batch_size // self.d_step
        self.expert_dataset.init_dataloader(batch_size)
        
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="GAIL"):
        """
        學習GAIL模型
        
        :param total_timesteps: 總時間步數
        :param callback: 回調函數
        :param log_interval: 日誌間隔
        :param tb_log_name: TensorBoard日誌名稱
        """
        assert self.expert_dataset is not None, "必須提供專家數據集來訓練GAIL"
        
        timesteps_per_batch = self.rl_learner.n_steps * self.rl_learner.n_envs
        n_batches = total_timesteps // timesteps_per_batch
        
        for batch_idx in range(n_batches):
            # 收集一批經驗
            rollout = self.rl_learner.collect_rollouts(self.env, reward_fn=self.reward_giver.get_reward)
            
            # 訓練判別器
            if batch_idx % self.d_step == 0:
                d_losses = []
                for _ in range(self.d_step):
                    # 獲取策略的觀測值和動作
                    policy_obs = rollout['observations']
                    policy_actions = rollout['actions']
                    
                    # 獲取專家的觀測值和動作
                    expert_obs, expert_actions = self.expert_dataset.get_next_batch()
                    
                    # 訓練判別器
                    d_loss = self.reward_giver.train_discriminator(
                        expert_obs, expert_actions, 
                        policy_obs, policy_actions
                    )
                    d_losses.append(d_loss)
            
            # 使用GAIL獎勵訓練策略
            self.rl_learner.train_on_batch(rollout)
            
            # 日誌記錄
            if (batch_idx + 1) % log_interval == 0:
                print(f"===== Batch {batch_idx + 1}/{n_batches} =====")
                if len(d_losses) > 0:
                    for k, v in d_losses[-1].items():
                        print(f"{k}: {v:.4f}")
                        
        return self.rl_learner