import os
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from gym_pybullet_drones.envs.CurriculumHoverAviaryV3 import CurriculumHoverAviaryV3

STAGE1_MODEL_PATH = os.path.join("curriculum_checkpoints_v2", "v2_stage1_4m_ppo_model.zip")
STAGE2_CHECKPOINT_DIR = os.path.join("curriculum_checkpoints_v3", "stage2_ckpts")
STAGE2_LOG_DIR = os.path.join("logs_v3", "stage2_ppo")
FINAL_SAVE_PATH = os.path.join("curriculum_checkpoints_v3", "v3_stage2_3m_ppo_model.zip")

os.makedirs(STAGE2_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(STAGE2_LOG_DIR, exist_ok=True)


class ActionShapeAdapter(gym.Wrapper):
    def __init__(self, env, target_act_shape=(1, 4)):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=target_act_shape, dtype=np.float32
        )

    def step(self, action):
        return self.env.step(action)


class ActionNoiseDecayCallback(BaseCallback):
    """Decays action std from 0.40 to 0.15 between 500k and 2.5M timesteps."""
    def __init__(self, start_step=500_000, end_step=2_500_000, initial_std=0.40, final_std=0.15, verbose=0):
        super().__init__(verbose)
        self.start_step = start_step
        self.end_step = end_step
        self.initial_log_std = np.log(initial_std)
        self.final_log_std = np.log(final_std)

    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        if current_step < self.start_step:
            target_log_std = self.initial_log_std
        elif current_step >= self.end_step:
            target_log_std = self.final_log_std
        else:
            progress = (current_step - self.start_step) / (self.end_step - self.start_step)
            target_log_std = self.initial_log_std + progress * (self.final_log_std - self.initial_log_std)

        with torch.no_grad():
            self.model.policy.log_std.fill_(target_log_std)

        return True


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def make_stage2_env(rank):
    def _init():
        time.sleep(rank * 0.1)
        base_env = CurriculumHoverAviaryV3(gui=False, stage=2)
        adapter_env = ActionShapeAdapter(base_env, target_act_shape=(1, 4))
        return Monitor(adapter_env)
    return _init


def train_v3_stage2():
    NUM_ENVS = 8
    TOTAL_TIMESTEPS = 3_000_000

    if not os.path.exists(STAGE1_MODEL_PATH):
        raise FileNotFoundError(f"Stage 1 model missing at: {STAGE1_MODEL_PATH}")

    print("=" * 70)
    print("   STARTING V3 STAGE 2 TRAINING (CRITIC RE-INITIALIZED & LR BALANCED)")
    print("=" * 70)

    try:
        env = SubprocVecEnv([make_stage2_env(i) for i in range(NUM_ENVS)])
    except Exception:
        env = DummyVecEnv([make_stage2_env(i) for i in range(NUM_ENVS)])

    # Load Stage 1 Policy
    model = PPO.load(
        STAGE1_MODEL_PATH,
        env=env,
        learning_rate=0.00015,   # Reduced LR to prevent KL spikes
        ent_coef=0.0,
        vf_coef=0.25,
        max_grad_norm=0.5,
        target_kl=0.04,          # Expanded target KL buffer
        tensorboard_log=STAGE2_LOG_DIR,
        custom_objects={
            "n_envs": NUM_ENVS,
            "ent_coef": 0.0,
            "vf_coef": 0.25,
            "learning_rate": 0.00015
        }
    )

    # CRITICAL FIX: Reset Critic head so V(s) adapts to scaled returns immediately
    model.policy.value_net.apply(init_weights)

    noise_decay_callback = ActionNoiseDecayCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1),
        save_path=STAGE2_CHECKPOINT_DIR,
        name_prefix="v3_stage2_ckpt"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, noise_decay_callback],
        tb_log_name="PPO_V3_Stage2_NoEarlyStop",
        reset_num_timesteps=True
    )

    model.save(FINAL_SAVE_PATH)
    print(f"\n[SUCCESS] V3 Stage 2 model saved to: {FINAL_SAVE_PATH}")
    env.close()


if __name__ == "__main__":
    train_v3_stage2()