import os
import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4

STAGE2_CHECKPOINT_DIR = os.path.join("curriculum_checkpoints_v4", "stage2_ckpts")
STAGE2_LOG_DIR = os.path.join("logs_v4", "stage2_ppo")
FINAL_SAVE_PATH = os.path.join("curriculum_checkpoints_v4", "v4_stage2_3m_ppo_model.zip")

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


def make_stage2_env(rank):
    def _init():
        time.sleep(rank * 0.05)
        base_env = CurriculumHoverAviaryV4(gui=False, stage=2)
        adapter_env = ActionShapeAdapter(base_env, target_act_shape=(1, 4))
        return Monitor(adapter_env)
    return _init


def train_v4_stage2():
    NUM_ENVS = 16            # High compute parallel environment workers
    TOTAL_TIMESTEPS = 5_000_000

    print("=" * 70)
    print("   STARTING V4 STAGE 2 HIGH-COMPUTE TRAINING (5.0M STEPS / 16 WORKERS)")
    print("=" * 70)

    try:
        env = SubprocVecEnv([make_stage2_env(i) for i in range(NUM_ENVS)])
    except Exception:
        env = DummyVecEnv([make_stage2_env(i) for i in range(NUM_ENVS)])

    # Native 16D PPO model initialization
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,      # Unconstrained optimization updates
        tensorboard_log=STAGE2_LOG_DIR,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(200_000 // NUM_ENVS, 1),
        save_path=STAGE2_CHECKPOINT_DIR,
        name_prefix="v4_stage2_ckpt"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback],
        tb_log_name="PPO_V4_Stage2_HighCompute",
        reset_num_timesteps=True
    )

    model.save(FINAL_SAVE_PATH)
    print(f"\n[SUCCESS] V4 Stage 2 model saved to: {FINAL_SAVE_PATH}")
    env.close()


if __name__ == "__main__":
    train_v4_stage2()