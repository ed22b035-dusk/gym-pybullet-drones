import os
import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_pybullet_drones.envs.CurriculumHoverAviaryV2 import CurriculumHoverAviaryV2

STAGE1_MODEL_PATH = os.path.join("curriculum_checkpoints_v2", "v2_stage1_4m_ppo_model.zip")
STAGE2_CHECKPOINT_DIR = os.path.join("curriculum_checkpoints_v2", "stage2_ckpts")
STAGE2_LOG_DIR = os.path.join("logs_v2", "stage2_ppo")
FINAL_SAVE_PATH = os.path.join("curriculum_checkpoints_v2", "v2_stage2_3m_ppo_model.zip")

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
        time.sleep(rank * 0.1)
        base_env = CurriculumHoverAviaryV2(gui=False, stage=2)
        adapter_env = ActionShapeAdapter(base_env, target_act_shape=(1, 4))
        return Monitor(adapter_env)
    return _init


def train_v2_stage2():
    NUM_ENVS = 8
    TOTAL_TIMESTEPS = 3_000_000

    if not os.path.exists(STAGE1_MODEL_PATH):
        raise FileNotFoundError(f"V2 Stage 1 model missing at: {STAGE1_MODEL_PATH}")

    print("=" * 65)
    print("   STARTING V2 STAGE 2 TRAINING (3.0M STEPS - PROGRESSIVE WARMUP)")
    print("=" * 65)

    try:
        env = SubprocVecEnv([make_stage2_env(i) for i in range(NUM_ENVS)])
    except Exception:
        env = DummyVecEnv([make_stage2_env(i) for i in range(NUM_ENVS)])

    # Load Stage 1 checkpoint
    model = PPO.load(
        STAGE1_MODEL_PATH,
        env=env,
        learning_rate=0.0003,
        ent_coef=0.005,  # Prevents std from shrinking prematurely
        target_kl=0.03,
        tensorboard_log=STAGE2_LOG_DIR,
        custom_objects={"n_envs": NUM_ENVS, "ent_coef": 0.005}
    )

    # RE-INJECT ACTION EXPLORATION NOISE (std = 0.40)
    with torch.no_grad():
        # log(0.40) ≈ -0.9163
        model.policy.log_std.fill_(-0.9163)

    print("[STAGE 2 FIX APPLIED]")
    print("  -> Progressive Warmup: Ramping tilt 2.0 deg -> 14.0 deg over first 500k steps.")
    print("  -> Exploration Re-injected: Action std set to 0.40.")
    print("  -> Entropy Coefficient: Set to 0.005.")
    print("=" * 65)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1),
        save_path=STAGE2_CHECKPOINT_DIR,
        name_prefix="v2_stage2_ckpt"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name="PPO_V2_Stage2",
        reset_num_timesteps=True
    )

    model.save(FINAL_SAVE_PATH)
    print(f"\n[SUCCESS] V2 Stage 2 model saved to: {FINAL_SAVE_PATH}")
    env.close()


if __name__ == "__main__":
    train_v2_stage2()