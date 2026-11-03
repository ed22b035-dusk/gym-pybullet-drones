import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_pybullet_drones.envs.CurriculumHoverAviaryV2 import CurriculumHoverAviaryV2

CHECKPOINT_DIR = os.path.join("curriculum_checkpoints_v2", "stage1_ckpts")
LOG_DIR = os.path.join("logs_v2", "stage1_ppo")
FINAL_SAVE_PATH = os.path.join("curriculum_checkpoints_v2", "v2_stage1_4m_ppo_model.zip")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class ActionShapeAdapter(gym.Wrapper):
    def __init__(self, env, target_act_shape=(1, 4)):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=target_act_shape, dtype=np.float32
        )

    def step(self, action):
        return self.env.step(action)


def make_stage1_env(rank):
    def _init():
        time.sleep(rank * 0.1)
        base_env = CurriculumHoverAviaryV2(gui=False, stage=1)
        adapter_env = ActionShapeAdapter(base_env, target_act_shape=(1, 4))
        return Monitor(adapter_env)
    return _init


def train_v2_stage1():
    NUM_ENVS = 8
    TOTAL_TIMESTEPS = 4_000_000

    print("=" * 65)
    print("   STARTING V2 STAGE 1 TRAINING (4.0M STEPS - GROUND LAUNCH)")
    print("   Reward: V2 Smooth Sigmoid Barrier + Linear Velocity Damping")
    print("=" * 65)

    try:
        env = SubprocVecEnv([make_stage1_env(i) for i in range(NUM_ENVS)])
    except Exception:
        env = DummyVecEnv([make_stage1_env(i) for i in range(NUM_ENVS)])

    model = PPO(
        "MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=LOG_DIR
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="v2_stage1_ckpt"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name="PPO_V2_Stage1"
    )

    model.save(FINAL_SAVE_PATH)
    print(f"\n[SUCCESS] V2 Stage 1 model saved to: {FINAL_SAVE_PATH}")
    env.close()


if __name__ == "__main__":
    train_v2_stage1()