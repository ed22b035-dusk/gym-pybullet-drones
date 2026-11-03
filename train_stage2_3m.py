import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

try:
    from CurriculumHoverAviary import CurriculumHoverAviary
except ImportError:
    from gym_pybullet_drones.envs.CurriculumHoverAviary import CurriculumHoverAviary

STAGE1_MODEL_PATH = os.path.join("curriculum_checkpoints", "stage1_4m_ppo_model.zip")
STAGE2_CHECKPOINT_DIR = os.path.join("curriculum_checkpoints", "stage2_checkpoints")
STAGE2_LOG_DIR = os.path.join("logs", "stage2_ppo")
FINAL_SAVE_PATH = os.path.join("curriculum_checkpoints", "stage2_3m_ppo_model.zip")

os.makedirs(STAGE2_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(STAGE2_LOG_DIR, exist_ok=True)


class ActionShapeAdapter(gym.Wrapper):
    """Adapts CurriculumHoverAviary action space shape to match the loaded model."""
    def __init__(self, env, target_act_shape=(1, 4)):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=target_act_shape, dtype=np.float32
        )

    def step(self, action):
        return self.env.step(action)


def make_stage2_env(rank, target_act_shape=(1, 4)):
    def _init():
        time.sleep(rank * 0.1)
        base_env = CurriculumHoverAviary(gui=False, stage=2)
        adapter_env = ActionShapeAdapter(base_env, target_act_shape=target_act_shape)
        return Monitor(adapter_env)
    return _init


def train_stage2():
    NUM_ENVS = 8
    TOTAL_TIMESTEPS = 1_000_000  # Optimal Stage 2 fine-tuning horizon

    if not os.path.exists(STAGE1_MODEL_PATH):
        raise FileNotFoundError(f"Stage 1 checkpoint not found at: {STAGE1_MODEL_PATH}")

    # Inspect Stage 1 model's expected action space shape dynamically
    temp_model = PPO.load(STAGE1_MODEL_PATH)
    target_act_shape = temp_model.action_space.shape
    del temp_model

    print("=" * 60)
    print("   STAGE 2 TRAINING (CURRICULUM PAPER ENVIRONMENT)")
    print(f"-> Loaded Base Model    : {STAGE1_MODEL_PATH}")
    print(f"-> Target Action Shape  : {target_act_shape}")
    print("   Wind Gusts: std=0.02N (~7% drone weight), 20% burst rate")
    print("=" * 60)

    try:
        env = SubprocVecEnv([make_stage2_env(i, target_act_shape) for i in range(NUM_ENVS)])
    except Exception:
        print("[WARN] SubprocVecEnv failed. Falling back to DummyVecEnv...")
        env = DummyVecEnv([make_stage2_env(i, target_act_shape) for i in range(NUM_ENVS)])

    model = PPO.load(
        STAGE1_MODEL_PATH,
        env=env,
        tensorboard_log=STAGE2_LOG_DIR,
        learning_rate=0.0001,
        target_kl=0.03,
        custom_objects={"n_envs": NUM_ENVS}
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1),
        save_path=STAGE2_CHECKPOINT_DIR,
        name_prefix="stage2_ppo_ckpt"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name="PPO_Stage2_Paper",
        reset_num_timesteps=True
    )

    model.save(FINAL_SAVE_PATH)
    print(f"\n[SUCCESS] Stage 2 complete! Saved final model to: {FINAL_SAVE_PATH}")
    env.close()


if __name__ == "__main__":
    train_stage2()