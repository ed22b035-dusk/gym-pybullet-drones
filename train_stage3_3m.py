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

STAGE2_MODEL_PATH = os.path.join("curriculum_checkpoints", "stage2_3m_ppo_model.zip")
STAGE3_CHECKPOINT_DIR = os.path.join("curriculum_checkpoints", "stage3_checkpoints")
STAGE3_LOG_DIR = os.path.join("logs", "stage3_ppo")
FINAL_SAVE_PATH = os.path.join("curriculum_checkpoints", "stage3_3m_ppo_model.zip")

os.makedirs(STAGE3_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(STAGE3_LOG_DIR, exist_ok=True)


class ActionShapeAdapter(gym.Wrapper):
    """Adapts CurriculumHoverAviary action space shape to match the loaded model."""
    def __init__(self, env, target_act_shape=(1, 4)):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=target_act_shape, dtype=np.float32
        )

    def step(self, action):
        return self.env.step(action)


def make_stage3_env(rank, target_act_shape=(1, 4)):
    def _init():
        time.sleep(rank * 0.1)
        base_env = CurriculumHoverAviary(gui=False, stage=3)
        adapter_env = ActionShapeAdapter(base_env, target_act_shape=target_act_shape)
        return Monitor(adapter_env)
    return _init


def train_stage3():
    NUM_ENVS = 8
    TOTAL_TIMESTEPS = 1_000_000

    if not os.path.exists(STAGE2_MODEL_PATH):
        raise FileNotFoundError(f"Stage 2 checkpoint not found at: {STAGE2_MODEL_PATH}")

    temp_model = PPO.load(STAGE2_MODEL_PATH)
    target_act_shape = temp_model.action_space.shape
    del temp_model

    print("=" * 60)
    print("   STAGE 3 TRAINING (ROBUST RECOVERY)")
    print("   Tilts: +-14 deg | Velocity Shocks: +-0.5 m/s | Wind: std=0.03N")
    print("=" * 60)

    try:
        env = SubprocVecEnv([make_stage3_env(i, target_act_shape) for i in range(NUM_ENVS)])
    except Exception:
        env = DummyVecEnv([make_stage3_env(i, target_act_shape) for i in range(NUM_ENVS)])

    model = PPO.load(
        STAGE2_MODEL_PATH,
        env=env,
        tensorboard_log=STAGE3_LOG_DIR,
        learning_rate=0.0001,
        target_kl=0.03,
        custom_objects={"n_envs": NUM_ENVS}
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1),
        save_path=STAGE3_CHECKPOINT_DIR,
        name_prefix="stage3_ppo_ckpt"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name="PPO_Stage3_Paper",
        reset_num_timesteps=True
    )

    model.save(FINAL_SAVE_PATH)
    print(f"\n[SUCCESS] Stage 3 complete! Saved final model to: {FINAL_SAVE_PATH}")
    env.close()


if __name__ == "__main__":
    train_stage3()