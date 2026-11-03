import os
import sys
import glob
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4


def find_stage3_base_model():
    candidate_paths = [
        os.path.join("curriculum_checkpoints_v4_stage3", "v4_stage3_paper_aligned_model.zip"),
        os.path.join("master_curriculum_ckpts", "v4_stage3_model.zip"),
        os.path.join("curriculum_checkpoints_v4", "v4_stage3_model.zip"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
            
    matches = glob.glob("**/*stage3*.zip", recursive=True)
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]
    return None


class ActionShapeAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)

    def step(self, action):
        return self.env.step(action)


def make_env(rank):
    def _init():
        time.sleep(rank * 0.05)
        base_env = CurriculumHoverAviaryV4(gui=False, stage=3)
        return Monitor(ActionShapeAdapter(base_env))
    return _init


def train_v4_stage3_extended():
    model_path = find_stage3_base_model()
    if not model_path:
        print("[FATAL ERROR] Stage 3 base checkpoint not found!")
        sys.exit(1)

    save_dir = "curriculum_checkpoints_v4_stage3_extended"
    os.makedirs(save_dir, exist_ok=True)
    final_save_path = os.path.join(save_dir, "v4_stage3_extended_2m_model.zip")

    num_envs = 4 if os.name == 'nt' else 16
    ext_timesteps = 2_000_000

    print("=" * 85)
    print("   STAGE 3 EXTENDED FINE-TUNING (2.0M ADDITIONAL TIMESTEPS)")
    print("=" * 85)
    print(f"Loaded Base Checkpoint : {model_path}")
    print(f"Parallel Local Workers : {num_envs} Processes")
    print(f"Fine-Tuning Target     : {ext_timesteps:,} Timesteps")
    print(f"Fine-Tuning LR         : 5e-5 (Preserves weight stability during edge optimization)")
    print("=" * 85)

    try:
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    except Exception:
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])

    model = PPO.load(
        model_path,
        env=env,
        learning_rate=5e-5,
        tensorboard_log="logs_v4/stage3_extended",
        custom_objects={"action_space": env.action_space}
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // num_envs, 1),
        save_path=os.path.join(save_dir, "ckpts"),
        name_prefix="v4_stage3_ext_ckpt"
    )

    model.learn(
        total_timesteps=ext_timesteps,
        callback=[checkpoint_callback],
        tb_log_name="PPO_V4_Stage3_Extended",
        reset_num_timesteps=False
    )

    model.save(final_save_path)
    print(f"\n[SUCCESS] Extended Stage 3 Model Saved to: {final_save_path}")
    env.close()


if __name__ == "__main__":
    train_v4_stage3_extended()