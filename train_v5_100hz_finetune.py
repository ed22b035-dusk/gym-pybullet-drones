import os
import sys
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym_pybullet_drones"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym_pybullet_drones", "envs"))

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_pybullet_drones.envs.CurriculumHoverAviaryV5_100Hz import CurriculumHoverAviaryV5_100Hz


def find_latest_stage3_extended_model():
    candidate_patterns = [
        os.path.join("curriculum_checkpoints_v4_stage3_extended", "v4_stage3_extended_2m_model.zip"),
        os.path.join("curriculum_checkpoints_v4_stage3_extended", "*.zip"),
        "v4_stage3_extended_2m_model.zip",
        "**/*stage3*extended*.zip",
        "**/*2m_model*.zip"
    ]
    for pattern in candidate_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return os.path.abspath(matches[0])
    return None


def make_env(rank, stage=3, seed=0):
    def _init():
        env = CurriculumHoverAviaryV5_100Hz(gui=False, stage=stage)
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == "__main__":
    NUM_ENVS = 8
    STAGE = 3
    TOTAL_TIMESTEPS = 6_666_666  # ~6.66M timesteps @ 100 Hz (matches 4.0M @ 60 Hz flight duration)
    
    SAVE_DIR = "curriculum_checkpoints_v5_100hz"
    CKPT_DIR = os.path.join(SAVE_DIR, "ckpts")
    os.makedirs(CKPT_DIR, exist_ok=True)

    base_model_path = find_latest_stage3_extended_model()
    if not base_model_path or not os.path.exists(base_model_path):
        print(f"\n[ERROR] Pretrained model checkpoint not found!")
        sys.exit(1)

    print("=" * 80)
    print("   V5 100Hz FINE-TUNING RUN")
    print("=" * 80)
    print(f" Pretrained Checkpoint : {base_model_path}")
    print(f" Control Frequency     : 100 Hz (Δt = 10.00 ms)")
    print(f" Target Timesteps       : {TOTAL_TIMESTEPS:,}")
    print(f" Parallel Envs         : {NUM_ENVS}")
    print("=" * 80)

    vec_env = SubprocVecEnv([make_env(i, stage=STAGE) for i in range(NUM_ENVS)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000, 200_000 // NUM_ENVS),
        save_path=CKPT_DIR,
        name_prefix="v5_100hz_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model = PPO.load(
        base_model_path,
        env=vec_env,
        custom_objects={
            "action_space": vec_env.action_space,
            "learning_rate": 5e-5,
            "n_steps": 4096,
            "batch_size": 256,
            "tensorboard_log": "logs_v5/stage3_100hz",
        },
        device="auto"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )

    final_path = os.path.join(SAVE_DIR, "v5_100hz_6.6m_model.zip")
    model.save(final_path)
    print(f"\n[SUCCESS] V5 100Hz Model saved to: {final_path}")
    vec_env.close()