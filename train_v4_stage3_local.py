import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4

STAGE2_MODEL_PATH = os.path.join("curriculum_checkpoints_v4", "v4_stage2_3m_ppo_model.zip")
STAGE3_SAVE_DIR = "curriculum_checkpoints_v4_stage3"


def make_env(rank, seed=0):
    def _init():
        # Initialize Stage 3: Full Kinetic Perturbations (v0, omega0)
        env = CurriculumHoverAviaryV4(gui=False, stage=3)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_stage3_local():
    os.makedirs(STAGE3_SAVE_DIR, exist_ok=True)
    
    # 4 workers is the sweet spot for local Windows PCs to prevent RAM OOM crashes
    num_cpu = 4  

    print("=" * 80)
    print("   LOCAL PC — V4 STAGE 3 PAPER-ALIGNED CURRICULUM TRAINING")
    print("=" * 80)
    print(f"Loaded Stage 2 Weights     : {STAGE2_MODEL_PATH}")
    print(f"Parallel Local Workers     : {num_cpu} Processes")
    print(f"Target Horizon             : 3.0M Timesteps")
    print("Kinetic State Expansion    : v0 in [-1.0, 1.0] m/s | omega0 in [-3.0, 3.0] rad/s")
    print("=" * 80)

    if not os.path.exists(STAGE2_MODEL_PATH):
        print(f"\n[FATAL ERROR] Stage 2 model missing at: {STAGE2_MODEL_PATH}")
        sys.exit(1)

    # Initialize 4 vectorized environment processes
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # Load Stage 2 policy weights cleanly into Stage 3
    model = PPO.load(
        STAGE2_MODEL_PATH,
        env=env,
        learning_rate=1e-4,  # Fine-tuning learning rate for extreme kinetics
        tensorboard_log=os.path.join(STAGE3_SAVE_DIR, "tb_logs"),
        custom_objects={"action_space": env.action_space}
    )

    # Train for 3.0M timesteps on local PC
    print("\nStarting local Stage 3 training loop...")
    model.learn(
        total_timesteps=3000000,
        progress_bar=True
    )

    final_path = os.path.join(STAGE3_SAVE_DIR, "v4_stage3_paper_aligned_model.zip")
    model.save(final_path)
    print(f"\n[SUCCESS] Local Stage 3 model saved to: {final_path}")


if __name__ == "__main__":
    train_stage3_local()