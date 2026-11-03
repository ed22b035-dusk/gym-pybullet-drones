import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_pybullet_drones.envs.CurriculumHoverAviary import CurriculumHoverAviary


def run_stage1_continuation():
    input_model_path = os.path.join("curriculum_checkpoints", "stage1_ppo_model.zip")
    output_model_path = os.path.join("curriculum_checkpoints", "stage1_4m_ppo_model")

    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"Checkpoint not found at '{input_model_path}'.")

    # Train for 2,500,000 additional steps (1.5M existing + 2.5M additional = 4.0M Total)
    ADDITIONAL_TIMESTEPS = 2_500_000
    
    # Using 2 parallel environments so CPU cores are shared cleanly with Stage 2
    NUM_PARALLEL_ENVS = 2

    print("=" * 70)
    print(">>> PARALLEL STAGE 1 CONTINUATION (1.5M -> 4.0M STEPS)")
    print(f">>> Reading base checkpoint: {input_model_path}")
    print(f">>> Final model will save to: {output_model_path}.zip")
    print("=" * 70)

    # Re-create Stage 1 environment
    env_stage1 = make_vec_env(
        CurriculumHoverAviary,
        env_kwargs=dict(stage=1),
        n_envs=NUM_PARALLEL_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    # Load Stage 1 weights from disk
    model = PPO.load(input_model_path, env=env_stage1)

    # Continue training for 2.5M additional steps
    # reset_num_timesteps=False maintains the internal step count starting from 1.5M
    model.learn(total_timesteps=ADDITIONAL_TIMESTEPS, reset_num_timesteps=False)

    # Save output to a distinct 4M checkpoint file
    model.save(output_model_path)
    env_stage1.close()

    print("\n" + "=" * 70)
    print("[SUCCESS] Stage 1 extended training to 4.0M steps complete!")
    print(f"Saved policy location: {output_model_path}.zip")
    print("=" * 70)


if __name__ == '__main__':
    run_stage1_continuation()