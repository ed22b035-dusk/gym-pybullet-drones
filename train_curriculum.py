import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_pybullet_drones.envs.CurriculumHoverAviary import CurriculumHoverAviary


def run_training():
    os.makedirs("curriculum_checkpoints", exist_ok=True)

    # 6 Million Total Steps (1.5M -> 2.0M -> 2.5M)
    TIMESTEPS_STAGE_1 = 1_500_000
    TIMESTEPS_STAGE_2 = 2_000_000
    TIMESTEPS_STAGE_3 = 2_500_000

    NUM_PARALLEL_ENVS = 4  # Safe worker thread count for Windows spawn context

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh
    )

    # ==========================================
    # STAGE 1: Sub-task Omega_1 (Fixed Hover)
    # ==========================================
    print("\n=======================================================")
    print(">>> STAGE 1 / 3: Omega_1 (Fixed Hover from Ground)")
    print("=======================================================")
    env_stage1 = make_vec_env(
        CurriculumHoverAviary,
        env_kwargs=dict(stage=1),
        n_envs=NUM_PARALLEL_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    model = PPO(
        "MlpPolicy",
        env_stage1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1
    )

    model.learn(total_timesteps=TIMESTEPS_STAGE_1)
    stage1_path = os.path.join("curriculum_checkpoints", "stage1_ppo_model")
    model.save(stage1_path)
    env_stage1.close()
    print(f"[SUCCESS] Stage 1 finished. Model saved to {stage1_path}.zip")

    # ==========================================
    # STAGE 2: Sub-task Omega_2 (Pose Randomization)
    # ==========================================
    print("\n=======================================================")
    print(">>> STAGE 2 / 3: Omega_2 (Pose Randomization Transfer)")
    print("=======================================================")
    env_stage2 = make_vec_env(
        CurriculumHoverAviary,
        env_kwargs=dict(stage=2),
        n_envs=NUM_PARALLEL_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    model = PPO.load(stage1_path, env=env_stage2)
    model.learn(total_timesteps=TIMESTEPS_STAGE_2)
    stage2_path = os.path.join("curriculum_checkpoints", "stage2_ppo_model")
    model.save(stage2_path)
    env_stage2.close()
    print(f"[SUCCESS] Stage 2 finished. Model saved to {stage2_path}.zip")

    # ==========================================
    # STAGE 3: Sub-task Omega_3 (Full State Randomization)
    # ==========================================
    print("\n=======================================================")
    print(">>> STAGE 3 / 3: Omega_3 (Full State Randomization Transfer)")
    print("=======================================================")
    env_stage3 = make_vec_env(
        CurriculumHoverAviary,
        env_kwargs=dict(stage=3),
        n_envs=NUM_PARALLEL_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    model = PPO.load(stage2_path, env=env_stage3)
    model.learn(total_timesteps=TIMESTEPS_STAGE_3)
    stage3_path = os.path.join("curriculum_checkpoints", "stage3_ppo_model")
    model.save(stage3_path)
    env_stage3.close()

    print("\n=======================================================")
    print(f"[COMPLETE] ALL 3 STAGES FINISHED SUCCESSFULLY!")
    print(f"Final Trained Policy: {stage3_path}.zip")
    print("=======================================================")


if __name__ == '__main__':
    run_training()