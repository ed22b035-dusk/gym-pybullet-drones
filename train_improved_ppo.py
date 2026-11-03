"""
Improved PPO training script for comprehensive drone hover control.

Based on:
- Default learn.py from gym-pybullet-drones
- PPO best practices from literature
- Comprehensive reward shaping analysis

Training for 300k steps (~15-30 minutes on CPU)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from gym_pybullet_drones.envs.ImprovedHoverAviary import ImprovedHoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType


def main():
    print("=" * 70)
    print("IMPROVED PPO HOVER TRAINING")
    print("=" * 70)
    
    # Create output directory
    filename = 'results/improved_ppo'
    os.makedirs(filename, exist_ok=True)
    
    # === VECTORIZED ENVIRONMENTS ===
    print("Creating environments...")
    train_env = make_vec_env(
        ImprovedHoverAviary,
        env_kwargs=dict(
            drone_model=DroneModel.CF2X,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            ctrl_freq=30,
            initial_xyzs=np.array([[0, 0, 1.0]]),  # Start AT target
            initial_rpys=np.array([[0, 0, 0]]),
            episode_len_sec=30,
        ),
        n_envs=4,  # 4 parallel environments
        seed=0,
    )
    
    # Evaluation environment (same as training)
    eval_env = ImprovedHoverAviary(
        drone_model=DroneModel.CF2X,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        gui=False,
        initial_xyzs=np.array([[0, 0, 1.0]]),
        episode_len_sec=30,
    )
    
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)
    
    # === PPO MODEL ===
    print("Creating PPO model...")
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1,
        tensorboard_log=filename+'/tb/',
    )
    
    # === CALLBACKS ===
    print("Setting up callbacks...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=filename+'/',
        log_path=filename+'/',
        eval_freq=5000,  # Evaluate every 5k steps
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50k steps
        save_path=filename+'/checkpoints',
        name_prefix='ppo_hover',
    )
    
    # === TRAINING ===
    print("\nStarting training for 300k steps...")
    print("This should take ~15-30 minutes depending on your CPU.\n")
    
    model.learn(
        total_timesteps=300000,
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,
    )
    
    # === SAVE MODEL ===
    print("Saving model...")
    model.save(filename+'/final_model')
    train_env.close()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("Model saved to: results/improved_ppo/final_model.zip")
    print("=" * 70)
    
    # === PLOT TRAINING PROGRESS ===
    try:
        data = np.load(filename+'/evaluations.npz')
        timesteps = data['timesteps']
        results = data['results'][:, 0]
        
        print("\nTraining progression:")
        for j in range(timesteps.shape[0]):
            print(f"{timesteps[j]},{results[j]}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, results, 'b-o', markersize=4, linewidth=2)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Mean Episode Reward', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename+'/training_progress.png', dpi=150)
        print(f"\n✓ Plot saved to: {filename}/training_progress.png")
        plt.show()
        
    except Exception as e:
        print(f"\n[WARNING] Could not plot training progress: {e}")


if __name__ == '__main__':
    main()