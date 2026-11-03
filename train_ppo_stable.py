import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gym_pybullet_drones.envs.StableHoverAviary import StableHoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType

def make_env():
    """Create environment factory function."""
    def _init():
        return StableHoverAviary(
            drone_model=DroneModel.CF2X,  # Use enum, not string
            obs=ObservationType.KIN,      # Use enum, not string
            act=ActionType.RPM,           # Use enum, not string
            ctrl_freq=30,
            gui=False,
        )
    return _init

if __name__ == '__main__':
    # Create training environment (4 parallel envs for better gradients)
    train_env = DummyVecEnv([make_env() for _ in range(4)])
    
    # Create evaluation environment (single env, no parallelization)
    eval_env = StableHoverAviary(
        drone_model=DroneModel.CF2X,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        gui=False,
    )
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=2048,  # Steps per update per env
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,  # No entropy bonus (deterministic hover)
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        verbose=1,
        tensorboard_log='results/stable_hover/tb_logs',
    )
    
    # Evaluation callback (every 25k steps)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='results/stable_hover/best_model',
        log_path='results/stable_hover/eval_logs',
        eval_freq=25000,  # Evaluate every 25k steps
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )
    
    # Checkpoint callback (save every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='results/stable_hover/checkpoints',
        name_prefix='ppo_stable_hover',
    )
    
    # Train for 300k steps
    print("=== Training PPO for Stable Hover (300k steps) ===")
    model.learn(
        total_timesteps=300000,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10,
    )
    
    # Save final model
    model.save('results/stable_hover/final_model')
    
    print("=== Training Complete ===")
    print("Check results/stable_hover/ for models and logs")