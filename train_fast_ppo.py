import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gym_pybullet_drones.envs.FastPPOHoverAviary import FastPPOHoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType

def make_env():
    """Create environment."""
    def _init():
        return FastPPOHoverAviary(
            drone_model=DroneModel.CF2X,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            ctrl_freq=30,
            gui=False,
            initial_xyzs=np.array([[0, 0, 0.5]]),
            initial_rpys=np.array([[0, 0, 0]]),
        )
    return _init

if __name__ == '__main__':
    print("Starting training...")
    
    train_env = DummyVecEnv([make_env() for _ in range(4)])
    
    eval_env = FastPPOHoverAviary(
        drone_model=DroneModel.CF2X,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        gui=False,
        initial_xyzs=np.array([[0, 0, 0.5]]),
        initial_rpys=np.array([[0, 0, 0]]),
    )
    
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
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        verbose=1,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='results/fast_ppo/best_model',
        log_path='results/fast_ppo/eval_logs',
        eval_freq=25000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='results/fast_ppo/checkpoints',
        name_prefix='ppo_fast_hover',
    )
    
    print("Training for 500k steps...")
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, checkpoint_callback],
    )
    
    model.save('results/fast_ppo/final_model')
    
    print("Training complete! Model saved.")