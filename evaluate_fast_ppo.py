"""
Comprehensive evaluation script for PPO hover policy with detailed metrics.
"""
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.FastPPOHoverAviary import FastPPOHoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType
import os

MODEL_PATH = "results/fast_ppo/final_model.zip"
NUM_EPISODES = 20
MAX_EPISODE_STEPS = 500
TARGET_POS = np.array([0, 0, 1.0])
HOVER_THRESHOLD = 0.2

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    model = PPO.load(MODEL_PATH)
    print(f"✓ Loaded model from {MODEL_PATH}\n")
    
    env = FastPPOHoverAviary(
        drone_model=DroneModel.CF2X,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        gui=False,
        initial_xyzs=np.array([[0, 0, 0.5]]),
        initial_rpys=np.array([[0, 0, 0]]),
    )
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    avg_position_errors = []
    
    print(f"Evaluating {NUM_EPISODES} episodes...\n")
    print("=" * 70)
    
    for ep in range(NUM_EPISODES):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        episode_reward = 0
        positions = []
        avg_error = 999.0  # Default if no positions
        
        for step in range(MAX_EPISODE_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            
            step_result = env.step(action)
            obs = step_result[0]
            reward = step_result[1]
            terminated = step_result[2]
            truncated = step_result[3] if len(step_result) > 3 else False
            info = step_result[4] if len(step_result) > 4 else {}
            
            episode_reward += reward
            
            if 'pos' in info:
                positions.append(info['pos'].copy())
            
            if terminated or truncated:
                break
        
        episode_length = step + 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if len(positions) > 0:
            positions = np.array(positions)
            errors = np.linalg.norm(positions - TARGET_POS, axis=1)
            avg_error = np.mean(errors)
            avg_position_errors.append(avg_error)
            
            if avg_error < HOVER_THRESHOLD:
                success_count += 1
        
        status = "✓ SUCCESS" if avg_error < HOVER_THRESHOLD else "✗ FAIL"
        print(f"Ep {ep+1:2d}: Reward={episode_reward:7.2f}, Len={episode_length:3d}, AvgErr={avg_error:.3f}m {status}")
    
    print("=" * 70)
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Episodes evaluated:     {NUM_EPISODES}")
    
    print(f"\n🎯 Success Rate:")
    success_rate = (success_count / NUM_EPISODES) * 100
    print(f"  Successful episodes:    {success_count}/{NUM_EPISODES} ({success_rate:.1f}%)")
    
    print(f"\n📈 Reward Statistics:")
    print(f"  Mean reward:            {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min reward:             {np.min(episode_rewards):.2f}")
    print(f"  Max reward:             {np.max(episode_rewards):.2f}")
    
    print(f"\n📏 Episode Length:")
    print(f"  Mean length:            {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    
    if len(avg_position_errors) > 0:
        print(f"\n🎯 Position Tracking:")
        print(f"  Mean avg error:         {np.mean(avg_position_errors):.3f} ± {np.std(avg_position_errors):.3f} m")
        print(f"  Best avg error:         {np.min(avg_position_errors):.3f} m")
        print(f"  Worst avg error:        {np.max(avg_position_errors):.3f} m")
    
    print("=" * 70)
    print(f"\n🏆 FINAL VERDICT:")
    if success_rate >= 80:
        print(f"  EXCELLENT! Policy learned to hover! 🎉")
    elif success_rate >= 50:
        print(f"  GOOD!")
    elif success_rate >= 20:
        print(f"  FAIR. Needs more training.")
    else:
        print(f"  POOR. Policy hasn't learned yet.")
    
    print("=" * 70 + "\n")
    env.close()

if __name__ == "__main__":
    evaluate()