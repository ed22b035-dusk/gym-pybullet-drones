"""
Comprehensive evaluation script for improved PPO hover policy.

Evaluates policy over 20 episodes and generates detailed plots:
- Episode rewards
- Episode lengths
- Position errors (XY and Z)
- Attitude (roll, pitch)
- Success rate
"""

import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.ImprovedHoverAviary import ImprovedHoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType
import os
import matplotlib.pyplot as plt

MODEL_PATH = "results/improved_ppo/final_model.zip"
NUM_EPISODES = 20
MAX_EPISODE_STEPS = 900  # 30 seconds * 30 Hz
TARGET_POS = np.array([0, 0, 1.0])
HOVER_THRESHOLD = 0.2  # Within 20cm = success


def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please run train_improved_ppo.py first!")
        return
    
    model = PPO.load(MODEL_PATH)
    print(f"✓ Loaded model from {MODEL_PATH}\n")
    
    env = ImprovedHoverAviary(
        drone_model=DroneModel.CF2X,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=30,
        gui=False,
        initial_xyzs=np.array([[0, 0, 1.0]]),
        episode_len_sec=30,
    )
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    avg_position_errors = []
    avg_xy_errors = []
    avg_z_errors = []
    max_tilts = []
    
    print(f"Evaluating {NUM_EPISODES} episodes...\n")
    print("=" * 70)
    
    for ep in range(NUM_EPISODES):
        obs, info = env.reset()
        episode_reward = 0
        positions = []
        xy_errors = []
        z_errors = []
        tilts = []
        
        for step in range(MAX_EPISODE_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if 'position_error' in info:
                positions.append(info['position_error'])
            if 'xy_error' in info:
                xy_errors.append(info['xy_error'])
            if 'z_error' in info:
                z_errors.append(info['z_error'])
            if 'tilt' in info:
                tilts.append(info['tilt'])
            
            if terminated or truncated:
                break
        
        episode_length = step + 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if len(positions) > 0:
            avg_error = np.mean(positions)
            avg_xy = np.mean(xy_errors)
            avg_z = np.mean(z_errors)
            max_tilt = np.max(tilts)
            
            avg_position_errors.append(avg_error)
            avg_xy_errors.append(avg_xy)
            avg_z_errors.append(avg_z)
            max_tilts.append(max_tilt)
            
            if avg_error < HOVER_THRESHOLD:
                success_count += 1
        
        status = "✓ SUCCESS" if avg_error < HOVER_THRESHOLD else "✗ FAIL"
        print(f"Ep {ep+1:2d}: Reward={episode_reward:7.2f}, Len={episode_length:3d}, "
              f"AvgErr={avg_error:.3f}m (XY={avg_xy:.3f}, Z={avg_z:.3f}), "
              f"MaxTilt={np.rad2deg(max_tilt):5.1f}° {status}")
    
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
    print(f"  Max possible:           {MAX_EPISODE_STEPS} steps (30 sec)")
    
    if len(avg_position_errors) > 0:
        print(f"\n🎯 Position Tracking:")
        print(f"  Mean avg error:         {np.mean(avg_position_errors):.3f} ± {np.std(avg_position_errors):.3f} m")
        print(f"  Mean XY error:          {np.mean(avg_xy_errors):.3f} ± {np.std(avg_xy_errors):.3f} m")
        print(f"  Mean Z error:           {np.mean(avg_z_errors):.3f} ± {np.std(avg_z_errors):.3f} m")
        print(f"  Best avg error:         {np.min(avg_position_errors):.3f} m")
        
        print(f"\n📐 Attitude:")
        print(f"  Mean max tilt:          {np.rad2deg(np.mean(max_tilts)):.1f}° ± {np.rad2deg(np.std(max_tilts)):.1f}°")
    
    print("=" * 70)
    print(f"\n🏆 FINAL VERDICT:")
    if success_rate >= 80:
        print(f"  EXCELLENT! Policy learned to hover! 🎉")
    elif success_rate >= 50:
        print(f"  GOOD! Policy is learning.")
    elif success_rate >= 20:
        print(f"  FAIR. Needs more training.")
    else:
        print(f"  POOR. Policy hasn't learned yet.")
    
    print("=" * 70 + "\n")
    
    # === GENERATE PLOTS ===
    print("Generating plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Graph 1: Episode Rewards
    axes[0, 0].plot(range(1, NUM_EPISODES + 1), episode_rewards, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=np.mean(episode_rewards), color='r', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Total Reward', fontsize=12)
    axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Graph 2: Episode Lengths
    axes[0, 1].plot(range(1, NUM_EPISODES + 1), episode_lengths, 'g-o', linewidth=2, markersize=6)
    axes[0, 1].axhline(y=np.mean(episode_lengths), color='r', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(episode_lengths):.1f}')
    axes[0, 1].axhline(y=MAX_EPISODE_STEPS, color='orange', linestyle=':', linewidth=2, 
                       label=f'Max: {MAX_EPISODE_STEPS}')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Episode Length (steps)', fontsize=12)
    axes[0, 1].set_title('Episode Lengths', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Graph 3: Position Errors (XY and Z)
    if len(avg_position_errors) > 0:
        axes[1, 0].plot(range(1, len(avg_position_errors) + 1), avg_xy_errors, 'b-o', 
                       linewidth=2, markersize=6, label='XY error')
        axes[1, 0].plot(range(1, len(avg_z_errors) + 1), avg_z_errors, 'r-o', 
                       linewidth=2, markersize=6, label='Z error')
        axes[1, 0].axhline(y=HOVER_THRESHOLD, color='k', linestyle='--', linewidth=2, 
                          label=f'Threshold: {HOVER_THRESHOLD}m')
        axes[1, 0].set_xlabel('Episode', fontsize=12)
        axes[1, 0].set_ylabel('Average Error (m)', fontsize=12)
        axes[1, 0].set_title('Position Errors', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Graph 4: Max Tilt per Episode
    if len(max_tilts) > 0:
        max_tilts_deg = np.rad2deg(max_tilts)
        axes[1, 1].plot(range(1, NUM_EPISODES + 1), max_tilts_deg, 'm-o', 
                       linewidth=2, markersize=6)
        axes[1, 1].axhline(y=60, color='r', linestyle='--', linewidth=2, 
                          label='Truncation limit (60°)')
        axes[1, 1].set_xlabel('Episode', fontsize=12)
        axes[1, 1].set_ylabel('Max Tilt (degrees)', fontsize=12)
        axes[1, 1].set_title('Maximum Tilt per Episode', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Graph 5: Success/Failure per Episode
    if len(avg_position_errors) > 0:
        success_binary = [1 if avg_position_errors[i] < HOVER_THRESHOLD else 0 
                         for i in range(len(avg_position_errors))]
        colors = ['green' if s == 1 else 'red' for s in success_binary]
        axes[2, 0].bar(range(1, NUM_EPISODES + 1), success_binary, color=colors, alpha=0.7)
        axes[2, 0].set_xlabel('Episode', fontsize=12)
        axes[2, 0].set_ylabel('Success (1) / Failure (0)', fontsize=12)
        axes[2, 0].set_title(f'Success Rate: {success_rate:.1f}%', fontsize=14, fontweight='bold')
        axes[2, 0].set_ylim([0, 1.2])
        axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    # Graph 6: Reward vs Position Error
    if len(avg_position_errors) > 0:
        axes[2, 1].scatter(avg_position_errors, episode_rewards, c='blue', s=100, alpha=0.6)
        axes[2, 1].set_xlabel('Average Position Error (m)', fontsize=12)
        axes[2, 1].set_ylabel('Total Reward', fontsize=12)
        axes[2, 1].set_title('Reward vs Position Error', fontsize=14, fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/improved_ppo/evaluation_results.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as 'results/improved_ppo/evaluation_results.png'")
    plt.show()
    
    env.close()


if __name__ == "__main__":
    evaluate()