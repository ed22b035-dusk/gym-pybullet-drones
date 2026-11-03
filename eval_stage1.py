import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviary import CurriculumHoverAviary


def evaluate_stage1():
    model_path = "curriculum_checkpoints/stage1_ppo_model.zip"
    
    print("=" * 75)
    print(">>> LOADING STAGE 1 MODEL FOR IN-DEPTH DIAGNOSTIC & GUI VISUALIZATION")
    print(f">>> Path: {model_path}")
    print("=" * 75)

    # Load trained Stage 1 policy
    model = PPO.load(model_path)

    # Instantiate Stage 1 env with PyBullet GUI active
    env = CurriculumHoverAviary(gui=True, stage=1)
    obs, info = env.reset()

    # Telemetry storage for diagnostics
    timesteps = []
    positions = []         # [x, y, z]
    rpy_angles = []        # [roll, pitch, yaw] in degrees
    pos_errors_cm = []     # Euclidean distance to target (0,0,1) in cm
    rewards = []
    actions_list = []

    target_pos = np.array([0.0, 0.0, 1.0])
    total_steps = 240      # 8 seconds flight duration at 30Hz

    print("\n--- STEP-BY-STEP LIVE TELEMETRY LOG ---")
    print(f"{'Step':<6} | {'Pos (x, y, z) [m]':<22} | {'Err (cm)':<10} | {'Angles (R, P, Y) [deg]':<25} | {'Reward':<8}")
    print("-" * 80)

    for step in range(total_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Unpack observation vector
        pos = obs[0:3]
        rpy_rad = obs[3:6]
        rpy_deg = np.degrees(rpy_rad)
        
        pos_err_cm = np.linalg.norm(target_pos - pos) * 100.0  # m to cm

        # Store telemetry metrics
        timesteps.append(step / 30.0)  # Convert step to time in seconds
        positions.append(pos)
        rpy_angles.append(rpy_deg)
        pos_errors_cm.append(pos_err_cm)
        rewards.append(reward)
        actions_list.append(action)

        # Print telemetry every 15 steps (every 0.5s)
        if step % 15 == 0 or step == total_steps - 1:
            pos_str = f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
            rpy_str = f"[{rpy_deg[0]:.2f}°, {rpy_deg[1]:.2f}°, {rpy_deg[2]:.2f}°]"
            print(f"{step:<6} | {pos_str:<22} | {pos_err_cm:<10.2f} | {rpy_str:<25} | {reward:<8.2f}")

        env.render()
        time.sleep(1.0 / 30.0)  # Pacing for GUI

        if terminated or truncated:
            print(f"\n[INFO] Episode finished early at step {step}.")
            break

    env.close()

    # Convert logged lists to numpy arrays
    positions = np.array(positions)
    rpy_angles = np.array(rpy_angles)
    pos_errors_cm = np.array(pos_errors_cm)
    rewards = np.array(rewards)
    actions_list = np.array(actions_list)

    # Output detailed summary statistics for analysis
    print("\n" + "=" * 75)
    print(">>> IN-DEPTH STAGE 1 PERFORMANCE SUMMARY DIAGNOSTICS")
    print("=" * 75)
    print(f"Total Flight Duration Logged : {timesteps[-1]:.2f} seconds ({len(timesteps)} steps)")
    print(f"Final Hover Position         : X={positions[-1,0]:.4f}m, Y={positions[-1,1]:.4f}m, Z={positions[-1,2]:.4f}m")
    print(f"Mean Positional Error        : {np.mean(pos_errors_cm):.2f} cm")
    print(f"Min Positional Error         : {np.min(pos_errors_cm):.2f} cm")
    print(f"Steady-State Altitude Error  : {abs(1.0 - positions[-1,2])*100.0:.2f} cm")
    print(f"Max Tilt (Roll/Pitch)        : Roll={np.max(np.abs(rpy_angles[:,0])):.2f}°, Pitch={np.max(np.abs(rpy_angles[:,1])):.2f}°")
    print(f"Mean Step Reward             : {np.mean(rewards):.2f}")
    print(f"Total Episode Reward         : {np.sum(rewards):.2f}")
    
    # Paper benchmark evaluation: Delta_p = 2.5 cm radius threshold
    within_delta_p = np.sum(pos_errors_cm <= 2.5) / len(pos_errors_cm) * 100.0
    print(f"Hover Precision Metric       : {within_delta_p:.1f}% of flight time within paper target (<= 2.5 cm)")
    print("=" * 75 + "\n")

    # Generate multi-panel diagnostic plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Stage 1 (Omega_1) Diagnostic Flight Performance Analysis", fontsize=14, fontweight='bold')

    # 1. Altitude (Z Position) Trajectory
    axs[0, 0].plot(timesteps, positions[:, 2], label='Drone Height (Z)', color='blue', linewidth=2)
    axs[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Target Height (1.0m)')
    axs[0, 0].set_title("Altitude (Z-Axis) Trajectory")
    axs[0, 0].set_xlabel("Time (seconds)")
    axs[0, 0].set_ylabel("Height (meters)")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2. 3D Positional Error Over Time
    axs[0, 1].plot(timesteps, pos_errors_cm, color='purple', linewidth=2, label='Positional Error (cm)')
    axs[0, 1].axhline(y=2.5, color='green', linestyle='--', label='Paper Target Threshold (2.5 cm)')
    axs[0, 1].set_title("3D Distance Error to Target (0, 0, 1)")
    axs[0, 1].set_xlabel("Time (seconds)")
    axs[0, 1].set_ylabel("Error (cm)")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # 3. Attitude Angles (Roll, Pitch, Yaw)
    axs[1, 0].plot(timesteps, rpy_angles[:, 0], label='Roll (phi)', color='red')
    axs[1, 0].plot(timesteps, rpy_angles[:, 1], label='Pitch (theta)', color='green')
    axs[1, 0].plot(timesteps, rpy_angles[:, 2], label='Yaw (psi)', color='orange')
    axs[1, 0].axhline(y=15, color='black', linestyle=':', label='Truncation Boundary (+-15°)')
    axs[1, 0].axhline(y=-15, color='black', linestyle=':')
    axs[1, 0].set_title("Attitude Stability (Euler Angles)")
    axs[1, 0].set_xlabel("Time (seconds)")
    axs[1, 0].set_ylabel("Angle (degrees)")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # 4. Action Commands (Motor Outputs)
    axs[1, 1].plot(timesteps, actions_list[:, 0], label='Motor 1', alpha=0.8)
    axs[1, 1].plot(timesteps, actions_list[:, 1], label='Motor 2', alpha=0.8)
    axs[1, 1].plot(timesteps, actions_list[:, 2], label='Motor 3', alpha=0.8)
    axs[1, 1].plot(timesteps, actions_list[:, 3], label='Motor 4', alpha=0.8)
    axs[1, 1].set_title("Control Action Output (Normalized Motor Commands)")
    axs[1, 1].set_xlabel("Time (seconds)")
    axs[1, 1].set_ylabel("Normalized Action Value")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    evaluate_stage1()