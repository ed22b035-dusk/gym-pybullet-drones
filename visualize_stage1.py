import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviary import CurriculumHoverAviary


def main():
    model_path = "curriculum_checkpoints/stage1_ppo_model.zip"
    print(f"Loading Stage 1 Model from: {model_path}")
    model = PPO.load(model_path)

    # Initialize Stage 1 Environment in GUI mode
    env = CurriculumHoverAviary(gui=True, stage=1)
    obs, info = env.reset()

    # Telemetry data containers for plotting
    times = []
    positions = []
    attitudes = []
    pos_errors = []

    ctrl_freq = 30  # Control frequency (Hz)
    max_steps = 240  # 8 seconds simulation at 30 Hz

    print("\n--- Running Stage 1 GUI Visualization & Telemetry Logging ---")
    for step in range(max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract current state metrics
        pos = obs[0:3]          # [x, y, z]
        rpy = obs[3:6]          # [roll, pitch, yaw] in radians
        target_pos = np.array([0.0, 0.0, 1.0])
        err_cm = np.linalg.norm(target_pos - pos) * 100.0  # Positional error in cm

        # Store telemetry
        t_sec = step / ctrl_freq
        times.append(t_sec)
        positions.append(pos)
        attitudes.append(np.degrees(rpy))  # Convert to degrees for plotting
        pos_errors.append(err_cm)

        env.render()
        time.sleep(1.0 / ctrl_freq)  # Real-time execution pacing

        if terminated or truncated:
            print(f"Episode completed at step {step} ({t_sec:.2f} seconds).")
            break

    env.close()

    # Convert logged data to numpy arrays for plotting
    positions = np.array(positions)
    attitudes = np.array(attitudes)

    # ==========================================
    # DIAGNOSTIC PLOTS
    # ==========================================
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Stage 1 Hover Performance Diagnostics", fontsize=14, fontweight="bold")

    # Plot 1: Drone 3D Position vs Target
    axs[0].plot(times, positions[:, 0], label="X (m)", color="r")
    axs[0].plot(times, positions[:, 1], label="Y (m)", color="g")
    axs[0].plot(times, positions[:, 2], label="Z (m)", color="b", linewidth=2)
    axs[0].axhline(y=1.0, color="k", linestyle="--", alpha=0.7, label="Target Z (1.0m)")
    axs[0].axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    axs[0].set_ylabel("Position [m]")
    axs[0].set_title("3D Position Tracking (X, Y, Z)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Positional Steady-State Error (cm)
    axs[1].plot(times, pos_errors, color="purple", linewidth=2, label="Euclidean Error (cm)")
    axs[1].axhline(y=2.5, color="red", linestyle="--", alpha=0.7, label="Paper Target Margin (2.5 cm)")
    axs[1].set_ylabel("Error [cm]")
    axs[1].set_title("Positional Error from Target (0, 0, 1)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    # Plot 3: Attitude Angles (Roll, Pitch, Yaw)
    axs[2].plot(times, attitudes[:, 0], label="Roll φ (°)", color="orange")
    axs[2].plot(times, attitudes[:, 1], label="Pitch θ (°)", color="cyan")
    axs[2].plot(times, attitudes[:, 2], label="Yaw ψ (°)", color="magenta")
    axs[2].axhline(y=0.0, color="k", linestyle="--", alpha=0.5)
    axs[2].set_xlabel("Time [seconds]")
    axs[2].set_ylabel("Angle [°]")
    axs[2].set_title("Attitude Stabilization (Roll, Pitch, Yaw)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()  # Display diagnostic plots window immediately


if __name__ == "__main__":
    main()