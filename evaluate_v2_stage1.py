import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviaryV2 import CurriculumHoverAviaryV2

# ==============================================================================
# EVALUATION CONFIGURATION
# ==============================================================================
MODEL_PATH = os.path.join("curriculum_checkpoints_v2", "v2_stage1_4m_ppo_model.zip")
STAGE = 1
FLIGHT_DURATION_SEC = 8.0  # 8.0s = 240 timesteps
SAVE_PLOT_PATH = "v2_stage1_evaluation.png"
# ==============================================================================


def evaluate_and_plot():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Stage 1 V2 model not found at: {MODEL_PATH}")
        return

    print("=" * 65)
    print("   RUNNING V2 STAGE 1 QUANTITATIVE EVALUATION & PLOTTING")
    print(f"   Model: {MODEL_PATH}")
    print("=" * 65)

    model = PPO.load(MODEL_PATH)
    env = CurriculumHoverAviaryV2(gui=False, stage=STAGE)
    env.EPISODE_LEN_SEC = FLIGHT_DURATION_SEC

    max_steps = int(FLIGHT_DURATION_SEC * 30)

    # Telemetry storage
    time_points = []
    positions = []       # [x, y, z]
    orientations = []    # [roll, pitch, yaw] in degrees
    velocities = []      # [vx, vy, vz]
    ang_velocities = []  # [wx, wy, wz] in deg/s
    tracking_errors = [] # Te
    rewards = []

    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        flat_obs = np.array(obs, dtype=np.float32).flatten()[:12]
        action, _ = model.predict(flat_obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Extract telemetry
        pos = obs[0:3]
        rpy_deg = np.degrees(obs[3:6])
        vel = obs[6:9]
        ang_v_deg = np.degrees(obs[9:12])

        target_pos = np.array([0.0, 0.0, 1.0])
        dist_err = np.linalg.norm(pos - target_pos)
        yaw_err = abs(obs[5])
        T_e = dist_err + yaw_err

        t_curr = step / 30.0
        time_points.append(t_curr)
        positions.append(pos)
        orientations.append(rpy_deg)
        velocities.append(vel)
        ang_velocities.append(ang_v_deg)
        tracking_errors.append(T_e)
        rewards.append(reward)

        step += 1

    env.close()

    # Convert lists to NumPy arrays
    time_pts = np.array(time_points)
    pos_arr = np.array(positions)
    rpy_arr = np.array(orientations)
    vel_arr = np.array(velocities)
    ang_v_arr = np.array(ang_velocities)
    te_arr = np.array(tracking_errors)
    rew_arr = np.array(rewards)

    # Calculate Steady-State Hover Metrics (Last 3 seconds / 90 steps)
    ss_slice = slice(-90, None)
    ss_pos_err = np.linalg.norm(pos_arr[ss_slice] - np.array([0.0, 0.0, 1.0]), axis=1)
    rmse_ss = np.sqrt(np.mean(ss_pos_err**2))
    max_drift_ss = np.max(ss_pos_err)
    mean_reward = np.mean(rew_arr)

    print("\n---------------- TELEMETRY SUMMARY METRICS ----------------")
    print(f"  -> Total Steps Evaluated       : {step}/{max_steps}")
    print(f"  -> Steady-State RMSE Error     : {rmse_ss * 1000:.2f} mm")
    print(f"  -> Steady-State Max Drift      : {max_drift_ss * 1000:.2f} mm")
    print(f"  -> Mean Step Reward            : {mean_reward:.2f} / 85.0")
    print(f"  -> Final Position [x,y,z]      : {pos_arr[-1].round(4)} m")
    print(f"  -> Final Attitude RPY          : {rpy_arr[-1].round(2)} deg")
    print(f"  -> Final Velocity              : {vel_arr[-1].round(4)} m/s")
    print("-----------------------------------------------------------")

    # ==============================================================================
    # GENERATE PLOTS
    # ==============================================================================
    plt.style.use('seaborn-v0_8-grid' if 'seaborn-v0_8-grid' in plt.style.available else 'default')
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("V2 Stage 1 Flight Telemetry & Performance Analysis", fontsize=14, fontweight='bold')

    # Subplot 1: Position Tracking (X, Y, Z)
    axs[0, 0].plot(time_pts, pos_arr[:, 0], label='X (m)', color='tab:blue')
    axs[0, 0].plot(time_pts, pos_arr[:, 1], label='Y (m)', color='tab:orange')
    axs[0, 0].plot(time_pts, pos_arr[:, 2], label='Z (m)', color='tab:green')
    axs[0, 0].axhline(1.0, color='black', linestyle='--', linewidth=1, label='Target Z=1.0m')
    axs[0, 0].set_ylabel("Position (m)")
    axs[0, 0].set_title("3D Trajectory Tracking")
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid(True)

    # Subplot 2: Tracking Error Te
    axs[0, 1].plot(time_pts, te_arr, color='tab:red', linewidth=1.5)
    axs[0, 1].set_ylabel("Target Error $T_e$ (m)")
    axs[0, 1].set_title("Tracking Error $T_e = ||p - p_{target}||_2 + |\\psi_{err}|$")
    axs[0, 1].grid(True)

    # Subplot 3: Attitude Angles (Roll, Pitch, Yaw)
    axs[1, 0].plot(time_pts, rpy_arr[:, 0], label='Roll (deg)', color='tab:purple')
    axs[1, 0].plot(time_pts, rpy_arr[:, 1], label='Pitch (deg)', color='tab:brown')
    axs[1, 0].plot(time_pts, rpy_arr[:, 2], label='Yaw (deg)', color='tab:pink')
    axs[1, 0].set_ylabel("Attitude (deg)")
    axs[1, 0].set_title("Attitude Angles (RPY)")
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].grid(True)

    # Subplot 4: Linear Velocities
    axs[1, 1].plot(time_pts, vel_arr[:, 0], label='$v_x$', color='tab:blue')
    axs[1, 1].plot(time_pts, vel_arr[:, 1], label='$v_y$', color='tab:orange')
    axs[1, 1].plot(time_pts, vel_arr[:, 2], label='$v_z$', color='tab:green')
    axs[1, 1].set_ylabel("Velocity (m/s)")
    axs[1, 1].set_title("Linear Velocities")
    axs[1, 1].legend(loc='lower right')
    axs[1, 1].grid(True)

    # Subplot 5: Angular Velocities
    axs[2, 0].plot(time_pts, ang_v_arr[:, 0], label='$\\omega_x$', color='tab:purple')
    axs[2, 0].plot(time_pts, ang_v_arr[:, 1], label='$\\omega_y$', color='tab:brown')
    axs[2, 0].plot(time_pts, ang_v_arr[:, 2], label='$\\omega_z$', color='tab:pink')
    axs[2, 0].set_xlabel("Time (seconds)")
    axs[2, 0].set_ylabel("Angular Velocity (deg/s)")
    axs[2, 0].set_title("Body Angular Rates")
    axs[2, 0].legend(loc='lower right')
    axs[2, 0].grid(True)

    # Subplot 6: Step Rewards over Time
    axs[2, 1].plot(time_pts, rew_arr, color='tab:cyan', linewidth=1.5)
    axs[2, 1].axhline(85.0, color='tab:gray', linestyle=':', label='Theoretical Max (85.0)')
    axs[2, 1].set_xlabel("Time (seconds)")
    axs[2, 1].set_ylabel("Step Reward $R_t$")
    axs[2, 1].set_title("Step Reward Curve")
    axs[2, 1].legend(loc='lower right')
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(SAVE_PLOT_PATH, dpi=300)
    print(f"\n[SUCCESS] Plot saved to: {SAVE_PLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    evaluate_and_plot()