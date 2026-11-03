import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# --- PyBullet Drones Imports ---
try:
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ObservationType, ActionType
except ImportError:
    from gym_pybullet_drones.envs.single_agent_aviary.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ObservationType, ActionType

MODEL_PATH = os.path.join("curriculum_checkpoints", "stage1_4m_ppo_model.zip")
REPORT_DIR = "evaluation_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

class QuadcopterTelemetryLogger:
    """Collects per-step telemetry across evaluation episodes."""
    def __init__(self, target_pos=np.array([0.0, 0.0, 1.0])):
        self.target_pos = target_pos
        self.reset()

    def reset(self):
        self.positions = []
        self.orientations = []
        self.actions = []
        self.rewards = []
        self.episode_lengths = []

    def log_step(self, pos, euler_deg, act, reward):
        self.positions.append(pos)
        self.orientations.append(euler_deg)
        self.actions.append(act)
        self.rewards.append(reward)

    def end_episode(self, ep_length):
        self.episode_lengths.append(ep_length)

def format_obs(raw_obs, target_dim):
    """Flattens and trims/pads observation to match exact model expectations."""
    flat = np.array(raw_obs, dtype=np.float32).flatten()
    if len(flat) >= target_dim:
        return flat[:target_dim]
    return np.pad(flat, (0, target_dim - len(flat)))

def evaluate_stage1_policy(num_episodes=10, render=False):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")
        
    print(f"[EVAL] Loading checkpoint: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    target_obs_dim = model.observation_space.shape[0]
    print(f"[EVAL] Model observation dimension: {target_obs_dim}")

    print(f"[EVAL] Initializing HoverAviary environment (gui={render})...")
    env = HoverAviary(gui=render, obs=ObservationType.KIN, act=ActionType.RPM)

    target_pos = np.array([0.0, 0.0, 1.0])
    telemetry = QuadcopterTelemetryLogger(target_pos=target_pos)
    ep_returns = []

    for ep in range(num_episodes):
        obs_tuple = env.reset()
        raw_obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
        done = False
        step_count = 0
        ep_reward = 0.0

        while not done:
            # Prepare observation shape for model
            model_obs = format_obs(raw_obs, target_obs_dim)
            action, _ = model.predict(model_obs, deterministic=True)

            # Format action for HoverAviary (1, 4)
            action_env = action.reshape(1, -1) if action.ndim == 1 else action

            step_res = env.step(action_env)
            if len(step_res) == 5:
                raw_obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            else:
                raw_obs, reward, done, info = step_res

            # Extract kinematic state [x, y, z, roll, pitch, yaw]
            flat_kin = np.array(raw_obs, dtype=np.float32).flatten()
            pos = flat_kin[0:3]
            euler_deg = np.degrees(flat_kin[3:6]) if len(flat_kin) >= 6 else np.zeros(3)

            telemetry.log_step(pos, euler_deg, action.flatten(), float(reward))
            ep_reward += float(reward)
            step_count += 1

        telemetry.end_episode(step_count)
        ep_returns.append(ep_reward)
        print(f"Episode {ep+1:02d}/{num_episodes:02d} | Return: {ep_reward:.2f} | Length: {step_count} steps")

    env.close()

    # Calculate analytical metrics
    positions = np.array(telemetry.positions)
    orientations = np.array(telemetry.orientations)
    actions = np.array(telemetry.actions)

    pos_errors = positions - telemetry.target_pos
    pos_rmse = np.sqrt(np.mean(pos_errors**2, axis=0))
    total_pos_rmse = np.linalg.norm(pos_rmse)
    attitude_std = np.std(orientations, axis=0)
    action_deltas = np.diff(actions, axis=0)
    action_smoothness = np.mean(np.abs(action_deltas))

    metrics = {
        "episodes_evaluated": num_episodes,
        "mean_return": float(np.mean(ep_returns)),
        "std_return": float(np.std(ep_returns)),
        "mean_episode_length": float(np.mean(telemetry.episode_lengths)),
        "pos_rmse_x_m": float(pos_rmse[0]),
        "pos_rmse_y_m": float(pos_rmse[1]),
        "pos_rmse_z_m": float(pos_rmse[2]),
        "total_pos_rmse_m": float(total_pos_rmse),
        "attitude_std_roll_deg": float(attitude_std[0]),
        "attitude_std_pitch_deg": float(attitude_std[1]),
        "attitude_std_yaw_deg": float(attitude_std[2]),
        "action_smoothness_delta": float(action_smoothness)
    }

    json_path = os.path.join(REPORT_DIR, "stage1_4m_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "="*55)
    print("       STAGE 1 (4.0M) IN-DEPTH EVALUATION SUMMARY")
    print("="*55)
    print(f" Mean Return         : {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f" Mean Horizon Length : {metrics['mean_episode_length']:.1f} steps")
    print(f" Positional RMSE     : {metrics['total_pos_rmse_m']*100:.2f} cm (X:{pos_rmse[0]*100:.2f}cm, Y:{pos_rmse[1]*100:.2f}cm, Z:{pos_rmse[2]*100:.2f}cm)")
    print(f" Attitude Variance   : Roll: {attitude_std[0]:.2f}°, Pitch: {attitude_std[1]:.2f}°, Yaw: {attitude_std[2]:.2f}°")
    print(f" Motor Delta Effort  : {action_smoothness:.4f} (Control jitter index)")
    print("="*55 + "\n")

    # Save Telemetry Dashboard Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Stage 1 (4.0M) Telemetry Performance Analysis", fontsize=16, fontweight='bold')

    steps = np.arange(len(positions))
    axs[0, 0].plot(steps, pos_errors[:, 0], label="X error (m)", alpha=0.8)
    axs[0, 0].plot(steps, pos_errors[:, 1], label="Y error (m)", alpha=0.8)
    axs[0, 0].plot(steps, pos_errors[:, 2], label="Z error (m)", alpha=0.8)
    axs[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs[0, 0].set_title("Setpoint Position Error (m)")
    axs[0, 0].set_xlabel("Steps")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(steps, orientations[:, 0], label="Roll (°)", color='r', alpha=0.8)
    axs[0, 1].plot(steps, orientations[:, 1], label="Pitch (°)", color='g', alpha=0.8)
    axs[0, 1].plot(steps, orientations[:, 2], label="Yaw (°)", color='b', alpha=0.8)
    axs[0, 1].set_title("Attitude Angles (°)")
    axs[0, 1].set_xlabel("Steps")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    for i in range(actions.shape[1]):
        axs[1, 0].plot(steps, actions[:, i], label=f"Motor {i+1}", alpha=0.7)
    axs[1, 0].set_title("Rotor Command Outputs")
    axs[1, 0].set_xlabel("Steps")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='purple', alpha=0.6, label='Flight Path')
    ax_3d.scatter(target_pos[0], target_pos[1], target_pos[2], color='red', s=100, label='Target Setpoint')
    ax_3d.set_title("3D Spatial Drift")
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.legend()

    plot_path = os.path.join(REPORT_DIR, "stage1_4m_telemetry_dashboard.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"[EVAL] Telemetry dashboard saved to: {plot_path}")

if __name__ == "__main__":
    evaluate_stage1_policy(num_episodes=10, render=False)