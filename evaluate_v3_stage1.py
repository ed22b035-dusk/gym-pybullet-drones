import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gymnasium as gym
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviaryV3 import CurriculumHoverAviaryV3

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = os.path.join("curriculum_checkpoints_v2", "v2_stage1_4m_ppo_model.zip")
NUM_EVAL_EPISODES = 10
EVAL_STAGE = 1
RENDER_GUI = False  # Set True to visualize live in PyBullet
TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)


class ActionShapeAdapter(gym.Wrapper):
    def __init__(self, env, target_act_shape=(1, 4)):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=target_act_shape, dtype=np.float32
        )

    def step(self, action):
        return self.env.step(action)


# ==============================================================================
# EVALUATION CORE
# ==============================================================================
def run_evaluation():
    if not os.path.exists(MODEL_PATH):
        # Fallback check for V3 naming convention if V2 baseline path doesn't exist
        fallback_path = os.path.join("curriculum_checkpoints_v3", "v3_stage1_ppo_model.zip")
        if os.path.exists(fallback_path):
            model_to_load = fallback_path
        else:
            raise FileNotFoundError(f"Stage 1 checkpoint not found at: {MODEL_PATH}")
    else:
        model_to_load = MODEL_PATH

    # Initialize Environment & Model
    base_env = CurriculumHoverAviaryV3(gui=RENDER_GUI, stage=EVAL_STAGE)
    env = ActionShapeAdapter(base_env, target_act_shape=(1, 4))
    model = PPO.load(model_to_load)

    all_ep_returns = []
    all_ep_lengths = []
    episodes_data = []

    print(f"\nRunning Stage 1 evaluation across {NUM_EVAL_EPISODES} episodes...")
    for ep in range(NUM_EVAL_EPISODES):
        obs, info = env.reset(seed=100 + ep)
        done = False
        
        ep_reward = 0.0
        step_count = 0
        
        pos_history = []
        rpy_history = []
        vel_history = []
        ang_vel_history = []
        action_history = []

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Extract raw state vector from aviary
            state = env.unwrapped._getDroneStateVector(0)
            
            pos_history.append(state[0:3].copy())
            rpy_history.append(state[7:10].copy())
            vel_history.append(state[10:13].copy())
            ang_vel_history.append(state[13:16].copy())
            action_history.append(np.array(action).flatten())

            ep_reward += reward
            step_count += 1

        all_ep_returns.append(ep_reward)
        all_ep_lengths.append(step_count)
        
        episodes_data.append({
            "pos": np.array(pos_history),
            "rpy": np.array(rpy_history) * 180.0 / np.pi,  # Degrees
            "vel": np.array(vel_history),
            "ang_vel": np.array(ang_vel_history) * 180.0 / np.pi,
            "actions": np.array(action_history)
        })

    env.close()

    # ==========================================================================
    # METRIC CALCULATIONS
    # ==========================================================================
    flat_pos = np.concatenate([ep["pos"] for ep in episodes_data], axis=0)
    flat_rpy = np.concatenate([ep["rpy"] for ep in episodes_data], axis=0)
    flat_vel = np.concatenate([ep["vel"] for ep in episodes_data], axis=0)
    flat_ang_vel = np.concatenate([ep["ang_vel"] for ep in episodes_data], axis=0)
    flat_actions = np.concatenate([ep["actions"] for ep in episodes_data], axis=0)

    # Position tracking metrics
    err_3d = np.linalg.norm(flat_pos - TARGET_POS, axis=1)
    err_xy = np.linalg.norm(flat_pos[:, 0:2] - TARGET_POS[0:2], axis=1)
    err_z = np.abs(flat_pos[:, 2] - TARGET_POS[2])
    
    # Precision hover compliance (<2.5 cm position error & <2 deg attitude tilt)
    precision_hover_mask = (err_3d < 0.025) & (np.abs(flat_rpy[:, 0]) < 2.0) & (np.abs(flat_rpy[:, 1]) < 2.0)
    precision_hover_pct = (np.sum(precision_hover_mask) / len(err_3d)) * 100.0

    # Motor chatter / smoothness evaluation
    action_deltas = []
    for ep in episodes_data:
        acts = ep["actions"]
        if len(acts) > 1:
            diffs = np.diff(acts, axis=0)
            action_deltas.append(np.mean(diffs**2))
    mean_action_jitter = np.mean(action_deltas) if action_deltas else 0.0

    survival_secs = np.array(all_ep_lengths) / 30.0

    # ==========================================================================
    # TERMINAL SUMMARY REPORT (COPY-PASTE READY)
    # ==========================================================================
    report = f"""
================================================================================
               STAGE 1 BASELINE POLICY EVALUATION REPORT                        
================================================================================
[MODEL & STAGE CONFIGURATION]
Model Loaded         : {model_to_load}
Evaluation Stage     : Stage 1 (Origin Takeoff to Hover Target [0, 0, 1])
Total Eval Episodes  : {NUM_EVAL_EPISODES}
Control Frequency    : 30 Hz (0.033s step period)
Max Episode Horizon  : 240 steps (8.0 seconds)

[EPISODE SURVIVAL & REWARD METRICS]
Mean Episode Return  : {np.mean(all_ep_returns):10.2f} +/- {np.std(all_ep_returns):.2f} (Min: {np.min(all_ep_returns):.2f}, Max: {np.max(all_ep_returns):.2f})
Mean Flight Time     : {np.mean(survival_secs):10.2f}s +/- {np.std(survival_secs):.2f}s (Min: {np.min(survival_secs):.2f}s, Max: {np.max(survival_secs):.2f}s)
Mean Steps Survived  : {np.mean(all_ep_lengths):10.1f}   +/- {np.std(all_ep_lengths):.1f} steps (Max: 240)
Full Horizon Rate    : {(np.sum(np.array(all_ep_lengths) >= 240) / NUM_EVAL_EPISODES) * 100.0:10.1f}%

[SPATIAL POSITIONING ACCURACY]
3D Tracking Error    : {np.mean(err_3d)*100:10.2f} cm  +/- {np.std(err_3d)*100:.2f} cm (Max: {np.max(err_3d)*100:.2f} cm)
XY Radial Error      : {np.mean(err_xy)*100:10.2f} cm  +/- {np.std(err_xy)*100:.2f} cm (Max: {np.max(err_xy)*100:.2f} cm)
Z Altitude Error     : {np.mean(err_z)*100:10.2f} cm  +/- {np.std(err_z)*100:.2f} cm (Max: {np.max(err_z)*100:.2f} cm)
Precision Hover Rate : {precision_hover_pct:10.2f}% (<2.5cm error & <2 deg tilt)

[ATTITUDE & STABILITY]
Mean Absolute Roll   : {np.mean(np.abs(flat_rpy[:, 0])):10.2f} deg (Max: {np.max(np.abs(flat_rpy[:, 0])):.2f} deg)
Mean Absolute Pitch  : {np.mean(np.abs(flat_rpy[:, 1])):10.2f} deg (Max: {np.max(np.abs(flat_rpy[:, 1])):.2f} deg)
Mean Absolute Yaw    : {np.mean(np.abs(flat_rpy[:, 2])):10.2f} deg (Max: {np.max(np.abs(flat_rpy[:, 2])):.2f} deg)

[KINEMATICS & CONTROL EFFORT]
Linear Speed         : {np.mean(np.linalg.norm(flat_vel, axis=1)):10.3f} m/s (Max: {np.max(np.linalg.norm(flat_vel, axis=1)):.3f} m/s)
Angular Velocity     : {np.mean(np.linalg.norm(flat_ang_vel, axis=1)):10.2f} deg/s (Max: {np.max(np.linalg.norm(flat_ang_vel, axis=1)):.2f} deg/s)
Mean Motor Command   : {np.mean(flat_actions):10.3f} [-1.0 to +1.0 scale]
Control Jitter (MSD) : {mean_action_jitter:10.5f} (Lower = smoother rotor response)
================================================================================
"""
    print(report)

    # ==========================================================================
    # VISUALIZATION DASHBOARD (DISPLAY ONLY)
    # ==========================================================================
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Stage 1 Baseline Policy Performance Dashboard", fontsize=16, fontweight='bold', y=0.98)
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25)

    best_ep_idx = np.argmax(all_ep_lengths)
    rep_ep = episodes_data[best_ep_idx]
    time_axis = np.arange(len(rep_ep["pos"])) / 30.0

    c_blue = "#1f77b4"
    c_orange = "#ff7f0e"
    c_green = "#2ca02c"
    c_red = "#d62728"
    c_purple = "#9467bd"

    # --- 1. 3D Takeoff & Ascent Trajectories ---
    ax1 = fig.add_subplot(gs[0:2, 0], projection='3d')
    for i, ep in enumerate(episodes_data):
        alpha = 0.85 if i == best_ep_idx else 0.35
        lw = 2.0 if i == best_ep_idx else 1.0
        ax1.plot(ep["pos"][:, 0], ep["pos"][:, 1], ep["pos"][:, 2], alpha=alpha, linewidth=lw)
    
    ax1.scatter(0, 0, 0, color='black', s=60, marker='o', label='Origin (0,0,0)')
    ax1.scatter(TARGET_POS[0], TARGET_POS[1], TARGET_POS[2], color='red', s=120, marker='*', label='Target (0,0,1)')
    ax1.set_title("3D Ascent Trajectories", fontweight='bold')
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend(loc='upper right')

    # --- 2. Altitude & Position Tracking ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_axis, rep_ep["pos"][:, 0], label='X', color=c_blue, linewidth=1.5)
    ax2.plot(time_axis, rep_ep["pos"][:, 1], label='Y', color=c_orange, linewidth=1.5)
    ax2.plot(time_axis, rep_ep["pos"][:, 2], label='Z Altitude', color=c_green, linewidth=2.0)
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Target Z=1.0m')
    ax2.axhspan(0.975, 1.025, color='green', alpha=0.15, label='Precision Zone (+/-2.5cm)')
    ax2.set_title("Ascent & Hover Stabilization", fontweight='bold')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Position (m)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=8)

    # --- 3. Attitude Angles ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time_axis, rep_ep["rpy"][:, 0], label='Roll', color=c_blue, linewidth=1.2)
    ax3.plot(time_axis, rep_ep["rpy"][:, 1], label='Pitch', color=c_orange, linewidth=1.2)
    ax3.plot(time_axis, rep_ep["rpy"][:, 2], label='Yaw', color=c_purple, linewidth=1.2)
    ax3.set_title("Attitude Angles (Roll, Pitch, Yaw)", fontweight='bold')
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Angle (degrees)")
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8)

    # --- 4. Rotor RPM Commands ---
    ax4 = fig.add_subplot(gs[1, 1])
    for m in range(4):
        ax4.plot(time_axis, rep_ep["actions"][:, m], label=f'Rotor {m+1}', alpha=0.85, linewidth=1.2)
    ax4.set_title("Rotor Action Commands", fontweight='bold')
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Normalized Action [-1, +1]")
    ax4.set_ylim(-1.1, 1.1)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower right', ncol=2, fontsize=8)

    # --- 5. Error CDF ---
    ax5 = fig.add_subplot(gs[1, 2])
    sorted_err = np.sort(err_3d * 100.0)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax5.plot(sorted_err, cdf, color=c_blue, linewidth=2.0)
    ax5.axvline(2.5, color=c_green, linestyle='--', label='2.5 cm Threshold')
    ax5.set_title("3D Position Error CDF (All Steps)", fontweight='bold')
    ax5.set_xlabel("3D Distance Error (cm)")
    ax5.set_ylabel("Cumulative Probability")
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.legend(loc='lower right', fontsize=8)

    # --- 6. Step Durations Bar Chart ---
    ax6 = fig.add_subplot(gs[2, 0:2])
    ep_ids = np.arange(1, NUM_EVAL_EPISODES + 1)
    bars = ax6.bar(ep_ids, all_ep_lengths, color=c_blue, alpha=0.75, width=0.5, label='Steps Survived')
    ax6.axhline(240, color=c_red, linestyle='--', label='Target Episode Horizon (240 Steps)')
    ax6.set_title("Stage 1 Survival Durations", fontweight='bold')
    ax6.set_xlabel("Evaluation Episode Index")
    ax6.set_ylabel("Steps Survived")
    ax6.set_xticks(ep_ids)
    ax6.set_ylim(0, 260)
    ax6.grid(True, linestyle=':', alpha=0.6)
    ax6.legend(loc='lower right', fontsize=8)

    for bar in bars:
        yval = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f"{int(yval)}", ha='center', va='bottom', fontsize=8)

    # --- 7. Linear Speed Profile ---
    ax7 = fig.add_subplot(gs[2, 2])
    vel_norm = np.linalg.norm(rep_ep["vel"], axis=1)
    ax7.plot(time_axis, vel_norm, color=c_red, linewidth=1.5, label='Speed (m/s)')
    ax7.fill_between(time_axis, 0, vel_norm, color=c_red, alpha=0.15)
    ax7.set_title("Linear Speed Profile", fontweight='bold')
    ax7.set_xlabel("Time (seconds)")
    ax7.set_ylabel("Speed (m/s)")
    ax7.grid(True, linestyle=':', alpha=0.6)
    ax7.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_evaluation()