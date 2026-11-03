import os
import sys
import glob
import time
import numpy as np
import pybullet as p
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4


# -----------------------------------------------------------------------------
# AUTOMATIC CHECKPOINT FINDER (STAGE 1, STAGE 2, STAGE 3)
# -----------------------------------------------------------------------------
def locate_checkpoint(stage_num):
    """Locates the latest checkpoint zip file for a given curriculum stage."""
    search_patterns = [
        f"master_curriculum_ckpts/v4_stage{stage_num}_model.zip",
        f"curriculum_checkpoints_v4/v4_stage{stage_num}_model.zip",
        f"curriculum_checkpoints_v4_stage{stage_num}/v4_stage{stage_num}_paper_aligned_model.zip",
        f"curriculum_checkpoints_v4/v4_stage{stage_num}_3m_ppo_model.zip",
        f"curriculum_checkpoints_v2/v2_stage{stage_num}_4m_ppo_model.zip",
        f"**/*stage{stage_num}*.zip",
    ]
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    return None


# Locate Checkpoints
STAGE1_PATH = locate_checkpoint(1)
STAGE2_PATH = locate_checkpoint(2)
STAGE3_PATH = locate_checkpoint(3)


# -----------------------------------------------------------------------------
# EXHAUSTIVE 16-CASE STATE-SPACE MATRIX
# -----------------------------------------------------------------------------
ALL_EXTREME_CASES = [
    {"id": 1, "cat": "Spatial", "name": "Max East Boundary (+2.0m)", "pos": np.array([2.0, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 2, "cat": "Spatial", "name": "Max West Boundary (-2.0m)", "pos": np.array([-2.0, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 3, "cat": "Spatial", "name": "Max North Boundary (+2.0m)", "pos": np.array([0.0, 2.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 4, "cat": "Spatial", "name": "Max South Boundary (-2.0m)", "pos": np.array([0.0, -2.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 5, "cat": "Altitude", "name": "Ceiling Proximity (z=1.85m)", "pos": np.array([0.0, 0.0, 1.85]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 6, "cat": "Altitude", "name": "Floor Proximity (z=0.15m)", "pos": np.array([0.0, 0.0, 0.15]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 7, "cat": "Attitude", "name": "Pitch Tilt (+15.0°)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([0.0, 15.0 * np.pi / 180.0, 0.0]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 8, "cat": "Attitude", "name": "Roll Tilt (+15.0°)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([15.0 * np.pi / 180.0, 0.0, 0.0]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 9, "cat": "Attitude", "name": "Reversed Heading (Yaw=+180°)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([0.0, 0.0, np.pi]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 10, "cat": "Kinetic", "name": "Outward Drift (vx=+1.0m/s)", "pos": np.array([1.5, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.array([1.0, 0.0, 0.0]), "ang_vel": np.zeros(3)},
    {"id": 11, "cat": "Kinetic", "name": "Angular Tumble (ωz=+3.0rad/s)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.array([0.0, 0.0, 3.0])},
    {"id": 12, "cat": "Kinetic", "name": "Downward Sink (vz=-0.8m/s)", "pos": np.array([0.0, 0.0, 0.30]), "rpy": np.zeros(3), "vel": np.array([0.0, 0.0, -0.8]), "ang_vel": np.zeros(3)},
    {"id": 13, "cat": "Compound", "name": "Diagonal Corner + Tilt (12°)", "pos": np.array([1.41, 1.41, 0.50]), "rpy": np.array([12.0 * np.pi / 180.0, 12.0 * np.pi / 180.0, 0.0]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 14, "cat": "Environmental", "name": "Outer Boundary + Wind Gusts", "pos": np.array([1.8, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 15, "cat": "Compound", "name": "Floor Sink + Reversed Heading", "pos": np.array([-1.2, -1.2, 0.20]), "rpy": np.array([-12.0 * np.pi / 180.0, 12.0 * np.pi / 180.0, np.pi]), "vel": np.array([0.0, 0.0, -0.5]), "ang_vel": np.zeros(3)},
    {"id": 16, "cat": "Adversary", "name": "Ultimate Adversary Matrix", "pos": np.array([-1.41, 1.41, 0.25]), "rpy": np.array([15.0 * np.pi / 180.0, -15.0 * np.pi / 180.0, np.pi]), "vel": np.array([-0.8, 0.8, -0.4]), "ang_vel": np.array([1.0, -1.0, 2.0])}
]


def predict_action_safe(model, obs_16d):
    """Dynamically slices 16D observation space to match model expected dimension (12D or 16D)."""
    expected_dim = model.observation_space.shape[0]
    obs_input = obs_16d[:expected_dim]
    action, _ = model.predict(obs_input, deterministic=True)
    return action.flatten()


def run_single_rollout(env, model, case):
    """Executes a 240-timestep rollout for a given case and returns time-series data."""
    env.reset()
    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0], case["pos"].tolist(),
        p.getQuaternionFromEuler(case["rpy"]), physicsClientId=env.CLIENT
    )
    p.resetBaseVelocity(
        env.DRONE_IDS[0], linearVelocity=case["vel"].tolist(),
        angularVelocity=case["ang_vel"].tolist(), physicsClientId=env.CLIENT
    )

    obs = env._computeObs()
    
    dist_ts, alt_ts, vel_ts, chatter_ts = [], [], [], []
    last_act = np.zeros(4, dtype=np.float32)

    for _ in range(240):  # 240 Timesteps (8.0 seconds at 30 Hz)
        act = predict_action_safe(model, obs)
        obs, reward, terminated, truncated, info = env.step(act)

        pos = obs[0:3]
        vel = obs[6:9]
        
        dist_cm = np.linalg.norm(pos - np.array([0.0, 0.0, 1.0])) * 100.0
        alt_m = pos[2]
        speed_ms = np.linalg.norm(vel)
        act_delta = np.sum((act - last_act) ** 2)
        last_act = act.copy()

        dist_ts.append(dist_cm)
        alt_ts.append(alt_m)
        vel_ts.append(speed_ms)
        chatter_ts.append(act_delta)

        if terminated or truncated:
            # Pad truncated trajectories with last value to keep length = 240
            pad_len = 240 - len(dist_ts)
            dist_ts.extend([dist_ts[-1]] * pad_len)
            alt_ts.extend([alt_ts[-1]] * pad_len)
            vel_ts.extend([vel_ts[-1]] * pad_len)
            chatter_ts.extend([0.0] * pad_len)
            break

    return {
        "dist": np.array(dist_ts),
        "alt": np.array(alt_ts),
        "vel": np.array(vel_ts),
        "chatter": np.array(chatter_ts)
    }


def compare_all_stages():
    print("=" * 90)
    print("      STAGE 1 vs. STAGE 2 vs. STAGE 3 — MASTER TIME-SERIES COMPARISON SUITE")
    print("=" * 90)
    print(f"Stage 1 Checkpoint : {STAGE1_PATH}")
    print(f"Stage 2 Checkpoint : {STAGE2_PATH}")
    print(f"Stage 3 Checkpoint : {STAGE3_PATH}")
    print("X-Axis Benchmark   : Timesteps (1 to 240 / 8.0 Seconds Horizon)")
    print("=" * 90)

    if not all([STAGE1_PATH, STAGE2_PATH, STAGE3_PATH]):
        print("[FATAL ERROR] Could not locate checkpoints for all 3 stages. Please check paths.")
        sys.exit(1)

    # Initialize Environment
    env = CurriculumHoverAviaryV4(gui=False, stage=3)

    # Load Models
    m1 = PPO.load(STAGE1_PATH)
    m2 = PPO.load(STAGE2_PATH)
    m3 = PPO.load(STAGE3_PATH)

    # Collect time-series trajectory data across all 16 cases
    data = {1: [], 2: [], 3: []}

    print("\nExecuting comparative rollouts across all 16 cases...")
    for idx, case in enumerate(ALL_EXTREME_CASES):
        print(f"  -> Rolling out Case {case['id']:02d}/16: {case['name']}")
        res1 = run_single_rollout(env, m1, case)
        res2 = run_single_rollout(env, m2, case)
        res3 = run_single_rollout(env, m3, case)

        data[1].append(res1)
        data[2].append(res2)
        data[3].append(res3)

    env.close()

    timesteps = np.arange(1, 241)

    # Styling Configurations
    style = {
        1: {"color": "#e74c3c", "label": "Stage 1 (Origin Baseline)", "ls": "--"},
        2: {"color": "#2980b9", "label": "Stage 2 (Spatial Boundary)", "ls": "-."},
        3: {"color": "#2ecc71", "label": "Stage 3 (Full Kinetic Expansion)", "ls": "-"}
    }

    # -------------------------------------------------------------------------
    # FIGURE 1: 3D TRACKING ERROR CONVERGENCE vs. TIMESTEPS (KEY CASES)
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))
    fig1.suptitle("Figure 1: 3D Tracking Error Convergence (cm) vs. Timesteps", fontsize=15, fontweight='bold')

    selected_cases = [0, 2, 13, 15]  # Cases 1, 3, 14, 16
    for idx, c_idx in enumerate(selected_cases):
        ax = axes1[idx // 2, idx % 2]
        case_info = ALL_EXTREME_CASES[c_idx]

        for s in [1, 2, 3]:
            ax.plot(timesteps, data[s][c_idx]["dist"], color=style[s]["color"],
                    linestyle=style[s]["ls"], linewidth=2.0, label=style[s]["label"])

        ax.axhline(5.0, color='red', linestyle=':', alpha=0.7, label="5cm Target Zone")
        ax.axhline(2.0, color='gold', linestyle=':', alpha=0.7, label="2cm Precision Goal")
        ax.set_title(f"Case {case_info['id']:02d}: {case_info['name']}", fontweight='bold')
        ax.set_xlabel("Timesteps (t)")
        ax.set_ylabel("3D Tracking Error (cm)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    print("\n[DISPLAYING FIGURE 1/4] Close plot window to proceed to Figure 2...")
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURE 2: ALTITUDE (Z) RECOVERY PROFILES vs. TIMESTEPS
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle("Figure 2: Altitude Z (m) Recovery Trajectories vs. Timesteps", fontsize=15, fontweight='bold')

    alt_cases = [4, 5, 11, 14]  # Cases 5 (Ceiling), 6 (Floor), 12 (Sink), 15 (Compound Sink)
    for idx, c_idx in enumerate(alt_cases):
        ax = axes2[idx // 2, idx % 2]
        case_info = ALL_EXTREME_CASES[c_idx]

        for s in [1, 2, 3]:
            ax.plot(timesteps, data[s][c_idx]["alt"], color=style[s]["color"],
                    linestyle=style[s]["ls"], linewidth=2.0, label=style[s]["label"])

        ax.axhline(1.0, color='black', linestyle='--', alpha=0.8, label="Target Altitude Z=1.0m")
        ax.axhline(0.05, color='darkred', linestyle='-', alpha=0.5, label="Floor Collision (z=0.05m)")
        ax.set_title(f"Case {case_info['id']:02d}: {case_info['name']}", fontweight='bold')
        ax.set_xlabel("Timesteps (t)")
        ax.set_ylabel("Altitude Z (m)")
        ax.set_ylim(-0.1, 2.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    print("[DISPLAYING FIGURE 2/4] Close plot window to proceed to Figure 3...")
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURE 3: MOTOR ACTION CHATTER DERIVATIVE vs. TIMESTEPS
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))
    fig3.suptitle("Figure 3: Motor Actuator Jitter (||u_t - u_{t-1}||^2) vs. Timesteps", fontsize=15, fontweight='bold')

    chatter_cases = [0, 10, 13, 15]  # Cases 1, 11, 14, 16
    for idx, c_idx in enumerate(chatter_cases):
        ax = axes3[idx // 2, idx % 2]
        case_info = ALL_EXTREME_CASES[c_idx]

        for s in [1, 2, 3]:
            ax.plot(timesteps, data[s][c_idx]["chatter"], color=style[s]["color"],
                    linestyle=style[s]["ls"], linewidth=1.5, alpha=0.85, label=style[s]["label"])

        ax.set_title(f"Case {case_info['id']:02d}: {case_info['name']}", fontweight='bold')
        ax.set_xlabel("Timesteps (t)")
        ax.set_ylabel("Action Delta ||Δu||²")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    print("[DISPLAYING FIGURE 3/4] Close plot window to proceed to Figure 4...")
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURE 4: GLOBAL AGGREGATE METRICS ACROSS ALL 16 CASES vs. TIMESTEPS
    # -------------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10))
    fig4.suptitle("Figure 4: Global Mean Performance Across All 16 Cases vs. Timesteps", fontsize=15, fontweight='bold')

    # Compute Global Means across 16 cases at each timestep
    mean_dist = {s: np.mean([data[s][c]["dist"] for c in range(16)], axis=0) for s in [1, 2, 3]}
    mean_alt_err = {s: np.mean([np.abs(data[s][c]["alt"] - 1.0) for c in range(16)], axis=0) for s in [1, 2, 3]}
    mean_vel = {s: np.mean([data[s][c]["vel"] for c in range(16)], axis=0) for s in [1, 2, 3]}
    mean_chatter = {s: np.mean([data[s][c]["chatter"] for c in range(16)], axis=0) for s in [1, 2, 3]}

    # Plot 1: Global Mean Tracking Error
    ax = axes4[0, 0]
    for s in [1, 2, 3]:
        ax.plot(timesteps, mean_dist[s], color=style[s]["color"], linestyle=style[s]["ls"], linewidth=2.5, label=style[s]["label"])
    ax.set_title("Global Mean 3D Distance Error (cm)", fontweight='bold')
    ax.set_xlabel("Timesteps (t)")
    ax.set_ylabel("Error (cm)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Global Mean Altitude Error
    ax = axes4[0, 1]
    for s in [1, 2, 3]:
        ax.plot(timesteps, mean_alt_err[s], color=style[s]["color"], linestyle=style[s]["ls"], linewidth=2.5, label=style[s]["label"])
    ax.set_title("Global Mean Altitude Absolute Error |z - 1.0| (m)", fontweight='bold')
    ax.set_xlabel("Timesteps (t)")
    ax.set_ylabel("Absolute Z Error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Global Mean Linear Speed
    ax = axes4[1, 0]
    for s in [1, 2, 3]:
        ax.plot(timesteps, mean_vel[s], color=style[s]["color"], linestyle=style[s]["ls"], linewidth=2.5, label=style[s]["label"])
    ax.set_title("Global Mean Linear Velocity ||v|| (m/s)", fontweight='bold')
    ax.set_xlabel("Timesteps (t)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Global Mean Actuator Jitter
    ax = axes4[1, 1]
    for s in [1, 2, 3]:
        ax.plot(timesteps, mean_chatter[s], color=style[s]["color"], linestyle=style[s]["ls"], linewidth=2.5, label=style[s]["label"])
    ax.set_title("Global Mean Actuator Chatter ||Δu||²", fontweight='bold')
    ax.set_xlabel("Timesteps (t)")
    ax.set_ylabel("Jitter Variance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    print("[DISPLAYING FIGURE 4/4] Final plot rendering complete.")
    plt.show()


if __name__ == "__main__":
    compare_all_stages()