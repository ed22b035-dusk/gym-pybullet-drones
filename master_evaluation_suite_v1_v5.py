"""
==================================================================================================
  MASTER IN-DEPTH CROSS-GENERATIONAL EVALUATION & VISUALIZATION SUITE (V1 to V5)
  Quadcopter Autonomous Control via Reinforcement Learning
==================================================================================================
This master script evaluates, visualizes, and compares all model generations (V1, V2, V3, V4, V5-60Hz, V5-100Hz)
across the 16-case extreme benchmark matrix in PyBullet 3D GUI with real-time telemetry and comparative diagnostic plots.
"""

import os
import sys
import glob
import time
import numpy as np
import pybullet as p
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Path Resolution for gym_pybullet_drones
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym-pybullet-drones"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym_pybullet_drones"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym-pybullet-drones", "envs"))

# Import Environment Versions
try:
    from gym_pybullet_drones.envs.CurriculumHoverAviary import CurriculumHoverAviary as EnvV1
except ImportError:
    EnvV1 = None

try:
    from gym_pybullet_drones.envs.CurriculumHoverAviaryV2 import CurriculumHoverAviaryV2 as EnvV2
except ImportError:
    EnvV2 = None

try:
    from gym_pybullet_drones.envs.CurriculumHoverAviaryV3 import CurriculumHoverAviaryV3 as EnvV3
except ImportError:
    EnvV3 = None

try:
    from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4 as EnvV4
except ImportError:
    EnvV4 = None

try:
    from gym_pybullet_drones.envs.CurriculumHoverAviaryV5_60Hz import CurriculumHoverAviaryV5_60Hz as EnvV5_60Hz
except ImportError:
    EnvV5_60Hz = None

try:
    from gym_pybullet_drones.envs.CurriculumHoverAviaryV5_100Hz import CurriculumHoverAviaryV5_100Hz as EnvV5_100Hz
except ImportError:
    EnvV5_100Hz = None


# -----------------------------------------------------------------------------
# CONFIGURATION & PACING
# -----------------------------------------------------------------------------
PLAYBACK_SPEED = 1.0        # 1.0 = Real-Time, 0.5 = Half-Speed Slow-Mo
PAUSE_BETWEEN_CASES = False  # Set True to press Enter before each case in terminal
ENABLE_3D_GUI = True        # PyBullet GUI toggle


# -----------------------------------------------------------------------------
# ACTION SPACE & OBSERVATION ADAPTERS
# -----------------------------------------------------------------------------
class ActionShapeAdapter(gym.Wrapper):
    """Adapts single environment Gym action space Box(4,) to Box(1, 4) or vice versa."""
    def __init__(self, env, target_shape=(1, 4)):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=target_shape, dtype=np.float32)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action_flat = np.array(action, dtype=np.float32).flatten()
        return self.env.step(action_flat)


# -----------------------------------------------------------------------------
# EXHAUSTIVE 16-CASE BENCHMARK MATRIX (ALL GENERATIONS)
# -----------------------------------------------------------------------------
ALL_EXTREME_CASES = [
    # Spatial Extremes (Boundary Radius r = 2.0m)
    {"id": 1, "cat": "Spatial Bound", "name": "Max East Boundary (x=+2.0m, y=0.0m, z=1.0m)", "pos": np.array([2.0, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 2, "cat": "Spatial Bound", "name": "Max West Boundary (x=-2.0m, y=0.0m, z=1.0m)", "pos": np.array([-2.0, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 3, "cat": "Spatial Bound", "name": "Max North Boundary (x=0.0m, y=+2.0m, z=1.0m)", "pos": np.array([0.0, 2.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 4, "cat": "Spatial Bound", "name": "Max South Boundary (x=0.0m, y=-2.0m, z=1.0m)", "pos": np.array([0.0, -2.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},

    # Altitude Danger Zones
    {"id": 5, "cat": "Altitude Bound", "name": "Near Ceiling Max Altitude (z=1.85m, Target=1.0m)", "pos": np.array([0.0, 0.0, 1.85]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 6, "cat": "Altitude Bound", "name": "Floor Proximity Danger Zone (z=0.15m, Target=1.0m)", "pos": np.array([0.0, 0.0, 0.15]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},

    # Attitude Extremes
    {"id": 7, "cat": "Attitude Bound", "name": "Max Forward Pitch Tilt (+15.0 Pitch)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([0.0, 15.0 * np.pi / 180.0, 0.0]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 8, "cat": "Attitude Bound", "name": "Max Right Roll Tilt (+15.0 Roll)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([15.0 * np.pi / 180.0, 0.0, 0.0]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 9, "cat": "Attitude Bound", "name": "Completely Reversed Heading (Yaw = +180.0)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([0.0, 0.0, np.pi]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},

    # Kinetic Momentum & Low-Alt Sink Rates
    {"id": 10, "cat": "Kinetic Extreme", "name": "High Outward Linear Velocity (v0_x = +1.00 m/s)", "pos": np.array([1.5, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.array([1.00, 0.0, 0.0]), "ang_vel": np.zeros(3)},
    {"id": 11, "cat": "Kinetic Extreme", "name": "Extreme Angular Tumble Rate (w0_z = +3.0 rad/s)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.array([0.0, 0.0, 3.0])},
    {"id": 12, "cat": "Kinetic Extreme", "name": "Fast Downward Sink Velocity (v0_z = -0.80 m/s, z=0.30m)", "pos": np.array([0.0, 0.0, 0.30]), "rpy": np.zeros(3), "vel": np.array([0.0, 0.0, -0.80]), "ang_vel": np.zeros(3)},

    # Compound Stress, Wind & Ultimate Adversary
    {"id": 13, "cat": "Compound Stress", "name": "Diagonal Corner Spawn + Compound Tilt (r=2.0m, Roll/Pitch=+12)", "pos": np.array([1.41, 1.41, 0.50]), "rpy": np.array([12.0 * np.pi / 180.0, 12.0 * np.pi / 180.0, 0.0]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 14, "cat": "Environmental", "name": "Max Outer Boundary Under Active Wind Gusts", "pos": np.array([1.80, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 15, "cat": "Compound Kinetic", "name": "Floor Proximity + Downward Velocity + Tilt (z=0.20m, v0_z=-0.5m/s)", "pos": np.array([-1.20, -1.20, 0.20]), "rpy": np.array([-12.0 * np.pi / 180.0, 12.0 * np.pi / 180.0, np.pi]), "vel": np.array([0.0, 0.0, -0.50]), "ang_vel": np.zeros(3)},
    {"id": 16, "cat": "Ultimate Adversary", "name": "The Ultimate Adversary (r=2.0m, Tilt=15, Yaw=180, v0=[-0.8,0.8,-0.4], w0=[1,-1,2])", "pos": np.array([-1.41, 1.41, 0.25]), "rpy": np.array([15.0 * np.pi / 180.0, -15.0 * np.pi / 180.0, np.pi]), "vel": np.array([-0.80, 0.80, -0.40]), "ang_vel": np.array([1.0, -1.0, 2.0])}
]


# -----------------------------------------------------------------------------
# AUTOMATIC CHECKPOINT DISCOVERY REGISTRY
# -----------------------------------------------------------------------------
def discover_model_checkpoints():
    """Scans local directory for checkpoints of all generations V1 to V5."""
    registry = {}

    patterns = {
        "V1 Baseline (30Hz)": [
            "curriculum_checkpoints/stage3_3m_ppo_model.zip",
            "curriculum_checkpoints/stage3_ppo_model.zip",
            "**/*stage3*3m*.zip"
        ],
        "V2 Sigmoid (30Hz)": [
            "curriculum_checkpoints_v2/v2_stage2_3m_ppo_model.zip",
            "curriculum_checkpoints_v2/v2_stage1_4m_ppo_model.zip",
            "**/*v2*stage2*.zip"
        ],
        "V3 Scaled (30Hz)": [
            "curriculum_checkpoints_v3/v3_stage2_3m_ppo_model.zip",
            "curriculum_checkpoints_v3/v3_stage2_ckpt*.zip",
            "**/*v3*stage2*.zip"
        ],
        "V4 Extended 5.0M (30Hz)": [
            "curriculum_checkpoints_v4_stage3_extended/v4_stage3_extended_2m_model.zip",
            "curriculum_checkpoints_v4_stage3_extended/ckpts/*.zip",
            "**/*v4*extended*.zip"
        ],
        "V5 Fine-Tuned (60Hz)": [
            "curriculum_checkpoints_v5_60hz/v5_60hz_4m_model.zip",
            "curriculum_checkpoints_v5_60hz/ckpts/*.zip",
            "**/*v5_60hz*.zip"
        ],
        "V5 Fine-Tuned (100Hz)": [
            "curriculum_checkpoints_v5_100hz/v5_100hz_6.6m_model.zip",
            "curriculum_checkpoints_v5_100hz/ckpts/*.zip",
            "**/*v5_100hz*.zip"
        ]
    }

    for name, pat_list in patterns.items():
        found_path = None
        for pat in pat_list:
            matches = glob.glob(os.path.join(SCRIPT_DIR, pat), recursive=True)
            if matches:
                matches.sort(key=os.path.getmtime, reverse=True)
                found_path = os.path.abspath(matches[0])
                break
        if found_path:
            registry[name] = found_path

    return registry


# -----------------------------------------------------------------------------
# ENVIRONMENT FACTORY BY GENERATION
# -----------------------------------------------------------------------------
def get_env_class_and_freq(gen_name):
    if "V1" in gen_name:
        return EnvV1, 30, 240
    elif "V2" in gen_name:
        return EnvV2, 30, 240
    elif "V3" in gen_name:
        return EnvV3, 30, 240
    elif "V4" in gen_name:
        return EnvV4, 30, 240
    elif "60Hz" in gen_name:
        return EnvV5_60Hz, 60, 480
    elif "100Hz" in gen_name:
        return EnvV5_100Hz, 100, 800
    else:
        return EnvV4, 30, 240


# -----------------------------------------------------------------------------
# CORE EVALUATION ENGINE FOR A SINGLE MODEL
# -----------------------------------------------------------------------------
def evaluate_single_generation(gen_name, model_path, gui=True):
    EnvClass, ctrl_freq, max_steps = get_env_class_and_freq(gen_name)

    if EnvClass is None:
        print(f"[SKIP] Environment class for {gen_name} is not loaded.")
        return None

    print("\n" + "=" * 105)
    print(f"   EVALUATING MODEL GENERATION: [{gen_name}]")
    print(f"   Checkpoint Path: {model_path}")
    print(f"   Control Frequency: {ctrl_freq} Hz | Max Steps Horizon: {max_steps} ({max_steps/ctrl_freq:.1f}s)")
    print("=" * 105)

    raw_env = EnvClass(gui=gui, stage=3)
    raw_env.MAX_STEPS = max_steps
    raw_env.EPISODE_LEN_SEC = max_steps / ctrl_freq

    env = ActionShapeAdapter(raw_env, target_shape=(1, 4))

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0, cameraYaw=45, cameraPitch=-30,
            cameraTargetPosition=[0.0, 0.0, 1.0], physicsClientId=raw_env.CLIENT
        )

    try:
        model = PPO.load(
            model_path, env=env,
            custom_objects={"observation_space": env.observation_space, "action_space": env.action_space}
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        raw_env.close()
        return None

    case_evals = []
    step_dt = 1.0 / float(ctrl_freq)
    target_sleep_dt = step_dt / PLAYBACK_SPEED

    for case in ALL_EXTREME_CASES:
        c_id, c_name, c_cat = case["id"], case["name"], case["cat"]

        if PAUSE_BETWEEN_CASES and gui:
            input(f"\n[PAUSE] Press [ENTER] to execute Case {c_id:02d}: {c_name}...")

        obs, info = env.reset()

        raw_env.INIT_XYZS = np.array([case["pos"]], dtype=np.float32)
        raw_env.INIT_RPYS = np.array([case["rpy"]], dtype=np.float32)

        p.resetBasePositionAndOrientation(
            raw_env.DRONE_IDS[0], case["pos"].tolist(),
            p.getQuaternionFromEuler(case["rpy"]), physicsClientId=raw_env.CLIENT
        )
        p.resetBaseVelocity(
            raw_env.DRONE_IDS[0], linearVelocity=case["vel"].tolist(),
            angularVelocity=case["ang_vel"].tolist(), physicsClientId=raw_env.CLIENT
        )

        obs = raw_env._computeObs()

        pos_hist, rpy_hist, act_hist, dist_hist = [], [], [], []
        time_to_5cm, time_to_2cm = None, None

        for step in range(max_steps):
            t_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            action_flat = np.array(action, dtype=np.float32).flatten()

            obs, reward, terminated, truncated, info = env.step(action_flat)

            pos = obs[0:3]
            rpy = obs[3:6] if len(obs) >= 6 else np.zeros(3)

            pos_hist.append(pos)
            rpy_hist.append(rpy)
            act_hist.append(action_flat)

            dist_cm = np.linalg.norm(pos - np.array([0.0, 0.0, 1.0])) * 100.0
            dist_hist.append(dist_cm)

            curr_t = (step + 1) * step_dt
            if dist_cm < 5.0 and time_to_5cm is None:
                time_to_5cm = curr_t
            if dist_cm < 2.0 and time_to_2cm is None:
                time_to_2cm = curr_t

            if gui:
                t_elapsed = time.time() - t_start
                if t_elapsed < target_sleep_dt:
                    time.sleep(target_sleep_dt - t_elapsed)

            if terminated or truncated:
                break

        pos_arr = np.array(pos_hist)
        rpy_arr = np.array(rpy_hist) * (180.0 / np.pi)
        act_arr = np.array(act_hist)
        dist_arr = np.array(dist_hist)
        time_arr = np.arange(1, len(dist_arr) + 1) * step_dt

        # Strict Physical Verification
        steady_start_idx = int(1.0 * ctrl_freq)
        hover_dists = dist_arr[steady_start_idx:] if len(dist_arr) > steady_start_idx else dist_arr
        steady_rmse_cm = np.sqrt(np.mean((hover_dists / 100.0) ** 2)) * 100.0
        in_5cm_pct = (np.sum(hover_dists < 5.0) / len(hover_dists)) * 100.0
        in_2cm_pct = (np.sum(hover_dists < 2.0) / len(hover_dists)) * 100.0

        act_diffs = np.diff(act_arr, axis=0)
        chatter_var = float(np.mean(act_diffs ** 2)) if len(act_diffs) > 0 else 0.0

        min_alt = float(np.min(pos_arr[:, 2]))
        final_alt = float(np.mean(pos_arr[-int(ctrl_freq*0.5):, 2])) if len(pos_arr) >= int(ctrl_freq*0.5) else float(pos_arr[-1, 2])

        crashed_floor = min_alt < 0.275 or final_alt < 0.35
        never_reached_target = time_to_5cm is None
        unstable_hover = steady_rmse_cm > 25.0

        survived = (not crashed_floor) and (not never_reached_target) and (not unstable_hover)

        t_5cm_str = f"{time_to_5cm:.2f} s" if time_to_5cm is not None else "N/A"
        t_2cm_str = f"{time_to_2cm:.2f} s" if time_to_2cm is not None else "N/A"
        status_str = "PASS" if survived else "FAIL"

        print(f"  [Case {c_id:02d}/16] {c_cat:<16} | Status: {status_str:<4} | Rec (<5cm): {t_5cm_str:<6} | RMSE: {steady_rmse_cm:6.2f} cm | Sub-2cm: {in_2cm_pct:5.1f}% | Chatter: {chatter_var:.6f}")

        case_evals.append({
            "id": c_id, "cat": c_cat, "name": c_name, "survived": survived,
            "init_dist": dist_arr[0], "t_5cm": t_5cm_str, "t_2cm": t_2cm_str,
            "rmse_cm": steady_rmse_cm, "in_5cm": in_5cm_pct, "in_2cm": in_2cm_pct,
            "chatter": chatter_var, "min_alt": min_alt, "pos": pos_arr, "rpy": rpy_arr,
            "act": act_arr, "dist": dist_arr, "time": time_arr
        })

    env.close()

    total_cases = len(case_evals)
    passed_cases = sum(1 for r in case_evals if r["survived"])
    surv_rate = (passed_cases / total_cases) * 100.0

    mean_rmse = float(np.mean([r["rmse_cm"] for r in case_evals]))
    mean_5cm = float(np.mean([r["in_5cm"] for r in case_evals]))
    mean_2cm = float(np.mean([r["in_2cm"] for r in case_evals]))
    mean_chatter = float(np.mean([r["chatter"] for r in case_evals]))

    return {
        "gen_name": gen_name,
        "model_path": model_path,
        "ctrl_freq": ctrl_freq,
        "surv_rate": surv_rate,
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        "mean_rmse": mean_rmse,
        "mean_5cm": mean_5cm,
        "mean_2cm": mean_2cm,
        "mean_chatter": mean_chatter,
        "cases": case_evals
    }


# -----------------------------------------------------------------------------
# MASTER CROSS-GENERATION COMPARATIVE DASHBOARD (PLOTS)
# -----------------------------------------------------------------------------
def generate_master_comparative_plots(all_gen_results):
    """Generates comprehensive matplotlib diagnostic figures comparing all generations."""
    if not all_gen_results:
        print("[WARN] No evaluation data available to plot.")
        return

    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(22, 13))
    fig.suptitle("Quadcopter RL Control Evolution (V1 to V5) — Master In-Depth Performance Dashboard", fontsize=16, fontweight='bold')

    num_gens = len(all_gen_results)
    gen_colors = plt.cm.plasma(np.linspace(0.1, 0.9, num_gens))

    # Panel 1: Survival Rate Comparison Across Generations (Bar Chart)
    ax1 = fig.add_subplot(2, 3, 1)
    gen_names = [r["gen_name"] for r in all_gen_results]
    surv_rates = [r["surv_rate"] for r in all_gen_results]
    bars = ax1.bar(gen_names, surv_rates, color=gen_colors, width=0.55, edgecolor='black', alpha=0.85)
    ax1.set_title("1. Global 16-Case Survival Rate (%)", fontweight='bold')
    ax1.set_ylabel("Survival Rate (%)")
    ax1.set_ylim(0, 110)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.xticks(rotation=20, ha='right', fontsize=8)

    # Panel 2: Average Hover Tracking RMSE (cm) Across Generations
    ax2 = fig.add_subplot(2, 3, 2)
    rmse_vals = [r["mean_rmse"] for r in all_gen_results]
    bars2 = ax2.bar(gen_names, rmse_vals, color=gen_colors, width=0.55, edgecolor='black', alpha=0.85)
    ax2.set_title("2. Steady-State Hover RMSE (cm) [Lower is Better]", fontweight='bold')
    ax2.set_ylabel("RMSE (cm)")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}cm", ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=20, ha='right', fontsize=8)

    # Panel 3: Trajectory Error Convergence (Case 16 Ultimate Adversary)
    ax3 = fig.add_subplot(2, 3, 3)
    for idx, gen_res in enumerate(all_gen_results):
        c16 = [c for c in gen_res["cases"] if c["id"] == 16]
        if c16:
            res = c16[0]
            ax3.plot(res["time"], res["dist"], label=gen_res["gen_name"], color=gen_colors[idx], lw=1.8, alpha=0.9)
    ax3.axhline(5.0, color='r', linestyle='--', alpha=0.7, label="5cm Target Sphere")
    ax3.axhline(2.0, color='g', linestyle=':', alpha=0.7, label="2cm Goal Zone")
    ax3.set_title("3. Case 16 Trajectory Error (cm) vs Time", fontweight='bold')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("3D Distance Error (cm)")
    ax3.set_yscale('log')
    ax3.grid(True, which="both", ls="--", alpha=0.3)
    ax3.legend(fontsize=7, loc='upper right')

    # Panel 4: Case 12 Low-Altitude Sink Recovery Profiles (Z Altitude vs Time)
    ax4 = fig.add_subplot(2, 3, 4)
    for idx, gen_res in enumerate(all_gen_results):
        c12 = [c for c in gen_res["cases"] if c["id"] == 12]
        if c12:
            res = c12[0]
            ax4.plot(res["time"], res["pos"][:, 2], label=gen_res["gen_name"], color=gen_colors[idx], lw=1.8, alpha=0.9)
    ax4.axhline(1.0, color='r', linestyle='--', alpha=0.8, label="Target Z=1.0m")
    ax4.axhline(0.2739, color='black', linestyle='-', alpha=0.9, label="Floor Collision Threshold")
    ax4.set_title("4. Case 12 Sink Recovery: Altitude Z (m) vs Time", fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Altitude Z (m)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=7)

    # Panel 5: Motor Action Signal & Control Chatter Comparison
    ax5 = fig.add_subplot(2, 3, 5)
    chatter_vals = [r["mean_chatter"] for r in all_gen_results]
    bars5 = ax5.bar(gen_names, chatter_vals, color=gen_colors, width=0.55, edgecolor='black', alpha=0.85)
    ax5.set_title("5. Motor Command Chatter Variance Var(du)", fontweight='bold')
    ax5.set_ylabel("Chatter Variance")
    for bar in bars5:
        yval = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0005, f"{yval:.5f}", ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=20, ha='right', fontsize=8)

    # Panel 6: Master Comparative Summary Matrix Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    table_data = [["Gen Version", "Survival", "Hover RMSE", "Sub-2cm %", "Chatter Var"]]
    for r in all_gen_results:
        table_data.append([
            r["gen_name"][:16],
            f"{r['surv_rate']:.1f}%",
            f"{r['mean_rmse']:.2f}cm",
            f"{r['mean_2cm']:.1f}%",
            f"{r['mean_chatter']:.5f}"
        ])

    table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.3)
    ax6.set_title("6. Master Performance Matrix", fontweight='bold')

    plt.tight_layout()
    print("\n" + "=" * 105)
    print(" Displaying Master Comparative Diagnostic Plot Dashboard...")
    print("=" * 105)
    plt.show()


# -----------------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# -----------------------------------------------------------------------------
def run_master_evaluation():
    print("\n" + "=" * 105)
    print("   MASTER CROSS-GENERATION EVALUATION & VISUALIZATION PIPELINE (V1 TO V5)")
    print("=" * 105)

    registry = discover_model_checkpoints()

    if not registry:
        print("\n[WARN] No model checkpoints were automatically discovered.")
        print("Please verify that your checkpoint files are located inside standard checkpoint directories:")
        print(" -> curriculum_checkpoints/stage3_3m_ppo_model.zip (V1)")
        print(" -> curriculum_checkpoints_v2/v2_stage2_3m_ppo_model.zip (V2)")
        print(" -> curriculum_checkpoints_v3/v3_stage2_3m_ppo_model.zip (V3)")
        print(" -> curriculum_checkpoints_v4_stage3_extended/v4_stage3_extended_2m_model.zip (V4)")
        print(" -> curriculum_checkpoints_v5_60hz/v5_60hz_4m_model.zip (V5 60Hz)")
        print(" -> curriculum_checkpoints_v5_100hz/v5_100hz_6.6m_model.zip (V5 100Hz)")
        return

    print(" Discovered Model Checkpoints:")
    for name, path in registry.items():
        print(f"  * [{name}]: {path}")

    all_results = []

    for gen_name, model_path in registry.items():
        res = evaluate_single_generation(gen_name, model_path, gui=ENABLE_3D_GUI)
        if res:
            all_results.append(res)

    # Print Master Side-by-Side Comparison Matrix to Terminal
    if all_results:
        print("\n" + "=" * 105)
        print("                      CONSOLIDATED MASTER CROSS-GENERATIONAL COMPARISON REPORT")
        print("=" * 105)
        print(f"{'Model Generation':<22} | {'Survival %':<12} | {'Pass Cases':<12} | {'Hover RMSE':<12} | {'Sub-2cm %':<10} | {'Chatter Var':<12}")
        print("-" * 105)
        for r in all_results:
            print(f"{r['gen_name']:<22} | {r['surv_rate']:10.1f} % | {r['passed_cases']:2d}/{r['total_cases']:2d} cases  | {r['mean_rmse']:8.2f} cm | {r['mean_2cm']:8.1f} % | {r['mean_chatter']:10.6f}")
        print("=" * 105 + "\n")

        # Display Diagnostic Dashboard Plots
        generate_master_comparative_plots(all_results)


if __name__ == "__main__":
    run_master_evaluation()