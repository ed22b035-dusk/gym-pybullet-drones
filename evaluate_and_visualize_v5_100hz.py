import os
import sys
import glob
import time
import numpy as np
import pybullet as p
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Path resolution for standalone directory execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym-pybullet-drones"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym_pybullet_drones"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym-pybullet-drones", "envs"))

from gym_pybullet_drones.envs.CurriculumHoverAviaryV5_100Hz import CurriculumHoverAviaryV5_100Hz


def find_v5_100hz_model():
    """Searches for the latest 100Hz model checkpoint or final zip."""
    candidate_paths = [
        os.path.join(SCRIPT_DIR, "curriculum_checkpoints_v5_100hz", "v5_100hz_6.6m_model.zip"),
        os.path.join(SCRIPT_DIR, "curriculum_checkpoints_v5_100hz", "ckpts", "*.zip"),
        os.path.join(SCRIPT_DIR, "gym-pybullet-drones", "curriculum_checkpoints_v5_100hz", "ckpts", "*.zip"),
        "**/*v5_100hz*.zip",
        "**/*100hz*.zip"
    ]

    for pattern in candidate_paths:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return os.path.abspath(matches[0])
    return None


MODEL_PATH = find_v5_100hz_model()


class ActionShapeAdapter(gym.Wrapper):
    """Ensures action space shape compatibility between VectorEnv training Box(1, 4) and single env evaluation."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action_flat = np.array(action, dtype=np.float32).flatten()
        return self.env.step(action_flat)


# -----------------------------------------------------------------------------
# EXHAUSTIVE 16-CASE BENCHMARK MATRIX
# -----------------------------------------------------------------------------
ALL_EXTREME_CASES = [
    {"id": 1, "cat": "Spatial Bound", "name": "Max East Boundary (x=+2.0m, y=0.0m, z=1.0m)", "pos": np.array([2.0, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 2, "cat": "Spatial Bound", "name": "Max West Boundary (x=-2.0m, y=0.0m, z=1.0m)", "pos": np.array([-2.0, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 3, "cat": "Spatial Bound", "name": "Max North Boundary (x=0.0m, y=+2.0m, z=1.0m)", "pos": np.array([0.0, 2.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 4, "cat": "Spatial Bound", "name": "Max South Boundary (x=0.0m, y=-2.0m, z=1.0m)", "pos": np.array([0.0, -2.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 5, "cat": "Altitude Bound", "name": "Near Ceiling Max Altitude (z=1.85m, Target=1.0m)", "pos": np.array([0.0, 0.0, 1.85]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 6, "cat": "Altitude Bound", "name": "Floor Proximity Danger Zone (z=0.15m, Target=1.0m)", "pos": np.array([0.0, 0.0, 0.15]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 7, "cat": "Attitude Bound", "name": "Max Forward Pitch Tilt (+15.0° Pitch)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([0.0, 15.0 * np.pi / 180.0, 0.0]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 8, "cat": "Attitude Bound", "name": "Max Right Roll Tilt (+15.0° Roll)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([15.0 * np.pi / 180.0, 0.0, 0.0]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 9, "cat": "Attitude Bound", "name": "Completely Reversed Heading (Yaw = +180.0°)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([0.0, 0.0, np.pi]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 10, "cat": "Kinetic Extreme", "name": "High Outward Linear Velocity (v0_x = +1.00 m/s)", "pos": np.array([1.5, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.array([1.00, 0.0, 0.0]), "ang_vel": np.zeros(3)},
    {"id": 11, "cat": "Kinetic Extreme", "name": "Extreme Angular Tumble Rate (ω0_z = +3.0 rad/s)", "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.array([0.0, 0.0, 3.0])},
    {"id": 12, "cat": "Kinetic Extreme", "name": "Fast Downward Sink Velocity (v0_z = -0.80 m/s, z=0.30m)", "pos": np.array([0.0, 0.0, 0.30]), "rpy": np.zeros(3), "vel": np.array([0.0, 0.0, -0.80]), "ang_vel": np.zeros(3)},
    {"id": 13, "cat": "Compound Stress", "name": "Diagonal Corner Spawn + Compound Tilt (r=2.0m, Roll/Pitch=+12°)", "pos": np.array([1.41, 1.41, 0.50]), "rpy": np.array([12.0 * np.pi / 180.0, 12.0 * np.pi / 180.0, 0.0]), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 14, "cat": "Environmental", "name": "Max Outer Boundary Under Active Wind Gusts", "pos": np.array([1.80, 0.0, 1.0]), "rpy": np.zeros(3), "vel": np.zeros(3), "ang_vel": np.zeros(3)},
    {"id": 15, "cat": "Compound Kinetic", "name": "Floor Proximity + Downward Velocity + Tilt (z=0.20m, v0_z=-0.5m/s)", "pos": np.array([-1.20, -1.20, 0.20]), "rpy": np.array([-12.0 * np.pi / 180.0, 12.0 * np.pi / 180.0, np.pi]), "vel": np.array([0.0, 0.0, -0.50]), "ang_vel": np.zeros(3)},
    {"id": 16, "cat": "Ultimate Adversary", "name": "The Ultimate Adversary (r=2.0m, Tilt=15°, Yaw=180°, v0=[-0.8,0.8,-0.4], ω0=[1,-1,2])", "pos": np.array([-1.41, 1.41, 0.25]), "rpy": np.array([15.0 * np.pi / 180.0, -15.0 * np.pi / 180.0, np.pi]), "vel": np.array([-0.80, 0.80, -0.40]), "ang_vel": np.array([1.0, -1.0, 2.0])}
]


def evaluate_and_visualize_v5_100hz():
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print("\n" + "=" * 80)
        print(" [ERROR] V5 100Hz MODEL CHECKPOINT NOT FOUND!")
        print("=" * 80)
        print(f" Working Directory: {os.getcwd()}")
        sys.exit(1)

    print("=" * 105)
    print("   V5 100Hz CONTROL FREQUENCY (Δt = 10.00 ms) LIVE 3D VISUALIZATION & DIAGNOSTIC SUITE")
    print("=" * 105)
    print(f"Detected Model Checkpoint : {MODEL_PATH}")
    print(f"Control Loop Frequency    : 100 Hz")
    print(f"Test Suite Scope          : 16 Systematic State-Space Edge Cases")
    print("=" * 105)

    raw_env = CurriculumHoverAviaryV5_100Hz(gui=True, stage=3)
    raw_env.MAX_STEPS = 800  # Force step horizon to 800 steps (8.0s @ 100Hz)
    raw_env.EPISODE_LEN_SEC = 8.0

    env = ActionShapeAdapter(raw_env)

    p.resetDebugVisualizerCamera(
        cameraDistance=3.0, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0.0, 0.0, 1.0], physicsClientId=raw_env.CLIENT
    )

    model = PPO.load(
        MODEL_PATH, env=env,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space
        }
    )

    case_results = []
    start_suite_time = time.time()
    MAX_STEPS = 800
    step_dt = 1.0 / 100.0

    for case in ALL_EXTREME_CASES:
        c_id, c_name, c_cat = case["id"], case["name"], case["cat"]

        print(f"\n---------------------------------------------------------------------------------------------------------")
        print(f"[CASE {c_id:02d}/16] Category: [{c_cat}]")
        print(f"Description : {c_name}")

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

        pos_history, rpy_history, act_history, dist_history = [], [], [], []
        time_to_5cm, time_to_2cm = None, None
        crashed = False

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            action_flat = np.array(action, dtype=np.float32).flatten()

            obs, reward, terminated, truncated, info = env.step(action_flat)

            pos = obs[0:3]
            rpy = obs[3:6]

            pos_history.append(pos)
            rpy_history.append(rpy)
            act_history.append(action_flat)

            dist_cm = np.linalg.norm(pos - np.array([0.0, 0.0, 1.0])) * 100.0
            dist_history.append(dist_cm)

            current_time_sec = (step + 1) * step_dt
            if dist_cm < 5.0 and time_to_5cm is None:
                time_to_5cm = current_time_sec
            if dist_cm < 2.0 and time_to_2cm is None:
                time_to_2cm = current_time_sec

            time.sleep(step_dt * 0.5)

            if terminated:
                crashed = True
                break
            if truncated:
                break

        pos_arr = np.array(pos_history)
        rpy_arr = np.array(rpy_history) * (180.0 / np.pi)
        act_arr = np.array(act_history)
        dist_arr = np.array(dist_history)
        time_arr = np.arange(1, len(dist_arr) + 1) * step_dt

        survived = not crashed
        status_str = "PASS [SURVIVED]" if survived else "FAIL [CRASHED]"

        hover_dists = dist_arr[100:] if len(dist_arr) > 100 else dist_arr
        steady_rmse_cm = np.sqrt(np.mean((hover_dists / 100.0) ** 2)) * 100.0
        in_5cm_pct = (np.sum(hover_dists < 5.0) / len(hover_dists)) * 100.0
        in_2cm_pct = (np.sum(hover_dists < 2.0) / len(hover_dists)) * 100.0

        act_diffs = np.diff(act_arr, axis=0)
        chatter_var = np.mean(act_diffs ** 2) if len(act_diffs) > 0 else 0.0

        t_5cm_str = f"{time_to_5cm:.2f} s" if time_to_5cm is not None else "N/A"
        t_2cm_str = f"{time_to_2cm:.2f} s" if time_to_2cm is not None else "N/A"

        print(f"  --> EXECUTION STATUS            : {status_str} ({len(pos_history)} steps executed)")
        print(f"  --> Initial Offset Distance     : {dist_arr[0]:.2f} cm")
        print(f"  --> Time to Enter 5cm Sphere    : {t_5cm_str}")
        print(f"  --> Time to Enter 2cm Sphere    : {t_2cm_str}")
        print(f"  --> Steady Hover RMSE (cm)      : {steady_rmse_cm:.2f} cm")
        print(f"  --> Sub-2cm Ultra-Precision Hold: {in_2cm_pct:.2f} %")

        case_results.append({
            "id": c_id, "cat": c_cat, "name": c_name, "survived": survived,
            "init_dist": dist_arr[0], "t_5cm": t_5cm_str, "t_2cm": t_2cm_str,
            "rmse_cm": steady_rmse_cm, "in_5cm": in_5cm_pct, "in_2cm": in_2cm_pct,
            "chatter": chatter_var, "pos": pos_arr, "rpy": rpy_arr, "act": act_arr,
            "dist": dist_arr, "time": time_arr
        })

        time.sleep(0.1)

    env.close()
    suite_duration = time.time() - start_suite_time

    total_cases = len(case_results)
    passed_cases = sum(1 for r in case_results if r["survived"])
    global_survival_rate = (passed_cases / total_cases) * 100.0

    mean_rmse = np.mean([r["rmse_cm"] for r in case_results])
    mean_5cm_hold = np.mean([r["in_5cm"] for r in case_results])
    mean_2cm_hold = np.mean([r["in_2cm"] for r in case_results])
    mean_chatter = np.mean([r["chatter"] for r in case_results])
    max_rmse_case = max(case_results, key=lambda x: x["rmse_cm"])

    print("\n" + "=" * 105)
    print("                     CONSOLIDATED MASTER BENCHMARK REPORT (V5 100Hz EVALUATION)")
    print("=" * 105)
    print(f"Total Evaluated Cases      : {total_cases}")
    print(f"Successful Recoveries      : {passed_cases} / {total_cases}")
    print(f"Global Survival Rate       : {global_survival_rate:.1f} %")
    print(f"Average Hover RMSE         : {mean_rmse:.2f} cm")
    print(f"Average Sub-5cm Sphere Hold: {mean_5cm_hold:.2f} %")
    print(f"Average Sub-2cm Sphere Hold: {mean_2cm_hold:.2f} %")
    print(f"Average Motor Chatter Var  : {mean_chatter:.6f}")
    print(f"Worst-Case RMSE            : Case {max_rmse_case['id']} ({max_rmse_case['name']}) -> {max_rmse_case['rmse_cm']:.2f} cm")
    print(f"Total Benchmark Suite Time : {suite_duration:.2f} seconds")
    print("=" * 105)

    print("\n" + "-" * 105)
    print(f"{'ID':<3} | {'Category':<16} | {'Init Dist':<10} | {'Rec (<5cm)':<10} | {'Hover RMSE':<11} | {'Sub-2cm %':<10} | {'Status':<6}")
    print("-" * 105)
    for r in case_results:
        status_code = "PASS" if r["survived"] else "FAIL"
        print(f"{r['id']:02d}  | {r['cat']:<16} | {r['init_dist']:6.1f} cm | {r['t_5cm']:<10} | {r['rmse_cm']:6.2f} cm   | {r['in_2cm']:6.1f} %   | {status_code:<6}")
    print("-" * 105 + "\n")

    # Diagnostic Dashboard Pop-up
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("V5 100Hz Evaluation — Corrected Diagnostic Dashboard", fontsize=16, fontweight='bold')
    colors = plt.cm.tab20(np.linspace(0, 1, 16))

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for idx, res in enumerate(case_results):
        p_arr = res["pos"]
        ax1.plot(p_arr[:, 0], p_arr[:, 1], p_arr[:, 2], color=colors[idx], lw=1.2, alpha=0.8)
    ax1.scatter([0], [0], [1.0], color='r', s=150, marker='*', label="Target (0,0,1)")
    ax1.set_title("3D Trajectories (100Hz)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")

    ax2 = fig.add_subplot(2, 3, 2)
    for idx, res in enumerate(case_results):
        ax2.plot(res["time"], res["dist"], color=colors[idx], lw=1.2, alpha=0.8)
    ax2.axhline(5.0, color='r', linestyle='--', alpha=0.7, label="5cm Sphere")
    ax2.axhline(2.0, color='g', linestyle=':', alpha=0.7, label="2cm Goal")
    ax2.set_title("3D Tracking Error Convergence (cm)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (cm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 3, 3)
    for idx, res in enumerate(case_results):
        ax3.plot(res["time"], res["pos"][:, 2], color=colors[idx], lw=1.2, alpha=0.8)
    ax3.axhline(1.0, color='r', linestyle='--', alpha=0.8, label="Target Z (1.0m)")
    ax3.set_title("Altitude (Z) Profiles (m)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Z Altitude (m)")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4)
    c1 = case_results[0]
    ax4.plot(c1["time"], c1["rpy"][:, 0], label="Roll φ", color='c')
    ax4.plot(c1["time"], c1["rpy"][:, 1], label="Pitch θ", color='m')
    ax4.plot(c1["time"], c1["rpy"][:, 2], label="Yaw ψ", color='y')
    ax4.set_title("Case 01 Attitude Damping (deg)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angle (deg)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(2, 3, 5)
    for i in range(4):
        ax5.plot(c1["time"], c1["act"][:, i], label=f"Rotor u_{i+1}", alpha=0.8)
    ax5.set_title("Case 01 Rotor Commands [-1, +1]")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Action Signal")
    ax5.grid(True, alpha=0.3)
    ax5.legend(ncol=2, fontsize=8)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    table_data = [["ID", "Category", "Init Dist", "RMSE", "Status"]]
    for r in case_results:
        table_data.append([f"C{r['id']:02d}", r["cat"][:10], f"{r['init_dist']:.1f}cm", f"{r['rmse_cm']:.2f}cm", "PASS" if r["survived"] else "FAIL"])
    table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.15)
    ax6.set_title("V5 100Hz Corrected Metric Matrix", fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_and_visualize_v5_100hz()