import os
import sys
import time
import numpy as np
import pybullet as p
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4

# Check for Stage 2 checkpoint locations
MODEL_PATH = os.path.join("curriculum_checkpoints_v4", "v4_stage2_3m_ppo_model.zip")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("master_curriculum_ckpts", "v4_stage2_model.zip")


class ActionShapeAdapter(gym.Wrapper):
    """Guarantees uniform (1, 4) action space metadata for SB3 policy loading."""
    def __init__(self, env, target_act_shape=(1, 4)):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=target_act_shape, dtype=np.float32
        )

    def step(self, action):
        return self.env.step(action)


# -----------------------------------------------------------------------------
# EXHAUSTIVE 16-CASE BENCHMARK MATRIX FOR STAGE 2
# -----------------------------------------------------------------------------
ALL_EXTREME_CASES_STAGE2 = [
    # Category 1: Spatial Extremes (r = 2.0m)
    {
        "id": 1, "cat": "Spatial Bound",
        "name": "Max East Boundary (x=+2.0m, y=0.0m, z=1.0m)",
        "pos": np.array([2.0, 0.0, 1.0]), "rpy": np.zeros(3),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },
    {
        "id": 2, "cat": "Spatial Bound",
        "name": "Max West Boundary (x=-2.0m, y=0.0m, z=1.0m)",
        "pos": np.array([-2.0, 0.0, 1.0]), "rpy": np.zeros(3),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },
    {
        "id": 3, "cat": "Spatial Bound",
        "name": "Max North Boundary (x=0.0m, y=+2.0m, z=1.0m)",
        "pos": np.array([0.0, 2.0, 1.0]), "rpy": np.zeros(3),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },
    {
        "id": 4, "cat": "Spatial Bound",
        "name": "Max South Boundary (x=0.0m, y=-2.0m, z=1.0m)",
        "pos": np.array([0.0, -2.0, 1.0]), "rpy": np.zeros(3),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },

    # Category 2: Altitude Danger Zones
    {
        "id": 5, "cat": "Altitude Bound",
        "name": "Near Ceiling Max Altitude (z=1.85m, Target=1.0m)",
        "pos": np.array([0.0, 0.0, 1.85]), "rpy": np.zeros(3),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },
    {
        "id": 6, "cat": "Altitude Bound",
        "name": "Floor Proximity Danger Zone (z=0.15m, Target=1.0m)",
        "pos": np.array([0.0, 0.0, 0.15]), "rpy": np.zeros(3),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },

    # Category 3: Attitude Extremes
    {
        "id": 7, "cat": "Attitude Bound",
        "name": "Max Forward Pitch Tilt (+14.0° Pitch)",
        "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([0.0, 14.0 * np.pi / 180.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },
    {
        "id": 8, "cat": "Attitude Bound",
        "name": "Max Right Roll Tilt (+14.0° Roll)",
        "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([14.0 * np.pi / 180.0, 0.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },
    {
        "id": 9, "cat": "Attitude Bound",
        "name": "Completely Reversed Heading (Yaw = +180.0°)",
        "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.array([0.0, 0.0, np.pi]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },

    # Category 4: Kinetic Momentum & Tumble Extremes
    {
        "id": 10, "cat": "Kinetic Extreme",
        "name": "High Outward Linear Velocity (v0_x = +0.80 m/s)",
        "pos": np.array([1.5, 0.0, 1.0]), "rpy": np.zeros(3),
        "vel": np.array([0.80, 0.0, 0.0]), "ang_vel": np.zeros(3)
    },
    {
        "id": 11, "cat": "Kinetic Extreme",
        "name": "Extreme Angular Tumble Rate (ω0_z = +3.0 rad/s)",
        "pos": np.array([0.0, 0.0, 1.0]), "rpy": np.zeros(3),
        "vel": np.zeros(3), "ang_vel": np.array([0.0, 0.0, 3.0])
    },
    {
        "id": 12, "cat": "Kinetic Extreme",
        "name": "Downward Velocity Sink (v0_z = -0.60 m/s, z=0.30m)",
        "pos": np.array([0.0, 0.0, 0.30]), "rpy": np.zeros(3),
        "vel": np.array([0.0, 0.0, -0.60]), "ang_vel": np.zeros(3)
    },

    # Category 5: Compound Stress & Active Stage 2 Wind
    {
        "id": 13, "cat": "Compound Stress",
        "name": "Diagonal Corner Spawn + Compound Tilt (r=2.0m, Roll/Pitch=+10°)",
        "pos": np.array([1.41, 1.41, 0.50]), "rpy": np.array([10.0 * np.pi / 180.0, 10.0 * np.pi / 180.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },
    {
        "id": 14, "cat": "Environmental",
        "name": "Max Outer Boundary Under Active Wind Gusts",
        "pos": np.array([1.80, 0.0, 1.0]), "rpy": np.zeros(3),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },
    {
        "id": 15, "cat": "Compound Stress",
        "name": "Floor Proximity + Max Compound Tilt + Reversed Heading",
        "pos": np.array([-1.20, -1.20, 0.20]), "rpy": np.array([-12.0 * np.pi / 180.0, 12.0 * np.pi / 180.0, np.pi]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3)
    },
    {
        "id": 16, "cat": "Ultimate Adversary",
        "name": "The Ultimate Adversary (r=2.0m, Tilt=14°, Yaw=180°, v0=0.5m/s)",
        "pos": np.array([-1.41, 1.41, 0.25]), "rpy": np.array([14.0 * np.pi / 180.0, -14.0 * np.pi / 180.0, np.pi]),
        "vel": np.array([-0.50, 0.50, -0.20]), "ang_vel": np.array([0.5, -0.5, 1.0])
    }
]


def evaluate_and_visualize_stage2_gui():
    if not os.path.exists(MODEL_PATH):
        print(f"\n[FATAL ERROR] Stage 2 model checkpoint not found at: {MODEL_PATH}")
        sys.exit(1)

    print("=" * 105)
    print("   STAGE 2 LIVE 3D VISUALIZATION & EXHAUSTIVE DIAGNOSTIC EVALUATION SUITE")
    print("=" * 105)
    print(f"Model Checkpoint : {MODEL_PATH}")
    print(f"Test Suite Scope : 16 Systematic State-Space Edge Cases")
    print(f"Visualization    : PyBullet 3D GUI Active (30 Hz Real-Time Pacing)")
    print("=" * 105)

    raw_env = CurriculumHoverAviaryV4(gui=True, stage=2)
    env = ActionShapeAdapter(raw_env)

    p.resetDebugVisualizerCamera(
        cameraDistance=3.0, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0.0, 0.0, 1.0], physicsClientId=raw_env.CLIENT
    )

    model = PPO.load(
        MODEL_PATH, env=env,
        custom_objects={"action_space": env.action_space}
    )

    case_results = []
    start_suite_time = time.time()

    for case in ALL_EXTREME_CASES_STAGE2:
        c_id, c_name, c_cat = case["id"], case["name"], case["cat"]

        print(f"\n---------------------------------------------------------------------------------------------------------")
        print(f"[CASE {c_id:02d}/16] Category: [{c_cat}]")
        print(f"Description : {c_name}")
        print(f"Init State  : Pos={case['pos'].tolist()} m | RPY={np.round(np.degrees(case['rpy']), 1).tolist()} deg")
        print(f"Init Kinetic: Vel={case['vel'].tolist()} m/s | AngVel={case['ang_vel'].tolist()} rad/s")

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
        step_dt = 1.0 / 30.0
        time_to_5cm, time_to_2cm = None, None

        for step in range(240):
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

            time.sleep(step_dt * 0.75)

            if terminated or truncated:
                break

        pos_arr = np.array(pos_history)
        rpy_arr = np.array(rpy_history) * (180.0 / np.pi)
        act_arr = np.array(act_history)
        dist_arr = np.array(dist_history)
        time_arr = np.arange(1, len(dist_arr) + 1) * step_dt

        survived = len(pos_history) >= 240
        status_str = "PASS [SURVIVED]" if survived else "FAIL [CRASHED]"

        hover_dists = dist_arr[30:] if len(dist_arr) > 30 else dist_arr
        steady_rmse_cm = np.sqrt(np.mean((hover_dists / 100.0) ** 2)) * 100.0
        in_5cm_pct = (np.sum(hover_dists < 5.0) / len(hover_dists)) * 100.0
        in_2cm_pct = (np.sum(hover_dists < 2.0) / len(hover_dists)) * 100.0

        act_diffs = np.diff(act_arr, axis=0)
        chatter_var = np.mean(act_diffs ** 2) if len(act_diffs) > 0 else 0.0

        t_5cm_str = f"{time_to_5cm:.2f} s" if time_to_5cm is not None else "N/A"
        t_2cm_str = f"{time_to_2cm:.2f} s" if time_to_2cm is not None else "N/A"

        print(f"  --> EXECUTION STATUS            : {status_str} ({len(pos_history)}/240 steps)")
        print(f"  --> Initial Offset Distance     : {dist_arr[0]:.2f} cm")
        print(f"  --> Max Trajectory Displacement : {np.max(dist_arr):.2f} cm")
        print(f"  --> Time to Enter 5cm Sphere    : {t_5cm_str}")
        print(f"  --> Time to Enter 2cm Sphere    : {t_2cm_str}")
        print(f"  --> Steady Hover RMSE (m)       : {steady_rmse_cm / 100.0:.5f} m ({steady_rmse_cm:.2f} cm)")
        print(f"  --> Sub-5cm Precision Zone Hold : {in_5cm_pct:.2f} % time in zone")
        print(f"  --> Sub-2cm Ultra-Precision Hold: {in_2cm_pct:.2f} % time in zone")
        print(f"  --> Motor Action Chatter Var    : {chatter_var:.6f}")

        case_results.append({
            "id": c_id, "cat": c_cat, "name": c_name, "survived": survived,
            "init_dist": dist_arr[0], "t_5cm": t_5cm_str, "t_2cm": t_2cm_str,
            "rmse_cm": steady_rmse_cm, "in_5cm": in_5cm_pct, "in_2cm": in_2cm_pct,
            "chatter": chatter_var, "pos": pos_arr, "rpy": rpy_arr, "act": act_arr,
            "dist": dist_arr, "time": time_arr
        })

        time.sleep(0.3)

    env.close()
    suite_duration = time.time() - start_suite_time

    # Master Benchmark Report
    total_cases = len(case_results)
    passed_cases = sum(1 for r in case_results if r["survived"])
    global_survival_rate = (passed_cases / total_cases) * 100.0

    mean_rmse = np.mean([r["rmse_cm"] for r in case_results])
    mean_5cm_hold = np.mean([r["in_5cm"] for r in case_results])
    mean_2cm_hold = np.mean([r["in_2cm"] for r in case_results])
    mean_chatter = np.mean([r["chatter"] for r in case_results])
    max_rmse_case = max(case_results, key=lambda x: x["rmse_cm"])

    print("\n" + "=" * 105)
    print("                     CONSOLIDATED MASTER BENCHMARK REPORT (STAGE 2 ALL 16 CASES)")
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
    fig.suptitle("V4 Stage 2 — Diagnostic Evaluation Dashboard", fontsize=16, fontweight='bold')
    colors = plt.cm.tab20(np.linspace(0, 1, 16))

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for idx, res in enumerate(case_results):
        p_arr = res["pos"]
        ax1.plot(p_arr[:, 0], p_arr[:, 1], p_arr[:, 2], color=colors[idx], lw=1.2, alpha=0.8)
    ax1.scatter([0], [0], [1.0], color='r', s=150, marker='*', label="Target (0,0,1)")
    ax1.set_title("3D Convergence Trajectories (16 Cases)")
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
    c15 = case_results[14]
    ax4.plot(c15["time"], c15["rpy"][:, 0], label="Roll φ", color='c')
    ax4.plot(c15["time"], c15["rpy"][:, 1], label="Pitch θ", color='m')
    ax4.plot(c15["time"], c15["rpy"][:, 2], label="Yaw ψ", color='y')
    ax4.set_title("Case 15 Attitude Damping (deg)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angle (deg)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(2, 3, 5)
    for i in range(4):
        ax5.plot(c15["time"], c15["act"][:, i], label=f"Rotor u_{i+1}", alpha=0.8)
    ax5.set_title("Case 15 Rotor Commands [-1, +1]")
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
    ax6.set_title("Stage 2 Metric Matrix", fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_and_visualize_stage2_gui()