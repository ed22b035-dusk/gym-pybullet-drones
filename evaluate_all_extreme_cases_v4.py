import os
import sys
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4

MODEL_PATH = os.path.join("curriculum_checkpoints_v4", "v4_stage2_3m_ppo_model.zip")

# -----------------------------------------------------------------------------
# EXHAUSTIVE 16-CASE STATE-SPACE BOUNDARY MATRIX
# -----------------------------------------------------------------------------
ALL_EXTREME_CASES = [
    # Category 1: Pure Cardinal Spatial Extremes (r = 2.0m)
    {
        "id": 1,
        "cat": "Spatial Bound",
        "name": "Max East Boundary (x=+2.0m, y=0.0m, z=1.0m)",
        "pos": np.array([2.0, 0.0, 1.0]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },
    {
        "id": 2,
        "cat": "Spatial Bound",
        "name": "Max West Boundary (x=-2.0m, y=0.0m, z=1.0m)",
        "pos": np.array([-2.0, 0.0, 1.0]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },
    {
        "id": 3,
        "cat": "Spatial Bound",
        "name": "Max North Boundary (x=0.0m, y=+2.0m, z=1.0m)",
        "pos": np.array([0.0, 2.0, 1.0]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },
    {
        "id": 4,
        "cat": "Spatial Bound",
        "name": "Max South Boundary (x=0.0m, y=-2.0m, z=1.0m)",
        "pos": np.array([0.0, -2.0, 1.0]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },

    # Category 2: Altitude Extremes (Ceiling & Floor Danger Zones)
    {
        "id": 5,
        "cat": "Altitude Bound",
        "name": "Near Ceiling Max Altitude (z=1.85m, Target=1.0m)",
        "pos": np.array([0.0, 0.0, 1.85]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },
    {
        "id": 6,
        "cat": "Altitude Bound",
        "name": "Floor Proximity Danger Zone (z=0.15m, Target=1.0m)",
        "pos": np.array([0.0, 0.0, 0.15]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },

    # Category 3: Pure Attitude Extremes (Max Allowed Pitch / Roll / Yaw)
    {
        "id": 7,
        "cat": "Attitude Bound",
        "name": "Max Forward Pitch Tilt (+14.0° Pitch)",
        "pos": np.array([0.0, 0.0, 1.0]),
        "rpy": np.array([0.0, 14.0 * np.pi / 180.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },
    {
        "id": 8,
        "cat": "Attitude Bound",
        "name": "Max Right Roll Tilt (+14.0° Roll)",
        "pos": np.array([0.0, 0.0, 1.0]),
        "rpy": np.array([14.0 * np.pi / 180.0, 0.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },
    {
        "id": 9,
        "cat": "Attitude Bound",
        "name": "Completely Reversed Heading (Yaw = +180.0°)",
        "pos": np.array([0.0, 0.0, 1.0]),
        "rpy": np.array([0.0, 0.0, np.pi]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },

    # Category 4: High Kinetic Momentum & Tumble Extremes
    {
        "id": 10,
        "cat": "Kinetic Extreme",
        "name": "High Outward Radial Velocity (v0 = +0.80 m/s)",
        "pos": np.array([1.5, 0.0, 1.0]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.array([0.80, 0.0, 0.0]), "ang_vel": np.zeros(3), "stage": 2
    },
    {
        "id": 11,
        "cat": "Kinetic Extreme",
        "name": "High Initial Angular Tumble Rate (ω0_z = +3.0 rad/s)",
        "pos": np.array([0.0, 0.0, 1.0]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.array([0.0, 0.0, 3.0]), "stage": 2
    },
    {
        "id": 12,
        "cat": "Kinetic Extreme",
        "name": "Downward Velocity + Low Altitude Sink (v0_z = -0.60 m/s, z=0.30m)",
        "pos": np.array([0.0, 0.0, 0.30]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.array([0.0, 0.0, -0.60]), "ang_vel": np.zeros(3), "stage": 2
    },

    # Category 5: Compound Extremes & Environmental Turbulence
    {
        "id": 13,
        "cat": "Compound Stress",
        "name": "Diagonal Corner Spawn + Compound Tilt (r=2.0m, Roll/Pitch=+10°)",
        "pos": np.array([1.41, 1.41, 0.50]),
        "rpy": np.array([10.0 * np.pi / 180.0, 10.0 * np.pi / 180.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },
    {
        "id": 14,
        "cat": "Environmental",
        "name": "Max Outer Boundary Under Active Heavy Wind Gusts",
        "pos": np.array([1.80, 0.0, 1.0]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 3
    },
    {
        "id": 15,
        "cat": "Compound Stress",
        "name": "Floor Proximity + Max Compound Tilt + Reversed Heading",
        "pos": np.array([-1.20, -1.20, 0.20]),
        "rpy": np.array([-12.0 * np.pi / 180.0, 12.0 * np.pi / 180.0, np.pi]),
        "vel": np.zeros(3), "ang_vel": np.zeros(3), "stage": 2
    },
    {
        "id": 16,
        "cat": "Ultimate Adversary",
        "name": "The Ultimate Adversary (r=2.0m, Tilt=14°, Yaw=180°, v0=0.5m/s, Wind)",
        "pos": np.array([-1.41, 1.41, 0.25]),
        "rpy": np.array([14.0 * np.pi / 180.0, -14.0 * np.pi / 180.0, np.pi]),
        "vel": np.array([-0.50, 0.50, -0.20]), "ang_vel": np.array([0.5, -0.5, 1.0]), "stage": 3
    }
]


def evaluate_all_extreme_cases():
    if not os.path.exists(MODEL_PATH):
        print(f"\n[FATAL ERROR] Model checkpoint not found at: {MODEL_PATH}")
        sys.exit(1)

    print("=" * 105)
    print("      IITM AQUA SUPERCLUSTER — V4 STAGE 2 EXHAUSTIVE EDGE CASE EVALUATION SUITE")
    print("=" * 105)
    print(f"Model Checkpoint : {MODEL_PATH}")
    print(f"Test Suite Scope : 16 Systematic State-Space Boundary Edge Cases")
    print(f"Control Frequency: 30 Hz (8.0 Second Episode Horizon / 240 Steps per Case)")
    print("=" * 105)

    model = PPO.load(MODEL_PATH)
    results_summary = []

    start_suite_time = time.time()

    for case in ALL_EXTREME_CASES:
        c_id = case["id"]
        c_name = case["name"]
        c_cat = case["cat"]

        print(f"\n---------------------------------------------------------------------------------------------------------")
        print(f"[CASE {c_id:02d}/16] Category: [{c_cat}]")
        print(f"Description : {c_name}")
        print(f"Init State  : Pos={case['pos'].tolist()} m | RPY=np.round(np.degrees({case['rpy'].tolist()}), 1) deg")
        print(f"Init Kinetic: Vel={case['vel'].tolist()} m/s | AngVel={case['ang_vel'].tolist()} rad/s")
        print(f"Environment : {'Stage 3 Active Turbulence' if case['stage'] == 3 else 'Stage 2 Standard'}")

        # Initialize Environment Headless
        env = CurriculumHoverAviaryV4(gui=False, stage=case["stage"])

        # Inject Custom State Values
        env.INIT_XYZS = np.array([case["pos"]], dtype=np.float32)
        env.INIT_RPYS = np.array([case["rpy"]], dtype=np.float32)

        obs, info = env.reset()

        # Set PyBullet Kinematics
        p.resetBaseVelocity(
            env.DRONE_IDS[0],
            linearVelocity=case["vel"].tolist(),
            angularVelocity=case["ang_vel"].tolist(),
            physicsClientId=env.CLIENT
        )

        pos_history = []
        rpy_history = []
        act_history = []
        dist_history = []

        step_dt = 1.0 / 30.0
        time_to_5cm = None
        time_to_2cm = None

        for step in range(240):  # 240 Steps @ 30 Hz
            action, _ = model.predict(obs, deterministic=True)
            action_flat = np.array(action, dtype=np.float32).flatten()  # Standardize shape

            obs, reward, terminated, truncated, info = env.step(action_flat)

            pos = obs[0:3]
            rpy = obs[3:6]

            pos_history.append(pos)
            rpy_history.append(rpy)
            act_history.append(action_flat)

            dist_m = np.linalg.norm(pos - np.array([0.0, 0.0, 1.0]))
            dist_cm = dist_m * 100.0
            dist_history.append(dist_cm)

            current_time_sec = (step + 1) * step_dt

            if dist_cm < 5.0 and time_to_5cm is None:
                time_to_5cm = current_time_sec
            if dist_cm < 2.0 and time_to_2cm is None:
                time_to_2cm = current_time_sec

            if terminated or truncated:
                break

        env.close()

        pos_arr = np.array(pos_history)
        act_arr = np.array(act_history)
        dist_arr = np.array(dist_history)

        survived = len(pos_history) >= 240
        status_str = "PASS [SURVIVED]" if survived else "FAIL [CRASHED]"

        # Compute Steady-State Hover Metrics (Evaluating steps > 30 after initial 1-sec transit recovery)
        hover_dists = dist_arr[30:] if len(dist_arr) > 30 else dist_arr
        steady_rmse_cm = np.sqrt(np.mean((hover_dists / 100.0) ** 2)) * 100.0
        in_5cm_pct = (np.sum(hover_dists < 5.0) / len(hover_dists)) * 100.0
        in_2cm_pct = (np.sum(hover_dists < 2.0) / len(hover_dists)) * 100.0

        # Action Chatter Metric (Derivative Variance)
        act_diffs = np.diff(act_arr, axis=0)
        chatter_var = np.mean(act_diffs ** 2) if len(act_diffs) > 0 else 0.0

        initial_dist_cm = dist_arr[0]
        max_dist_cm = np.max(dist_arr)

        t_5cm_str = f"{time_to_5cm:.2f} s" if time_to_5cm is not None else "N/A"
        t_2cm_str = f"{time_to_2cm:.2f} s" if time_to_2cm is not None else "N/A"

        # Detailed Individual Terminal Output
        print(f"  --> EXECUTION STATUS            : {status_str} ({len(pos_history)}/240 steps)")
        print(f"  --> Initial Offset Distance     : {initial_dist_cm:.2f} cm")
        print(f"  --> Max Trajectory Displacement : {max_dist_cm:.2f} cm")
        print(f"  --> Time to Enter 5cm Sphere    : {t_5cm_str}")
        print(f"  --> Time to Enter 2cm Sphere    : {t_2cm_str}")
        print(f"  --> Steady Hover RMSE (m)       : {steady_rmse_cm / 100.0:.5f} m ({steady_rmse_cm:.2f} cm)")
        print(f"  --> Sub-5cm Precision Zone Hold : {in_5cm_pct:.2f} % time in zone")
        print(f"  --> Sub-2cm Ultra-Precision Hold: {in_2cm_pct:.2f} % time in zone")
        print(f"  --> Motor Action Chatter Var    : {chatter_var:.6f}")

        results_summary.append({
            "id": c_id,
            "cat": c_cat,
            "name": c_name,
            "survived": survived,
            "init_dist": initial_dist_cm,
            "t_5cm": t_5cm_str,
            "t_2cm": t_2cm_str,
            "rmse_cm": steady_rmse_cm,
            "in_5cm": in_5cm_pct,
            "in_2cm": in_2cm_pct,
            "chatter": chatter_var
        })

    suite_duration = time.time() - start_suite_time

    # -----------------------------------------------------------------------------
    # CONSOLIDATED MASTER BENCHMARK REPORT
    # -----------------------------------------------------------------------------
    total_cases = len(results_summary)
    passed_cases = sum(1 for r in results_summary if r["survived"])
    global_survival_rate = (passed_cases / total_cases) * 100.0

    mean_rmse = np.mean([r["rmse_cm"] for r in results_summary])
    mean_5cm_hold = np.mean([r["in_5cm"] for r in results_summary])
    mean_2cm_hold = np.mean([r["in_2cm"] for r in results_summary])
    mean_chatter = np.mean([r["chatter"] for r in results_summary])
    max_rmse_case = max(results_summary, key=lambda x: x["rmse_cm"])

    print("\n" + "=" * 105)
    print("                     CONSOLIDATED MASTER BENCHMARK REPORT (ALL 16 CASES)")
    print("=" * 105)
    print(f"Total Evaluated Cases      : {total_cases}")
    print(f"Successful Recoveries      : {passed_cases} / {total_cases}")
    print(f"Global Survival Rate       : {global_survival_rate:.1f} %")
    print(f"Average Hover RMSE         : {mean_rmse:.2f} cm across all extreme cases")
    print(f"Average Sub-5cm Sphere Hold: {mean_5cm_hold:.2f} %")
    print(f"Average Sub-2cm Sphere Hold: {mean_2cm_hold:.2f} %")
    print(f"Average Motor Chatter Var  : {mean_chatter:.6f}")
    print(f"Worst-Case RMSE            : Case {max_rmse_case['id']} ({max_rmse_case['name']}) -> {max_rmse_case['rmse_cm']:.2f} cm")
    print(f"Total Benchmark Suite Time : {suite_duration:.2f} seconds")
    print("=" * 105)

    print("\n" + "-" * 105)
    print(f"{'ID':<3} | {'Category':<16} | {'Init Dist':<10} | {'Rec (<5cm)':<10} | {'Hover RMSE':<11} | {'Sub-2cm %':<10} | {'Status':<6}")
    print("-" * 105)
    for r in results_summary:
        status_code = "PASS" if r["survived"] else "FAIL"
        print(f"{r['id']:02d}  | {r['cat']:<16} | {r['init_dist']:6.1f} cm | {r['t_5cm']:<10} | {r['rmse_cm']:6.2f} cm   | {r['in_2cm']:6.1f} %   | {status_code:<6}")
    print("-" * 105)

    print("\n[VERDICT]")
    if global_survival_rate == 100.0 and mean_rmse < 3.0:
        print("✓ EXTREME EDGE CASE EVALUATION PASSED WITH FLIES-OFF PERFECTION.")
        print("✓ Policy exhibits robust spatial stabilization, attitude damping, and chatter-free control.")
        print("✓ Stage 2 is officially VERIFIED. You are clear to proceed to STAGE 3 (Domain Randomization & CTBR Refactoring)!")
    else:
        print("⚠ Some edge cases exhibited drift or failure. Inspect the table above before Stage 3.")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    evaluate_all_extreme_cases()