import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4

MODEL_PATH = os.path.join("curriculum_checkpoints_v4", "v4_stage2_3m_ppo_model.zip")


# Define 10 Extreme Test Cases
EXTREME_CASES = [
    {
        "id": 1,
        "name": "Case 1: Max East Radius (x=+2.0m, Pitch=+14°)",
        "pos": np.array([2.0, 0.0, 0.5]),
        "rpy": np.array([0.0, 14.0 * np.pi / 180.0, 0.0]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "wind": False
    },
    {
        "id": 2,
        "name": "Case 2: Max West Radius (x=-2.0m, Pitch=-14°)",
        "pos": np.array([-2.0, 0.0, 0.5]),
        "rpy": np.array([0.0, -14.0 * np.pi / 180.0, 0.0]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "wind": False
    },
    {
        "id": 3,
        "name": "Case 3: Max North Radius (y=+2.0m, Roll=+14°)",
        "pos": np.array([0.0, 2.0, 0.5]),
        "rpy": np.array([14.0 * np.pi / 180.0, 0.0, 0.0]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "wind": False
    },
    {
        "id": 4,
        "name": "Case 4: Max South Radius (y=-2.0m, Roll=-14°)",
        "pos": np.array([0.0, -2.0, 0.5]),
        "rpy": np.array([-14.0 * np.pi / 180.0, 0.0, 0.0]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "wind": False
    },
    {
        "id": 5,
        "name": "Case 5: High Altitude & Reversed Heading (z=1.8m, Yaw=180°)",
        "pos": np.array([1.41, 1.41, 1.8]),
        "rpy": np.array([0.0, 0.0, np.pi]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "wind": False
    },
    {
        "id": 6,
        "name": "Case 6: Floor Proximity & Dual Tilt (z=0.15m, Roll/Pitch=+12°)",
        "pos": np.array([-1.41, -1.41, 0.15]),
        "rpy": np.array([12.0 * np.pi / 180.0, 12.0 * np.pi / 180.0, 0.0]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "wind": False
    },
    {
        "id": 7,
        "name": "Case 7: Corner Spawn & Negative Yaw (r=2.0m, Yaw=-135°)",
        "pos": np.array([1.41, -1.41, 0.4]),
        "rpy": np.array([-10.0 * np.pi / 180.0, 10.0 * np.pi / 180.0, -135.0 * np.pi / 180.0]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "wind": False
    },
    {
        "id": 8,
        "name": "Case 8: Continuous Heavy Cross-Wind Turbulence",
        "pos": np.array([1.5, 0.0, 0.6]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "wind": True
    },
    {
        "id": 9,
        "name": "Case 9: The Ultimate Compound Adversary (r=2.0m, Tilt=14°, Yaw=180°, Wind)",
        "pos": np.array([-1.41, 1.41, 0.25]),
        "rpy": np.array([14.0 * np.pi / 180.0, -14.0 * np.pi / 180.0, np.pi]),
        "vel": np.zeros(3),
        "ang_vel": np.zeros(3),
        "wind": True
    },
    {
        "id": 10,
        "name": "Case 10: High Outward Momentum Drift (v0=0.8m/s, ω0=2.0rad/s)",
        "pos": np.array([1.8, 0.0, 0.8]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "vel": np.array([0.8, 0.5, 0.0]),
        "ang_vel": np.array([0.0, 0.0, 2.0]),
        "wind": False
    }
]


def run_10_extreme_cases():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    print("=" * 85)
    print("   V4 STAGE 2 — 10 EXTREME EDGE CASE VISUALIZATION & BENCHMARK SUITE")
    print("=" * 85)

    model = PPO.load(MODEL_PATH)
    case_results = []

    # -------------------------------------------------------------------------
    # PART 1: LIVE 3D PYBULLET RENDERING & METRIC COLLECTION FOR ALL 10 CASES
    # -------------------------------------------------------------------------
    for case in EXTREME_CASES:
        print(f"\n[Executing Episode {case['id']}/10] {case['name']}...")
        
        # Initialize environment
        stage_val = 3 if case["wind"] else 2
        env = CurriculumHoverAviaryV4(gui=True, stage=stage_val)

        # Force exact custom initial state
        env.INIT_XYZS = np.array([case["pos"]], dtype=np.float32)
        env.INIT_RPYS = np.array([case["rpy"]], dtype=np.float32)

        obs, info = env.reset()

        # Override PyBullet velocities for custom kinetic states
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
        time_history = []

        step_dt = 1.0 / 30.0
        recovered_time = None

        for step in range(240):  # 8.0 Second Episode
            action, _ = model.predict(obs, deterministic=True)
            action_flat = np.array(action, dtype=np.float32).flatten()  # Guaranteed (4,)

            obs, reward, terminated, truncated, info = env.step(action_flat)

            pos = obs[0:3]
            rpy = obs[3:6]

            pos_history.append(pos)
            rpy_history.append(rpy)
            act_history.append(action_flat)

            dist_cm = np.linalg.norm(pos - np.array([0.0, 0.0, 1.0])) * 100.0
            dist_history.append(dist_cm)
            time_history.append(step * step_dt)

            if dist_cm < 5.0 and recovered_time is None:
                recovered_time = step * step_dt

            time.sleep(step_dt * 0.5)  # Accelerated real-time render pacing
            if terminated or truncated:
                break

        env.close()

        pos_arr = np.array(pos_history)
        rpy_arr = np.array(rpy_history) * (180.0 / np.pi)  # Rad to deg
        act_arr = np.array(act_history)
        dist_arr = np.array(dist_history)
        time_arr = np.array(time_history)

        # Metrics computation
        hover_dists = dist_arr[30:] if len(dist_arr) > 30 else dist_arr
        steady_rmse_cm = np.sqrt(np.mean((hover_dists / 100.0) ** 2)) * 100.0
        in_5cm_pct = (np.sum(hover_dists < 5.0) / len(hover_dists)) * 100.0
        in_2cm_pct = (np.sum(hover_dists < 2.0) / len(hover_dists)) * 100.0
        max_overshoot_cm = np.max(dist_arr)
        survived = len(pos_history) >= 240

        rec_str = f"{recovered_time:.2f} s" if recovered_time is not None else "N/A"
        print(f"  * Status: {'SURVIVED' if survived else 'CRASHED'}")
        print(f"  * Recovery Time (<5cm) : {rec_str}")
        print(f"  * Steady Hover RMSE    : {steady_rmse_cm:.2f} cm")
        print(f"  * Sub-2cm Hold Time    : {in_2cm_pct:.1f} %")

        case_results.append({
            "case": case,
            "pos": pos_arr,
            "rpy": rpy_arr,
            "act": act_arr,
            "dist": dist_arr,
            "time": time_arr,
            "survived": survived,
            "rec_time": rec_str,
            "rmse": steady_rmse_cm,
            "in_2cm": in_2cm_pct,
            "max_dist": max_overshoot_cm
        })

    # -------------------------------------------------------------------------
    # PART 2: MULTI-PANEL DIAGNOSTIC DASHBOARD GENERATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("   GENERATING HIGH-RESOLUTION MULTI-EPISODE DIAGNOSTIC DASHBOARD")
    print("=" * 85)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("V4 Stage 2 — 10 Extreme Edge Case Evaluation Dashboard", fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Panel 1: 3D Trajectories (All 10 Cases)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for idx, res in enumerate(case_results):
        p_arr = res["pos"]
        ax1.plot(p_arr[:, 0], p_arr[:, 1], p_arr[:, 2], color=colors[idx], lw=1.5, alpha=0.85, label=f"Case {idx+1}")
        ax1.scatter([p_arr[0, 0]], [p_arr[0, 1]], [p_arr[0, 2]], color=colors[idx], s=40, marker='o')

    ax1.scatter([0], [0], [1.0], color='r', s=150, marker='*', label="Target (0,0,1)")

    # Render 5cm Target Sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    xs = 0.05 * np.outer(np.cos(u), np.sin(v))
    ys = 0.05 * np.outer(np.sin(u), np.sin(v))
    zs = 1.0 + 0.05 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(xs, ys, zs, color='r', alpha=0.2)

    ax1.set_title("3D Convergence Trajectories (Cases 1–10)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend(loc='upper right', fontsize=7, ncol=2)

    # Panel 2: 3D Tracking Error Convergence (Cases 1–10)
    ax2 = fig.add_subplot(2, 3, 2)
    for idx, res in enumerate(case_results):
        ax2.plot(res["time"], res["dist"], color=colors[idx], lw=1.5, alpha=0.85, label=f"Case {idx+1}")
    ax2.axhline(5.0, color='r', linestyle='--', alpha=0.7, label="5cm Target Sphere")
    ax2.axhline(2.0, color='g', linestyle=':', alpha=0.7, label="2cm Precision Goal")
    ax2.set_title("3D Tracking Error Convergence (cm)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (cm)")
    ax2.set_ylim([0, 260])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=7, ncol=2)

    # Panel 3: Altitude Z Profiles (Cases 1–10)
    ax3 = fig.add_subplot(2, 3, 3)
    for idx, res in enumerate(case_results):
        ax3.plot(res["time"], res["pos"][:, 2], color=colors[idx], lw=1.5, alpha=0.85)
    ax3.axhline(1.0, color='r', linestyle='--', alpha=0.8, label="Target Z (1.0m)")
    ax3.axhline(0.05, color='k', linestyle=':', alpha=0.5, label="Floor Threshold")
    ax3.set_title("Altitude (Z) Recovery Profiles (m)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Z Altitude (m)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right', fontsize=8)

    # Panel 4: Case 9 Attitude Damping (Ultimate Compound Adversary)
    ax4 = fig.add_subplot(2, 3, 4)
    c9_res = case_results[8]  # Case 9
    ax4.plot(c9_res["time"], c9_res["rpy"][:, 0], label="Roll φ", color='c')
    ax4.plot(c9_res["time"], c9_res["rpy"][:, 1], label="Pitch θ", color='m')
    ax4.plot(c9_res["time"], c9_res["rpy"][:, 2], label="Yaw ψ", color='y')
    ax4.axhline(0.0, color='k', linestyle='--', alpha=0.4)
    ax4.set_title("Case 9 Attitude Damping [Roll, Pitch, Yaw] (deg)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angle (deg)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    # Panel 5: Case 9 Motor Command Allocation
    ax5 = fig.add_subplot(2, 3, 5)
    c9_act = c9_res["act"]
    rotor_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i in range(4):
        ax5.plot(c9_res["time"], c9_act[:, i], label=f"Rotor u_{i+1}", color=rotor_colors[i], alpha=0.8)
    ax5.set_title("Case 9 Normalized Rotor Commands [-1, +1]")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Action Signal")
    ax5.set_ylim([-1.1, 1.1])
    ax5.grid(True, alpha=0.3)
    ax5.legend(ncol=2, fontsize=8)

    # Panel 6: Comprehensive Benchmark Performance Summary Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    table_data = [["Case", "Initial Dist", "Rec. Time", "Hover RMSE", "Status"]]
    for res in case_results:
        c_id = res["case"]["id"]
        init_d = f"{res['dist'][0]:.1f} cm"
        r_t = res["rec_time"]
        rmse_val = f"{res['rmse']:.2f} cm"
        status_str = "PASS" if res["survived"] else "FAIL"
        table_data.append([f"Case {c_id}", init_d, r_t, rmse_val, status_str])

    table = ax6.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.18, 0.25, 0.22, 0.22, 0.18]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)

    # Highlight table header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')

    ax6.set_title("10-Case Benchmark Metric Matrix", y=0.96, fontweight='bold')

    plt.tight_layout()
    output_path = "v4_stage2_10_extreme_cases_diagnostics.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n -> 10-Case Diagnostic Dashboard saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    run_10_extreme_cases()