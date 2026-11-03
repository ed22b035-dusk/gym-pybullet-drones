import os
import sys
import glob
import time
import numpy as np
import pybullet as p
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym-pybullet-drones"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym_pybullet_drones"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gym-pybullet-drones", "envs"))

from gym_pybullet_drones.envs.CurriculumHoverAviaryV5_60Hz import CurriculumHoverAviaryV5_60Hz

# -----------------------------------------------------------------------------
# VISUALIZATION PACING SETTINGS
# -----------------------------------------------------------------------------
PLAYBACK_SPEED = 1.0    # 1.0 = Exact Real Time, 0.5 = Half-Speed Slow Motion
PAUSE_BETWEEN_CASES = False # Set True to press Enter before each case starts
# -----------------------------------------------------------------------------

def find_v5_60hz_model():
    candidate_paths = [
        os.path.join(SCRIPT_DIR, "curriculum_checkpoints_v5_60hz", "v5_60hz_4m_model.zip"),
        os.path.join(SCRIPT_DIR, "curriculum_checkpoints_v5_60hz", "ckpts", "*.zip"),
        os.path.join(SCRIPT_DIR, "gym-pybullet-drones", "curriculum_checkpoints_v5_60hz", "ckpts", "*.zip"),
        "**/*v5_60hz*.zip",
        "**/*60hz*.zip"
    ]
    for pattern in candidate_paths:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return os.path.abspath(matches[0])
    return None

MODEL_PATH = find_v5_60hz_model()

class ActionShapeAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action_flat = np.array(action, dtype=np.float32).flatten()
        return self.env.step(action_flat)

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

def evaluate_and_visualize_v5_60hz():
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Checkpoint not found at: {MODEL_PATH}")
        sys.exit(1)

    print("=" * 105)
    print("   V5 60Hz REAL-TIME VISUALIZATION & EVALUATION SUITE")
    print("=" * 105)
    print(f" Checkpoint Path : {MODEL_PATH}")
    print(f" Playback Speed  : {PLAYBACK_SPEED}x Real-Time")
    print(f" Pause On Start  : {PAUSE_BETWEEN_CASES}")
    print("=" * 105)

    raw_env = CurriculumHoverAviaryV5_60Hz(gui=True, stage=3)
    raw_env.MAX_STEPS = 480
    raw_env.EPISODE_LEN_SEC = 8.0
    env = ActionShapeAdapter(raw_env)

    p.resetDebugVisualizerCamera(
        cameraDistance=2.5, cameraYaw=45, cameraPitch=-25,
        cameraTargetPosition=[0.0, 0.0, 1.0], physicsClientId=raw_env.CLIENT
    )

    model = PPO.load(
        MODEL_PATH, env=env,
        custom_objects={"observation_space": env.observation_space, "action_space": env.action_space}
    )

    case_results = []
    MAX_STEPS = 480
    target_dt = (1.0 / 60.0) / PLAYBACK_SPEED

    for case in ALL_EXTREME_CASES:
        c_id, c_name, c_cat = case["id"], case["name"], case["cat"]
        
        print(f"\n---------------------------------------------------------------------------------------------------------")
        print(f"[CASE {c_id:02d}/16] Category: [{c_cat}] - {c_name}")
        if PAUSE_BETWEEN_CASES:
            input(" Press [ENTER] in terminal to launch this test case...")

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
        time_to_5cm = None

        for step in range(MAX_STEPS):
            step_start_time = time.time()

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

            current_time_sec = (step + 1) * (1.0 / 60.0)
            if dist_cm < 5.0 and time_to_5cm is None:
                time_to_5cm = current_time_sec

            # Precise frame rate pacing for smooth 1.0x viewing
            elapsed = time.time() - step_start_time
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

            if terminated or truncated:
                break

        pos_arr = np.array(pos_history)
        dist_arr = np.array(dist_history)

        hover_dists = dist_arr[60:] if len(dist_arr) > 60 else dist_arr
        steady_rmse_cm = np.sqrt(np.mean((hover_dists / 100.0) ** 2)) * 100.0
        in_2cm_pct = (np.sum(hover_dists < 2.0) / len(hover_dists)) * 100.0
        
        min_altitude = np.min(pos_arr[:, 2])
        final_altitude = np.mean(pos_arr[-30:, 2]) if len(pos_arr) >= 30 else pos_arr[-1, 2]
        
        crashed_ground = min_altitude < 0.275 or final_altitude < 0.35
        never_reached_target = time_to_5cm is None
        unstable_hover = steady_rmse_cm > 25.0

        survived = (not crashed_ground) and (not never_reached_target) and (not unstable_hover)
        status_str = "PASS" if survived else "FAIL"

        t_5cm_str = f"{time_to_5cm:.2f} s" if time_to_5cm is not None else "N/A"
        print(f" Result: [{status_str}] | Time to <5cm: {t_5cm_str} | Steady RMSE: {steady_rmse_cm:.2f} cm")

        case_results.append({
            "id": c_id, "cat": c_cat, "name": c_name, "survived": survived,
            "init_dist": dist_arr[0], "t_5cm": t_5cm_str, "rmse_cm": steady_rmse_cm,
            "in_2cm": in_2cm_pct, "pos": pos_arr, "dist": dist_arr
        })

    env.close()

    total_cases = len(case_results)
    passed_cases = sum(1 for r in case_results if r["survived"])

    print("\n" + "=" * 105)
    print("                     FINAL GROUND-TRUTH MASTER REPORT (V5 60Hz)")
    print("=" * 105)
    print(f" Total Evaluated Cases : {total_cases}")
    print(f" Successful Recoveries : {passed_cases} / {total_cases}")
    print(f" Survival Rate         : {(passed_cases / total_cases) * 100.0:.1f} %")
    print("=" * 105)
    print(f"{'ID':<3} | {'Category':<16} | {'Init Dist':<10} | {'Rec (<5cm)':<10} | {'Hover RMSE':<11} | {'Status':<6}")
    print("-" * 105)
    for r in case_results:
        status_code = "PASS" if r["survived"] else "FAIL"
        print(f"{r['id']:02d}  | {r['cat']:<16} | {r['init_dist']:6.1f} cm | {r['t_5cm']:<10} | {r['rmse_cm']:6.2f} cm   | {status_code:<6}")
    print("-" * 105 + "\n")

if __name__ == "__main__":
    evaluate_and_visualize_v5_60hz()