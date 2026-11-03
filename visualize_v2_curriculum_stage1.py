import os
import time
import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviaryV2 import CurriculumHoverAviaryV2

# ==============================================================================
# CONFIGURATION FOR STAGE 1 V2 VISUALIZATION
# ==============================================================================
MODEL_PATH = os.path.join("curriculum_checkpoints_v2", "v2_stage1_4m_ppo_model.zip")
STAGE = 1
NUM_EPISODES = 3
DURATION_SEC = 30  # Extended flight duration per episode in seconds
# ==============================================================================


def visualize_stage1():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Stage 1 V2 model checkpoint not found at: {MODEL_PATH}")
        return

    max_steps = int(DURATION_SEC * 30)  # 30 Hz control loop horizon

    print("=" * 65)
    print(f"   LAUNCHING V2 STAGE 1 VISUALIZATION ({DURATION_SEC} SECONDS)")
    print(f"   Model Path: {MODEL_PATH}")
    print(f"   Target Horizon: {max_steps} timesteps")
    print("=" * 65)

    # Load trained Stage 1 policy
    model = PPO.load(MODEL_PATH)

    # Initialize Stage 1 paper environment in 3D PyBullet GUI mode
    env = CurriculumHoverAviaryV2(gui=True, stage=STAGE)
    env.EPISODE_LEN_SEC = float(DURATION_SEC)  # Override default 8s truncation limit

    for ep in range(NUM_EPISODES):
        print(f"\n---------------- Episode {ep + 1}/{NUM_EPISODES} ----------------")
        obs, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0

        while not done and step_count < max_steps:
            # Flatten observation vector to 12D state
            flat_obs = np.array(obs, dtype=np.float32).flatten()[:12]
            action, _ = model.predict(flat_obs, deterministic=True)

            # Step environment dynamics
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            step_count += 1
            total_reward += reward

            # Pace playback speed to real-time 30 FPS
            time.sleep(1.0 / 30.0)

            # Telemetry progress print every 5 seconds (150 steps)
            if step_count % 150 == 0:
                print(f"  -> Hovering... Progress: {step_count}/{max_steps} timesteps")

        pos = obs[0:3]
        rpy_deg = np.degrees(obs[3:6])
        vel = obs[6:9]

        print(f"[SUMMARY] Episode {ep + 1} Complete:")
        print(f"  -> Timesteps Survived    : {step_count}/{max_steps}")
        print(f"  -> Total Episode Return  : {total_reward:.1f}")
        print(f"  -> Final Position [x,y,z]: {pos.round(3)} m")
        print(f"  -> Final Attitude RPY    : {rpy_deg.round(1)} deg")
        print(f"  -> Final Velocity        : {vel.round(3)} m/s")

        time.sleep(1.0)

    env.close()
    print("\n[VISUALIZE] Stage 1 playback complete.")


if __name__ == "__main__":
    visualize_stage1()