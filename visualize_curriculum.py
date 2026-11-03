import os
import time
import numpy as np
from stable_baselines3 import PPO

try:
    from CurriculumHoverAviary import CurriculumHoverAviary
except ImportError:
    from gym_pybullet_drones.envs.CurriculumHoverAviary import CurriculumHoverAviary

# ==============================================================================
# CONFIGURATION FOR EXTENDED VISUALIZATION
# ==============================================================================
MODEL_PATH = os.path.join("curriculum_checkpoints", "stage1_4m_ppo_model.zip")
STAGE = 1
NUM_EPISODES = 3
DURATION_SEC = 30  # <-- SET DESIRED FLIGHT TIME IN SECONDS (e.g., 30, 60, 120)
# ==============================================================================


def visualize():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Stage 1 baseline model not found at: {MODEL_PATH}")
        return

    max_steps = int(DURATION_SEC * 30)  # 30 Hz control loop

    print("=" * 65)
    print(f"   LAUNCHING STAGE 1 BASELINE VISUALIZATION ({DURATION_SEC} SECONDS)")
    print(f"   Model Path: {MODEL_PATH}")
    print(f"   Target Horizon: {max_steps} timesteps")
    print("=" * 65)

    model = PPO.load(MODEL_PATH)

    # Initialize environment and override episode length
    env = CurriculumHoverAviary(gui=True, stage=STAGE)
    env.EPISODE_LEN_SEC = DURATION_SEC  # <-- Prevents environment 8s cutoff

    for ep in range(NUM_EPISODES):
        print(f"\n---------------- Episode {ep + 1}/{NUM_EPISODES} ----------------")
        obs, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0

        while not done and step_count < max_steps:
            flat_obs = np.array(obs, dtype=np.float32).flatten()[:12]
            action, _ = model.predict(flat_obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            step_count += 1
            total_reward += reward

            # Pace playback at 30 FPS
            time.sleep(1.0 / 30.0)

            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"  -> Hovering... Timesteps reached: {step_count}/{max_steps}")

        pos = obs[0:3]
        rpy_deg = np.degrees(obs[3:6])
        vel = obs[6:9]

        print(f"[SUMMARY] Episode {ep + 1} Complete:")
        print(f"  -> Steps Survived        : {step_count}/{max_steps}")
        print(f"  -> Cumulative Return     : {total_reward:.1f}")
        print(f"  -> Final Position [x,y,z]: {pos.round(3)} m")
        print(f"  -> Final Attitude RPY    : {rpy_deg.round(1)} deg")
        print(f"  -> Final Velocity        : {vel.round(3)} m/s")

        time.sleep(1.0)

    env.close()
    print("\n[VISUALIZE] Playback session complete.")


if __name__ == "__main__":
    visualize()