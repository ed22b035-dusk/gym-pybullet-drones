import os
import time
import numpy as np
from stable_baselines3 import PPO

# --- PyBullet Drones Imports ---
try:
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ObservationType, ActionType
except ImportError:
    from gym_pybullet_drones.envs.single_agent_aviary.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ObservationType, ActionType

MODEL_PATH = os.path.join("curriculum_checkpoints", "stage1_4m_ppo_model.zip")

# Set desired visualization length in seconds
EPISODE_LEN_SEC = 30  # 30 seconds = ~1,450 timesteps

def format_obs(raw_obs, target_dim):
    """Flattens observation array to match model input dimension."""
    flat = np.array(raw_obs, dtype=np.float32).flatten()
    if len(flat) >= target_dim:
        return flat[:target_dim]
    return np.pad(flat, (0, target_dim - len(flat)))

def visualize_hover(num_episodes=3, duration_sec=30):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

    print(f"[VISUALIZE] Loading Stage 1 4.0M Model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    target_obs_dim = model.observation_space.shape[0]

    print(f"[VISUALIZE] Launching PyBullet GUI (Target Duration: {duration_sec}s per episode)...")
    
    # Try passing episode_len_sec directly if supported by your version
    try:
        env = HoverAviary(
            gui=True,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            episode_len_sec=duration_sec
        )
    except TypeError:
        env = HoverAviary(
            gui=True,
            obs=ObservationType.KIN,
            act=ActionType.RPM
        )

    # Convert seconds to max steps (HoverAviary runs control loop at ~48Hz-240Hz depending on PyBullet setup)
    max_steps_limit = int(duration_sec * 48)

    for ep in range(num_episodes):
        print(f"\n================ Episode {ep + 1}/{num_episodes} ================")
        obs_tuple = env.reset()
        raw_obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
        
        step_count = 0
        done = False

        while step_count < max_steps_limit and not done:
            model_obs = format_obs(raw_obs, target_obs_dim)
            action, _ = model.predict(model_obs, deterministic=True)
            action_env = action.reshape(1, -1) if action.ndim == 1 else action

            step_res = env.step(action_env)
            if len(step_res) == 5:
                raw_obs, reward, terminated, truncated, info = step_res
                # Ignore environmental time truncation so the drone keeps hovering
                done = terminated
            else:
                raw_obs, reward, done, info = step_res

            step_count += 1

            # Real-time pacing (~30 FPS playback speed)
            time.sleep(1.0 / 30.0)

            if step_count % 100 == 0:
                print(f" -> Hovering... Timesteps reached: {step_count}/{max_steps_limit}")

        print(f"[COMPLETED] Episode {ep + 1} finished! Total Duration: {step_count} timesteps.")
        time.sleep(1.0)

    env.close()
    print("\n[VISUALIZE] Playback finished.")

if __name__ == "__main__":
    visualize_hover(num_episodes=3, duration_sec=EPISODE_LEN_SEC)