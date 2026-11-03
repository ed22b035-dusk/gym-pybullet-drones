import time
import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviary import CurriculumHoverAviary

# Load trained Stage 3 model
model_path = "curriculum_checkpoints/stage3_ppo_model.zip"
print(f"Loading trained model from {model_path}...")
model = PPO.load(model_path)

# Test environment with GUI enabled in Stage 3 mode
env = CurriculumHoverAviary(gui=True, stage=3)
obs, info = env.reset()

print("\nRunning Evaluation Trajectory...")
for step in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Extract drone position and yaw error
    pos = obs[0:3]
    yaw = obs[5]
    pos_error = np.linalg.norm(np.array([0.0, 0.0, 1.0]) - pos)
    
    if step % 30 == 0:
        print(f"Step {step:03d} | Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | Pos Error: {pos_error*100:.2f} cm | Yaw: {np.degrees(yaw):.2f}°")

    env.render()
    time.sleep(1.0 / 30.0)

    if terminated or truncated:
        print("Episode finished. Resetting environment...")
        obs, info = env.reset()

env.close()