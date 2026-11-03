import os
import time
import numpy as np
from stable_baselines3 import PPO

try:
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ObservationType, ActionType
except ImportError:
    from gym_pybullet_drones.envs.single_agent_aviary.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ObservationType, ActionType

MODEL_PATH = os.path.join("curriculum_checkpoints", "stage2_3m_ppo_model.zip")

def verify():
    model = PPO.load(MODEL_PATH)
    env = HoverAviary(gui=True, obs=ObservationType.KIN, act=ActionType.RPM)
    
    for ep in range(3):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 242:
            flat_obs = np.array(obs, dtype=np.float32).flatten()[:12]
            action, _ = model.predict(flat_obs, deterministic=True)
            step_res = env.step(action.reshape(1, -1))
            obs = step_res[0]
            done = step_res[2] or step_res[3]
            steps += 1
            time.sleep(1/30)
        print(f"Episode {ep+1} survived {steps}/242 steps.")
    env.close()

if __name__ == "__main__":
    verify()