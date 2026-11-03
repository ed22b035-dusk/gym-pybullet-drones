import time

import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel


env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    initial_xyzs=np.array(
        [[0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
    gui=True,
    record=False,
)

try:
    observation, info = env.reset()

    print("CF2X loaded successfully.")
    print("Observation shape:", observation.shape)
    print("Action space:", env.action_space)
    print("GUI remains open for 30 seconds.")

    for _ in range(900):
        action = np.zeros((1, 4), dtype=np.float32)
        env.step(action)
        time.sleep(1.0 / 30.0)

finally:
    env.close()