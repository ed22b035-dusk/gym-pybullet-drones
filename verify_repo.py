import inspect
import sys

import pybullet
import gym_pybullet_drones

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary


print("Python executable:")
print(sys.executable)

print("\nPyBullet location:")
print(pybullet.__file__)

print("\ngym-pybullet-drones location:")
print(gym_pybullet_drones.__file__)

print("\nCtrlAviary:")
print(inspect.signature(CtrlAviary.__init__))

print("\nHoverAviary:")
print(inspect.signature(HoverAviary.__init__))

print("\nINSTALLATION OK")