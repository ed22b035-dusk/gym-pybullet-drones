import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4


class CurriculumHoverAviaryV5_60Hz(CurriculumHoverAviaryV4):
    """
    Curriculum Hover Aviary V5 (60 Hz Control Frequency)
    Inherits 100% of V4 state space, reward kernel, observations, and physics.
    Modifies control loop frequency from 30 Hz (Δt=33.3ms) to 60 Hz (Δt=16.67ms).
    """

    def __init__(
        self,
        drone_model=None,
        initial_xyzs=None,
        initial_rpys=None,
        physics=None,
        gui=False,
        record=False,
        obs=None,
        act=None,
        stage=3,
    ):
        kwargs = {}
        if drone_model is not None: kwargs["drone_model"] = drone_model
        if initial_xyzs is not None: kwargs["initial_xyzs"] = initial_xyzs
        if initial_rpys is not None: kwargs["initial_rpys"] = initial_rpys
        if physics is not None: kwargs["physics"] = physics
        if obs is not None: kwargs["obs"] = obs
        if act is not None: kwargs["act"] = act

        super().__init__(gui=gui, record=record, stage=stage, **kwargs)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)

        self.CTRL_FREQ = 60
        self.PYBULLET_FREQ = 240
        self.PYBULLET_STEPS = int(self.PYBULLET_FREQ / self.CTRL_FREQ)
        if hasattr(self, "AGGREGATE_PHYSICS_STEPS"):
            self.AGGREGATE_PHYSICS_STEPS = self.PYBULLET_STEPS

        self.EPISODE_LEN_SEC = 8.0
        self.MAX_STEPS = int(self.EPISODE_LEN_SEC * self.CTRL_FREQ)  # 480 steps

    def _isTruncated(self):
        """Overrides V4 240-step cap to allow full 480-step (8.0s @ 60Hz) rollouts."""
        return self.step_counter >= self.MAX_STEPS