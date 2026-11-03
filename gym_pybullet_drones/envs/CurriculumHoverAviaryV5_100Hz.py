import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4


class CurriculumHoverAviaryV5_100Hz(CurriculumHoverAviaryV4):
    """
    Curriculum Hover Aviary V5 (100 Hz Control Frequency)
    Inherits 100% of V4 state space, reward kernel, observations, and physics.
    Modifies control loop frequency from 30 Hz (Δt=33.3ms) to 100 Hz (Δt=10.0ms).
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

        # Initialize parent V4 environment
        super().__init__(gui=gui, record=record, stage=stage, **kwargs)

        # Explicitly set action space shape to (1, 4) to match BaseAviary & Stage 3 checkpoints
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)

        # Override control frequency parameters for 100 Hz
        self.CTRL_FREQ = 100
        self.PYBULLET_FREQ = 500
        self.PYBULLET_STEPS = int(self.PYBULLET_FREQ / self.CTRL_FREQ)  # 5 sub-steps per step
        if hasattr(self, "AGGREGATE_PHYSICS_STEPS"):
            self.AGGREGATE_PHYSICS_STEPS = self.PYBULLET_STEPS

        # Set PyBullet physical simulation timestep to 1/500 s
        p.setTimeStep(1.0 / self.PYBULLET_FREQ, physicsClientId=self.CLIENT)

        # Recalculate episode step limits: 100 Hz * 8.0 s = 800 steps
        self.EPISODE_LEN_SEC = 8.0
        self.MAX_STEPS = int(self.EPISODE_LEN_SEC * self.CTRL_FREQ)