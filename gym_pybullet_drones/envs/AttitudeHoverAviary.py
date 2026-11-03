import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)


class AttitudeHoverAviary(BaseRLAviary):
    """Single-drone RL hover task with attitude/rate reset randomization.

    The action is four independent normalized motor commands when
    ``act=ActionType.RPM``. The initial reward is position-only to provide
    a controlled baseline comparable with HoverAviary. Roll, pitch, yaw,
    angular velocity, and action history are present in the KIN observation
    through BaseRLAviary.
    """

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
        max_roll_deg: float = 0.0,
        max_pitch_deg: float = 0.0,
        max_yaw_deg: float = 0.0,
        max_ang_vel_rad_s: float = 0.0,
        episode_len_sec: float = 8.0,
        target_pos=None,
    ):
        if act != ActionType.RPM:
            raise ValueError(
                "AttitudeHoverAviary requires act=ActionType.RPM "
                "for four independent motor actions."
            )

        if obs != ObservationType.KIN:
            raise ValueError(
                "This initial implementation requires "
                "obs=ObservationType.KIN."
            )

        self.TARGET_POS = np.asarray(
            [0.0, 0.0, 1.0] if target_pos is None else target_pos,
            dtype=np.float32,
        )
        if self.TARGET_POS.shape != (3,):
            raise ValueError("target_pos must have shape (3,).")

        self.EPISODE_LEN_SEC = float(episode_len_sec)
        if self.EPISODE_LEN_SEC <= 0.0:
            raise ValueError("episode_len_sec must be positive.")

        self.MAX_ROLL_RAD = np.deg2rad(float(max_roll_deg))
        self.MAX_PITCH_RAD = np.deg2rad(float(max_pitch_deg))
        self.MAX_YAW_RAD = np.deg2rad(float(max_yaw_deg))
        self.MAX_ANG_VEL = float(max_ang_vel_rad_s)

        if (
            self.MAX_ROLL_RAD < 0.0
            or self.MAX_PITCH_RAD < 0.0
            or self.MAX_YAW_RAD < 0.0
            or self.MAX_ANG_VEL < 0.0
        ):
            raise ValueError("Curriculum limits cannot be negative.")

        if initial_xyzs is None:
            initial_xyzs = np.asarray(
                [self.TARGET_POS],
                dtype=np.float32,
            )
        else:
            initial_xyzs = np.asarray(
                initial_xyzs,
                dtype=np.float32,
            )

        if initial_xyzs.shape != (1, 3):
            raise ValueError("initial_xyzs must have shape (1, 3).")

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=np.zeros((1, 3), dtype=np.float32),
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        if physics == Physics.DYN:
            raise ValueError(
                "Use Physics.PYB for this initial implementation. "
                "Physics.DYN requires explicit rpy-rate reset handling."
            )

    def _sample_initial_conditions(self):
        """Sample initial Euler angles and body angular velocity."""

        rpy = np.array(
            [
                self.np_random.uniform(
                    -self.MAX_ROLL_RAD,
                    self.MAX_ROLL_RAD,
                ),
                self.np_random.uniform(
                    -self.MAX_PITCH_RAD,
                    self.MAX_PITCH_RAD,
                ),
                self.np_random.uniform(
                    -self.MAX_YAW_RAD,
                    self.MAX_YAW_RAD,
                ),
            ],
            dtype=np.float64,
        )

        angular_velocity = self.np_random.uniform(
            -self.MAX_ANG_VEL,
            self.MAX_ANG_VEL,
            size=3,
        ).astype(np.float64)

        return rpy, angular_velocity

    def reset(self, *, seed=None, options=None):
        """Reset the simulator and apply sampled attitude/rate conditions."""

        obs, info = super().reset(seed=seed, options=options)

        rpy, angular_velocity = self._sample_initial_conditions()
        quaternion = p.getQuaternionFromEuler(rpy.tolist())

        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            self.INIT_XYZS[0].tolist(),
            quaternion,
            physicsClientId=self.CLIENT,
        )

        p.resetBaseVelocity(
            self.DRONE_IDS[0],
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=angular_velocity.tolist(),
            physicsClientId=self.CLIENT,
        )

        self._updateAndStoreKinematicInformation()
        obs = self._computeObs()
        info = self._computeInfo()

        info.update({
            "initial_roll_rad": float(rpy[0]),
            "initial_pitch_rad": float(rpy[1]),
            "initial_yaw_rad": float(rpy[2]),
            "initial_roll_deg": float(np.rad2deg(rpy[0])),
            "initial_pitch_deg": float(np.rad2deg(rpy[1])),
            "initial_yaw_deg": float(np.rad2deg(rpy[2])),
            "initial_roll_rate_rad_s": float(angular_velocity[0]),
            "initial_pitch_rate_rad_s": float(angular_velocity[1]),
            "initial_yaw_rate_rad_s": float(angular_velocity[2]),
        })

        return obs, info

    def _computeReward(self):
        """Position-only baseline reward."""

        state = self._getDroneStateVector(0)
        position_error = np.linalg.norm(
            self.TARGET_POS - state[0:3]
        )

        return float(max(0.0, 2.0 - position_error**4))

    def _computeTerminated(self):
        """No success termination for the initial baseline."""

        return False

    def _computeTruncated(self):
        """Stop for bounds or time, not merely for attitude angle."""

        state = self._getDroneStateVector(0)
        x, y, z = state[0:3]

        if abs(x) > 3.0 or abs(y) > 3.0:
            return True
        if z < 0.05 or z > 3.0:
            return True
        if self.step_counter / self.PYB_FREQ >= self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        """Return diagnostics without changing the reward."""

        state = self._getDroneStateVector(0)
        position_error = np.linalg.norm(
            self.TARGET_POS - state[0:3]
        )

        return {
            "position_error": float(position_error),
            "roll_rad": float(state[7]),
            "pitch_rad": float(state[8]),
            "yaw_rad": float(state[9]),
            "roll_deg": float(np.rad2deg(state[7])),
            "pitch_deg": float(np.rad2deg(state[8])),
            "yaw_deg": float(np.rad2deg(state[9])),
            "roll_rate_rad_s": float(state[13]),
            "pitch_rate_rad_s": float(state[14]),
            "yaw_rate_rad_s": float(state[15]),
        }