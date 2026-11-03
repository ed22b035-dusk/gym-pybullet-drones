import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

class StableHoverAviary(BaseRLAviary):
    """Stable hover environment for PPO (300k-500k steps)."""

    def __init__(self,
                 drone_model='cf2x',
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics='pyb',
                 pyb_freq=240,
                 ctrl_freq=30,  # Keep your existing 30 Hz
                 gui=False,
                 record=False,
                 obs=ObservationType.KIN,
                 act=ActionType.RPM,  # 4 independent motors
                 ):
        self.TARGET_POS = np.array([0, 0, 1.0])

        # Previous action for smoothness (added after basic hover works)
        self.previous_action = None

        # Reward weights (tunable)
        self.W_POS_XY = 2.0
        self.W_POS_Z = 4.0
        self.W_VEL = 0.3
        self.W_TILT = 0.2
        self.W_ANG_VEL = 0.05
        self.W_SMOOTH = 0.02  # Add after basic hover

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

    def _computeReward(self):
        """Compute reward with velocity, tilt, and angular-rate penalties."""
        state = self._getDroneStateVector(0)

        # Position error
        pos = state[0:3]
        pos_err = pos - self.TARGET_POS
        xy_err_sq = np.sum(pos_err[0:2] ** 2)
        z_err_sq = pos_err[2] ** 2

        # Roll/pitch (tilt)
        rpy = state[7:10]
        tilt_sq = rpy[0] ** 2 + rpy[1] ** 2

        # Linear velocity
        vel = state[10:13]
        vel_sq = np.sum(vel ** 2)

        # Angular velocity
        ang_vel = state[13:16]
        ang_vel_sq = np.sum(ang_vel ** 2)

        # Base reward
        reward = (
            1.00
            - self.W_POS_XY * xy_err_sq
            - self.W_POS_Z * z_err_sq
            - self.W_VEL * vel_sq
            - self.W_TILT * tilt_sq
            - self.W_ANG_VEL * ang_vel_sq
        )

        # Action smoothness (add after basic hover works)
        if self.previous_action is not None and hasattr(self, 'current_policy_action'):
            action_delta_sq = np.sum(
                (self.current_policy_action - self.previous_action) ** 2
            )
            reward -= self.W_SMOOTH * action_delta_sq

        # Update previous action
        if hasattr(self, 'current_policy_action'):
            self.previous_action = self.current_policy_action.copy()

        # Clip to prevent extreme values
        reward = float(np.clip(reward, -2.0, 1.0))

        return reward

    def _computeTerminated(self):
        """Terminate if very close to target (success)."""
        state = self._getDroneStateVector(0)
        pos_error = np.linalg.norm(state[0:3] - self.TARGET_POS)

        # Small tolerance for numerical precision
        return pos_error < 0.005  # Within 5mm

    def _computeTruncated(self):
        """Truncate if crash or out of bounds."""
        state = self._getDroneStateVector(0)

        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]

        position_error = np.linalg.norm(pos - self.TARGET_POS)
        altitude = pos[2]

        # Tilt limits (45 degrees)
        tilted_too_far = (
            abs(rpy[0]) > np.deg2rad(45) or
            abs(rpy[1]) > np.deg2rad(45)
        )

        # Time limit (4 seconds)
        time_limit_reached = self.step_counter / self.CTRL_FREQ >= 4.0

        truncated = (
            altitude < 0.05 or  # Crash
            position_error > 2.0 or  # Too far
            np.linalg.norm(vel) > 3.0 or  # Too fast
            np.linalg.norm(ang_vel) > 12.0 or  # Spinning
            tilted_too_far or
            time_limit_reached
        )

        return truncated

    def _computeInfo(self):
        """Additional information."""
        state = self._getDroneStateVector(0)
        pos_error = np.linalg.norm(state[0:3] - self.TARGET_POS)
        altitude = state[2]
        return {
            'position_error': pos_error,
            'altitude': altitude,
            'target_reached': pos_error < 0.05,
        }

    def step(self, action):
        """Step with action tracking for smoothness penalty."""
        # Store current action for smoothness penalty
        self.current_policy_action = np.asarray(action, dtype=np.float32).copy()

        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info