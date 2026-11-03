import numpy as np

from gym_pybullet_drones.envs.AttitudeHoverAviary import (
    AttitudeHoverAviary,
)


class AttitudeHoverStableAviary(AttitudeHoverAviary):
    """Attitude-hover environment with stability-aware reward."""

    def _computeReward(self):
        state = self._getDroneStateVector(0)

        position_error = np.linalg.norm(
            self.TARGET_POS - state[0:3]
        )

        roll = state[7]
        pitch = state[8]

        linear_velocity = state[10:13]
        angular_velocity = state[13:16]

        position_reward = max(
            0.0,
            2.0 - position_error**4,
        )

        attitude_reward = np.exp(
            -2.0 * (roll**2 + pitch**2)
        )

        velocity_reward = np.exp(
            -0.1 * np.sum(linear_velocity**2)
        )

        rate_reward = np.exp(
            -0.2 * np.sum(angular_velocity**2)
        )

        reward = (
            1.0 * position_reward
            + 0.5 * attitude_reward
            + 0.25 * velocity_reward
            + 0.50 * rate_reward
        )

        return float(reward)