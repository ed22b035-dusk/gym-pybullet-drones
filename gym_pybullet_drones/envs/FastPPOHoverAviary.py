import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


class FastPPOHoverAviary(BaseRLAviary):
    """Fast PPO hover environment with survival bonus."""
    
    def __init__(self,
                 drone_model='cf2x',
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics='pyb',
                 pyb_freq=240,
                 ctrl_freq=30,
                 gui=False,
                 record=False,
                 obs=ObservationType.KIN,
                 act=ActionType.RPM,
                 ):
        self.TARGET_POS = np.array([0, 0, 1.0])
        self.previous_action = None
        
        # Reward weights
        self.W_POS_XY = 1.0
        self.W_POS_Z = 2.0
        self.W_VEL = 0.1
        self.W_TILT = 0.1
        self.W_ANG_VEL = 0.02
        
        # SURVIVAL BONUS
        self.SURVIVAL_BONUS = 1.0
        
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
        """Compute reward with survival bonus."""
        state = self._getDroneStateVector(0)
        
        pos = state[0:3]
        pos_err = pos - self.TARGET_POS
        xy_err_sq = np.sum(pos_err[0:2] ** 2)
        z_err_sq = pos_err[2] ** 2
        
        rpy = state[7:10]
        tilt_sq = rpy[0] ** 2 + rpy[1] ** 2
        
        vel = state[10:13]
        vel_sq = np.sum(vel ** 2)
        
        ang_vel = state[13:16]
        ang_vel_sq = np.sum(ang_vel ** 2)
        
        reward = (
            self.SURVIVAL_BONUS
            - self.W_POS_XY * xy_err_sq
            - self.W_POS_Z * z_err_sq
            - self.W_VEL * vel_sq
            - self.W_TILT * tilt_sq
            - self.W_ANG_VEL * ang_vel_sq
        )
        
        reward = float(np.clip(reward, -1.0, 2.0))
        
        return reward
    
    def _computeTerminated(self):
        return False
    
    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]
        
        position_error = np.linalg.norm(pos - self.TARGET_POS)
        altitude = pos[2]
        
        tilted_too_far = (
            abs(rpy[0]) > np.deg2rad(90) or
            abs(rpy[1]) > np.deg2rad(90)
        )
        
        time_limit_reached = self.step_counter >= int(8.0 * self.CTRL_FREQ)
        
        truncated = (
            altitude < 0.02 or
            position_error > 5.0 or
            np.linalg.norm(vel) > 10.0 or
            np.linalg.norm(ang_vel) > 50.0 or
            tilted_too_far or
            time_limit_reached
        )
        
        return truncated
    
    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        pos_error = np.linalg.norm(pos - self.TARGET_POS)
        return {
            'position_error': pos_error,
            'altitude': pos[2],
            'reward': self._computeReward(),
        }
    
    def step(self, action):
        self.previous_action = np.asarray(action, dtype=np.float32).copy()
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info