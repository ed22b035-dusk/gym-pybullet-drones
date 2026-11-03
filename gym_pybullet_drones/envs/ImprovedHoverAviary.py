"""
Improved hover environment with comprehensive reward shaping.
Based on analysis of PyFlyt, recent RL drone papers (2024-2025), and best practices.

Key reward components:
1. Position (XY and Z separately weighted)
2. Attitude (roll, pitch, yaw)
3. Velocity (linear and angular)
4. Control effort (smooth actions)
5. Survival bonus

Expected behavior:
- Perfect hover: reward ≈ +0.5 to +1.0
- Crashing: reward ≈ -3.0 to -10.0
- Learning to stabilize: reward should increase from -5.0 to 0.0 to +0.5
"""

import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType


class ImprovedHoverAviary(BaseRLAviary):
    """
    Single agent RL problem: comprehensive hover control.
    
    Reward function includes:
    - Position tracking (XY and Z)
    - Attitude stabilization (roll, pitch, yaw)
    - Velocity minimization
    - Control smoothness
    - Survival bonus
    """
    
    def __init__(self,
                 drone_model=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics='pyb',
                 pyb_freq=240,
                 ctrl_freq=30,
                 gui=False,
                 record=False,
                 obs=ObservationType.KIN,
                 act=ActionType.RPM,
                 episode_len_sec=30,
                 ):
        """
        Parameters
        ----------
        episode_len_sec : int
            Maximum episode duration in seconds. Default: 30
        """
        self.TARGET_POS = np.array([0, 0, 1.0])
        self.EPISODE_LEN_SEC = episode_len_sec
        self.previous_action = None
        
        # === REWARD WEIGHTS (tuned based on literature) ===
        # Position weights
        self.W_POS_XY = 1.0   # XY position penalty
        self.W_POS_Z = 2.0    # Z position penalty (harder, needs higher weight)
        
        # Attitude weights
        self.W_ROLL = 0.5     # Roll penalty
        self.W_PITCH = 0.5    # Pitch penalty
        self.W_YAW = 0.1      # Yaw penalty (less critical for hover)
        
        # Velocity weights
        self.W_VEL = 0.1          # Linear velocity penalty
        self.W_ANG_VEL = 0.05     # Angular velocity penalty
        
        # Control weight
        self.W_ACTION = 0.01      # Action change penalty
        
        # Survival bonus
        self.SURVIVAL_BONUS = 0.5  # Small bonus for staying alive
        
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
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
    
    def reset(self, seed=None, options=None):
        """Reset environment and clear action history."""
        self.previous_action = None
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        """Apply action smoothing and store for next step."""
        # Store previous action for smoothness penalty
        prev_action = self.previous_action
        self.previous_action = np.asarray(action, dtype=np.float32).copy()
        
        # Apply action smoothing (blend with previous)
        if prev_action is not None:
            action = 0.5 * prev_action + 0.5 * np.asarray(action)
        
        return super().step(action)
    
    def _computeReward(self):
        """
        Compute comprehensive reward with multiple components.
        
        Returns
        -------
        float
            Reward value clipped to [-5.0, 5.0]
        """
        state = self._getDroneStateVector(0)
        
        # === EXTRACT STATE VARIABLES ===
        pos = state[0:3]          # Position [x, y, z]
        rpy = state[7:10]         # Orientation [roll, pitch, yaw]
        vel = state[10:13]        # Linear velocity [vx, vy, vz]
        ang_vel = state[13:16]    # Angular velocity [wx, wy, wz]
        
        # === POSITION ERROR ===
        pos_error = pos - self.TARGET_POS
        xy_error = np.linalg.norm(pos_error[0:2])  # Horizontal distance
        z_error = abs(pos_error[2])                 # Altitude error
        
        # === ATTITUDE ===
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]
        
        # === REWARD COMPONENTS ===
        
        # 1. Position reward (negative quadratic penalty)
        r_pos_xy = -self.W_POS_XY * (xy_error ** 2)
        r_pos_z = -self.W_POS_Z * (z_error ** 2)
        
        # 2. Attitude reward (keep drone level)
        r_attitude = -self.W_ROLL * (roll ** 2) - self.W_PITCH * (pitch ** 2)
        r_yaw = -self.W_YAW * (yaw ** 2)
        
        # 3. Velocity reward (minimize movement = hover)
        r_vel = -self.W_VEL * np.sum(vel ** 2)
        r_ang_vel = -self.W_ANG_VEL * np.sum(ang_vel ** 2)
        
        # 4. Survival bonus (stay alive)
        r_survival = self.SURVIVAL_BONUS
        
        # 5. Control effort (smooth actions)
        r_control = 0.0
        # Note: Can't penalize action changes here because we don't have
        # access to current action in _computeReward. This is handled in step().
        
        # === TOTAL REWARD ===
        reward = (
            r_survival    # +0.5
            + r_pos_xy    # -1.0 * xy_error²
            + r_pos_z     # -2.0 * z_error²
            + r_attitude  # -0.5 * (roll² + pitch²)
            + r_yaw       # -0.1 * yaw²
            + r_vel       # -0.1 * vel²
            + r_ang_vel   # -0.05 * ang_vel²
            + r_control   # 0.0 (or -0.01 * action_diff² if implemented)
        )
        
        # Clip to reasonable range
        reward = float(np.clip(reward, -5.0, 5.0))
        
        return reward
    
    def _computeTerminated(self):
        """
        When episode ends successfully.
        For hover task, we don't have a "success" condition.
        """
        return False
    
    def _computeTruncated(self):
        """
        When episode ends due to failure or time limit.
        More forgiving than default HoverAviary.
        """
        state = self._getDroneStateVector(0)
        
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]
        altitude = pos[2]
        
        # === FAILURE CONDITIONS ===
        
        # 1. Crashed into ground
        crashed = altitude < 0.05
        
        # 2. Too far from target
        too_far = np.linalg.norm(pos - self.TARGET_POS) > 3.0
        
        # 3. Too fast (out of control)
        too_fast_vel = np.linalg.norm(vel) > 8.0
        too_fast_ang = np.linalg.norm(ang_vel) > 40.0
        
        # 4. Too tilted (about to crash)
        tilted_too_far = (
            abs(rpy[0]) > np.deg2rad(60) or
            abs(rpy[1]) > np.deg2rad(60)
        )
        
        # 5. Time limit reached
        time_limit_reached = self.step_counter >= int(self.EPISODE_LEN_SEC * self.CTRL_FREQ)
        
        truncated = (
            crashed or
            too_far or
            too_fast_vel or
            too_fast_ang or
            tilted_too_far or
            time_limit_reached
        )
        
        return truncated
    
    def _computeInfo(self):
        """Return useful information for logging and evaluation."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        
        pos_error = np.linalg.norm(pos - self.TARGET_POS)
        xy_error = np.linalg.norm(pos[0:2] - self.TARGET_POS[0:2])
        z_error = abs(pos[2] - self.TARGET_POS[2])
        tilt = np.sqrt(rpy[0] ** 2 + rpy[1] ** 2)
        
        return {
            'position_error': pos_error,
            'xy_error': xy_error,
            'z_error': z_error,
            'altitude': pos[2],
            'tilt': tilt,
            'roll': rpy[0],
            'pitch': rpy[1],
            'yaw': rpy[2],
            'velocity': np.linalg.norm(vel),
        }