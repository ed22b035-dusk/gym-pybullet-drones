import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class CurriculumHoverAviaryV4(BaseRLAviary):
    """
    Curriculum Hover Aviary V4 (High-Precision Compute Architecture)
    
    Fixes:
    - Standardized `_computeReward(self)` signature to match BaseAviary step interface.
    
    Upgrades:
    1. 16D State Observation: [pos(3), rpy(3), vel(3), ang_vel(3), prev_action(4)]
    2. Extended Recovery Airspace: R_trunc = 3.5m allows outer spawns (2.0m) buffer to bank back.
    3. Differentiable Gaussian Precision Kernels: Infinite gradient field for sub-centimeter hover.
    4. Direct Action Rate Penalty (-1.5 * ||u_t - u_{t-1}||^2): Eliminates rotor chatter.
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 stage: int = 2):

        self.stage = stage
        self.total_env_steps = 0  
        
        self.TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.TARGET_YAW = 0.0
        self.EPISODE_LEN_SEC = 8.0  # 240 control steps at 30 Hz

        # Airspace Boundaries
        self.R_cyl = 2.0        # Max spawn radius (m)
        self.R_trunc = 3.5      # Physical truncation boundary (m) gives 1.5m recovery buffer
        self.Z_max = 2.0        # Max altitude ceiling (m)
        self.Z_min = 0.05       # Floor contact threshold (m)

        # Reward Normalization
        self.REWARD_SCALE = 30.0

        # Wind Calibration
        self.wind_std = 0.02 if stage == 2 else (0.03 if stage == 3 else 0.0)
        self.gust_prob = 0.20 if stage == 2 else (0.25 if stage == 3 else 0.0)

        self.current_action = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)

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
            act=act
        )

        # 16-Dimensional Observation Space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

    def _actionSpace(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        kinematic_state = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]]).astype(np.float32)
        return np.hstack([kinematic_state, self.prev_action]).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.current_action = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)

        if self.stage == 1:
            init_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            init_rpy = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            init_vel = np.zeros(3, dtype=np.float32)
            init_ang_vel = np.zeros(3, dtype=np.float32)

        elif self.stage == 2:
            # Progressive Curriculum Ramp over 1.5M Steps
            warmup = min(1.0, self.total_env_steps / 1500000.0)
            
            max_tilt_deg = 2.0 + (12.0 * warmup)
            max_r_spawn = 0.5 + (1.5 * warmup)  # 0.5m -> 2.0m radius

            r = np.sqrt(np.random.uniform(0, 1)) * max_r_spawn
            theta = np.random.uniform(0, 2 * np.pi)
            x_0 = r * np.cos(theta)
            y_0 = r * np.sin(theta)
            z_0 = np.random.uniform(0.4, 1.2)
            init_pos = np.array([[x_0, y_0, z_0]], dtype=np.float32)

            roll_0 = np.random.uniform(-max_tilt_deg, max_tilt_deg) * np.pi / 180.0
            pitch_0 = np.random.uniform(-max_tilt_deg, max_tilt_deg) * np.pi / 180.0
            yaw_0 = np.random.uniform(-np.pi, np.pi)
            init_rpy = np.array([[roll_0, pitch_0, yaw_0]], dtype=np.float32)

            init_vel = np.zeros(3, dtype=np.float32)
            init_ang_vel = np.zeros(3, dtype=np.float32)

        elif self.stage == 3:
            r = np.sqrt(np.random.uniform(0, 1)) * self.R_cyl
            theta = np.random.uniform(0, 2 * np.pi)
            x_0 = r * np.cos(theta)
            y_0 = r * np.sin(theta)
            z_0 = np.random.uniform(0.3, 1.3)
            init_pos = np.array([[x_0, y_0, z_0]], dtype=np.float32)

            roll_0 = np.random.uniform(-14.0, 14.0) * np.pi / 180.0
            pitch_0 = np.random.uniform(-14.0, 14.0) * np.pi / 180.0
            yaw_0 = np.random.uniform(-np.pi, np.pi)
            init_rpy = np.array([[roll_0, pitch_0, yaw_0]], dtype=np.float32)

            init_vel = np.random.uniform(-0.5, 0.5, size=3).astype(np.float32)
            init_ang_vel = np.random.uniform(-0.3, 0.3, size=3).astype(np.float32)

        self.INIT_XYZS = init_pos
        self.INIT_RPYS = init_rpy

        obs, info = super().reset(seed=seed, options=options)

        p.resetBaseVelocity(
            self.DRONE_IDS[0],
            linearVelocity=init_vel.tolist(),
            angularVelocity=init_ang_vel.tolist(),
            physicsClientId=self.CLIENT
        )

        return self._computeObs(), info

    def step(self, action):
        self.total_env_steps += 1
        self.current_action = np.array(action, dtype=np.float32).flatten()

        # Apply external wind forces
        if self.stage in [2, 3] and np.random.rand() < self.gust_prob:
            current_wind_std = self.wind_std
            if self.stage == 2:
                warmup = min(1.0, self.total_env_steps / 1500000.0)
                current_wind_std = self.wind_std * warmup

            wind_force = np.random.normal(0.0, current_wind_std, size=3).tolist()
            p.applyExternalForce(
                objectUniqueId=self.DRONE_IDS[0],
                linkIndex=-1,
                forceObj=wind_force,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT
            )

        act_env = self.current_action.reshape(1, -1)
        obs, reward, terminated, truncated, info = super().step(act_env)

        # Update action history after step execution
        self.prev_action = self.current_action.copy()

        return self._computeObs(), reward, terminated, truncated, info

def _computeReward(self):
    # Extract drone state vector
    state = self._getDroneStateVector(0)
    pos = state[0:3]         # Position vector (p)
    rpy = state[7:10]        # Roll, Pitch, Yaw (phi, theta, psi)
    vel = state[10:13]       # Linear velocity (v)
    ang_vel = state[13:16]   # Angular velocity (omega)

    phi, theta = rpy[0], rpy[1]

    # 1. Gaussian Position Kernel (K_pos, sigma_pos = 0.04 m)
    dist_3d_sq = float(np.sum((pos - self.TARGET_POS) ** 2))
    K_pos = 12.0 * np.exp(-dist_3d_sq / (2.0 * (0.04 ** 2)))

    # 2. Gaussian Attitude Kernel (K_att, sigma_att = 0.04 rad)
    tilt_sq = float(phi**2 + theta**2)
    K_att = 5.0 * np.exp(-tilt_sq / (2.0 * (0.04 ** 2)))

    # 3. Velocity Damping Terms (L2-norms)
    lin_vel_penalty = 2.0 * float(np.linalg.norm(vel))
    ang_vel_penalty = 0.5 * float(np.linalg.norm(ang_vel))

    # 4. Action Smoothness Penalty (Squared L2-norm)
    action_diff = self.current_action - self.prev_action
    action_smooth_penalty = 1.5 * float(np.sum(action_diff ** 2))

    # Unscaled Raw Reward Bracket
    R_raw = (
        K_pos 
        + K_att 
        - lin_vel_penalty 
        - ang_vel_penalty 
        - action_smooth_penalty
    )

    # Scaled Output (1 / 25.0 factor)
    return float(R_raw / 25.0)

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]

        # Extended Airspace Truncation Checks
        dist_xy = np.linalg.norm(pos[0:2] - self.TARGET_POS[0:2])
        if dist_xy > self.R_trunc or pos[2] > self.Z_max or pos[2] < self.Z_min:
            return True

        max_angle = 60.0 * np.pi / 180.0
        if abs(rpy[0]) > max_angle or abs(rpy[1]) > max_angle:
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        return {"stage": self.stage}