import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class CurriculumHoverAviaryV3(BaseRLAviary):
    """
    Curriculum Hover Aviary V3 (Scaled Return Architecture)
    
    Fixes:
    1. Reward Scaling (1 / 25.0): Keeps step rewards in [-1, +1] range to cap V(s) <= 100.
    2. Dual Curriculum: Ramps tilt (2 -> 14 deg) and radius (0.8m -> 2.0m) over 500k steps.
    3. Adaptive Control Penalty: Reduces w_e penalty during recovery maneuvers.
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
                 stage: int = 1):

        self.stage = stage
        self.total_env_steps = 0  # Dynamic step counter for progressive curriculum
        
        self.TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.TARGET_YAW = 0.0
        self.EPISODE_LEN_SEC = 8.0  # 240 control steps at 30 Hz

        # Boundary Constants
        self.delta_R = 0.5     # Radial exploration boundary (m)
        self.delta_H = 0.5     # Height exploration boundary (m)
        self.Delta_p = 0.025   # Precision hover radius (2.5 cm)
        self.Delta_a = (2.0 * np.pi / 180.0) ** 2  # Precision attitude error (2 deg squared)
        self.R_cyl = 2.0       # Final spatial cylinder radius (m)

        # Reward Normalization Factor
        self.REWARD_SCALE = 25.0

        # Wind Calibration
        self.wind_std = 0.02 if stage == 2 else (0.03 if stage == 3 else 0.0)
        self.gust_prob = 0.20 if stage == 2 else (0.25 if stage == 3 else 0.0)

        self.prev_ang_vel = np.zeros(3, dtype=np.float32)
        self.initial_dist = 1.0

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

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def _actionSpace(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        return np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if self.stage == 1:
            init_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            init_rpy = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            init_vel = np.zeros(3, dtype=np.float32)
            init_ang_vel = np.zeros(3, dtype=np.float32)

        elif self.stage == 2:
            warmup = min(1.0, self.total_env_steps / 500000.0)
            
            max_tilt_deg = 2.0 + (12.0 * warmup)
            max_r_spawn = 0.8 + (1.2 * warmup)  # Expanding radius

            r = np.sqrt(np.random.uniform(0, 1)) * max_r_spawn
            theta = np.random.uniform(0, 2 * np.pi)
            x_0 = r * np.cos(theta)
            y_0 = r * np.sin(theta)
            z_0 = np.random.uniform(0.3, 1.2)
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
            z_0 = np.random.uniform(0.2, 1.3)
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

        self.prev_ang_vel = init_ang_vel.copy()
        self.initial_dist = float(np.linalg.norm(init_pos[0] - self.TARGET_POS))

        return self._computeObs(), info

    def step(self, action):
        self.total_env_steps += 1

        if self.stage in [2, 3] and np.random.rand() < self.gust_prob:
            current_wind_std = self.wind_std
            if self.stage == 2:
                warmup = min(1.0, self.total_env_steps / 500000.0)
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

        act_env = np.array(action, dtype=np.float32).reshape(1, -1)
        super().step(act_env)

        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        return obs, reward, terminated, truncated, info

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]

        phi, theta, psi = rpy[0], rpy[1], rpy[2]

        # 1. Target Tracking Error (Te)
        dist_to_target = float(np.linalg.norm(self.TARGET_POS - pos))
        yaw_error = abs(self.TARGET_YAW - psi)
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        T_e = dist_to_target + abs(yaw_error)

        # 2. Smooth Exploration Barrier (E_smooth)
        d_breach = dist_to_target - (self.initial_dist + self.delta_R)
        h_breach_top = pos[2] - (self.TARGET_POS[2] + self.delta_H)
        h_breach_bot = -pos[2]
        max_breach = float(max(d_breach, h_breach_top, h_breach_bot))
        E_smooth = float(1.2 / (1.0 + np.exp(-15.0 * max_breach)) - 0.2)

        # 3. Precision Hover Stability Bonus (S)
        angle_sq_sum = float(phi**2 + theta**2)
        if (dist_to_target < self.Delta_p) and (angle_sq_sum < self.Delta_a):
            S = 2.0
        else:
            S = -angle_sq_sum

        # 4. State-Adaptive Control Penalty (we)
        w_e_raw = float(np.sum((ang_vel - self.prev_ang_vel) ** 2))
        self.prev_ang_vel = ang_vel.copy()
        
        recovery_factor = min(1.0, (dist_to_target / 0.5) + (angle_sq_sum / 0.1))
        effective_we_coeff = 18.0 * (1.0 - 0.7 * recovery_factor)
        w_e = effective_we_coeff * w_e_raw

        # 5. Linear Velocity Damping Penalty
        lin_vel_penalty = float(np.linalg.norm(vel))

        R_raw = 25.0 - 20.0 * T_e - 100.0 * E_smooth + 20.0 * S - w_e - 2.0 * lin_vel_penalty
        
        # CRITICAL FIX: Scale reward to prevent value loss gradient drowning
        return float(R_raw / self.REWARD_SCALE)

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]

        dist_xy = np.linalg.norm(pos[0:2] - self.TARGET_POS[0:2])
        if dist_xy > (self.R_cyl + self.delta_R) or pos[2] > (self.TARGET_POS[2] + self.delta_H) or pos[2] < 0.0:
            return True

        max_angle = 60.0 * np.pi / 180.0
        if abs(rpy[0]) > max_angle or abs(rpy[1]) > max_angle:
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        return {"stage": self.stage}