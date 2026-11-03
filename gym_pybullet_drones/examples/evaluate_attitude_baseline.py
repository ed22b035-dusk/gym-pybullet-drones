import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.AttitudeHoverAviary import (
    AttitudeHoverAviary,
)
from gym_pybullet_drones.utils.enums import (
    ActionType,
    ObservationType,
    Physics,
)


def evaluate(model_path, episodes=5, gui=False):
    model = PPO.load(model_path)

    env = AttitudeHoverAviary(
        gui=gui,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        physics=Physics.PYB,
        max_roll_deg=0.0,
        max_pitch_deg=0.0,
        max_yaw_deg=0.0,
        max_ang_vel_rad_s=0.0,
        episode_len_sec=8.0,
    )

    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=1000 + episode)

            total_reward = 0.0
            positions = []
            attitudes = []
            rates = []
            actions = []
            step_count = 0

            while True:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)
                step_count += 1

                positions.append(
                    obs[0, 0:3].copy()
                )

                attitudes.append(
                    obs[0, 3:6].copy()
                )

                rates.append(
                    obs[0, 9:12].copy()
                )

                actions.append(
                    action[0].copy()
                )

                if terminated or truncated:
                    break

            positions = np.asarray(positions)
            attitudes = np.asarray(attitudes)
            rates = np.asarray(rates)
            actions = np.asarray(actions)

            target = np.array([0.0, 0.0, 1.0])
            position_errors = np.linalg.norm(
                positions - target,
                axis=1,
            )

            print("\n" + "=" * 70)
            print(f"Episode: {episode + 1}")
            print(f"Steps: {step_count}")
            print(f"Return: {total_reward:.4f}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(
                f"Position RMSE: "
                f"{np.sqrt(np.mean(position_errors**2)):.6f}"
            )
            print(
                f"Mean position error: "
                f"{np.mean(position_errors):.6f}"
            )
            print(
                f"Maximum absolute roll: "
                f"{np.max(np.abs(attitudes[:, 0])):.6f} rad"
            )
            print(
                f"Maximum absolute pitch: "
                f"{np.max(np.abs(attitudes[:, 1])):.6f} rad"
            )
            print(
                f"Maximum angular-rate norm: "
                f"{np.max(np.linalg.norm(rates, axis=1)):.6f}"
            )
            print(
                f"Mean action norm: "
                f"{np.mean(np.linalg.norm(actions, axis=1)):.6f}"
            )

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path to best_model.zip or final_model.zip",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--gui",
        action="store_true",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(args.model)

    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        gui=args.gui,
    )


if __name__ == "__main__":
    main()