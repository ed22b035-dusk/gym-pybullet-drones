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


def build_environment(
    phi_range_deg,
    omega_range_rad_s,
    yaw_range_deg=180.0,
    gui=False,
):
    """Create the actual PyBullet quadrotor evaluation environment."""

    return AttitudeHoverAviary(
        gui=gui,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        physics=Physics.PYB,
        max_roll_deg=phi_range_deg,
        max_pitch_deg=phi_range_deg,
        max_yaw_deg=yaw_range_deg,
        max_ang_vel_rad_s=omega_range_rad_s,
        episode_len_sec=8.0,
    )


def evaluate_model(
    model_path,
    phi_range_deg,
    omega_range_rad_s=0.0,
    yaw_range_deg=180.0,
    n_episodes=100,
    max_steps=None,
    gui=False,
    verbose=True,
):
    """Evaluate one saved PPO model on randomized quadrotor conditions."""

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    env = build_environment(
        phi_range_deg=phi_range_deg,
        omega_range_rad_s=omega_range_rad_s,
        yaw_range_deg=yaw_range_deg,
        gui=gui,
    )

    model = PPO.load(model_path)

    if max_steps is None:
        max_steps = int(
            env.EPISODE_LEN_SEC * env.CTRL_FREQ
        ) + 1

    rewards = []
    lengths = []
    successes = []

    position_rmses = []
    mean_position_errors = []
    max_rolls = []
    max_pitches = []
    max_rate_norms = []

    try:
        for episode in range(n_episodes):
            obs, info = env.reset(
                seed=10_000 + episode
            )

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            positions = []
            attitudes = []
            angular_rates = []

            for _ in range(max_steps):
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(
                    np.asarray(reward).squeeze()
                )

                steps += 1

                positions.append(
                    obs[0, 0:3].copy()
                )

                attitudes.append(
                    obs[0, 3:6].copy()
                )

                angular_rates.append(
                    obs[0, 9:12].copy()
                )

                if gui:
                    env.render()

                if terminated or truncated:
                    break

            positions = np.asarray(positions)
            attitudes = np.asarray(attitudes)
            angular_rates = np.asarray(
                angular_rates
            )

            target = np.asarray(
                env.TARGET_POS,
                dtype=np.float32,
            )

            position_errors = np.linalg.norm(
                positions - target,
                axis=1,
            )

            position_rmse = np.sqrt(
                np.mean(position_errors**2)
            )

            rate_norms = np.linalg.norm(
                angular_rates,
                axis=1,
            )

            # In this environment, terminated is not used
            # for ordinary time-limit success. Full survival
            # means reaching the time horizon without an
            # early truncation.
            survived_horizon = (
                steps >= max_steps
                and not truncated
            )

            # Because the environment itself normally returns
            # truncated=True at the time horizon, also accept
            # an episode whose length reached the expected
            # horizon.
            expected_horizon = int(
                env.EPISODE_LEN_SEC
                * env.CTRL_FREQ
            ) + 1

            survived_horizon = (
                steps >= expected_horizon
            )

            rewards.append(total_reward)
            lengths.append(steps)
            successes.append(
                1 if survived_horizon else 0
            )

            position_rmses.append(
                position_rmse
            )

            mean_position_errors.append(
                np.mean(position_errors)
            )

            max_rolls.append(
                np.max(np.abs(attitudes[:, 0]))
            )

            max_pitches.append(
                np.max(np.abs(attitudes[:, 1]))
            )

            max_rate_norms.append(
                np.max(rate_norms)
            )

            if verbose:
                print(
                    f"Episode {episode + 1:03d}: "
                    f"return={total_reward:.3f}, "
                    f"steps={steps}, "
                    f"terminated={terminated}, "
                    f"truncated={truncated}, "
                    f"position_rmse={position_rmse:.4f}, "
                    f"max_rate={np.max(rate_norms):.4f}"
                )

    finally:
        env.close()

    rewards = np.asarray(rewards)
    lengths = np.asarray(lengths)
    successes = np.asarray(successes)

    position_rmses = np.asarray(
        position_rmses
    )

    mean_position_errors = np.asarray(
        mean_position_errors
    )

    max_rolls = np.asarray(max_rolls)
    max_pitches = np.asarray(max_pitches)
    max_rate_norms = np.asarray(
        max_rate_norms
    )

    result = {
        "model": model_path,
        "phi_range_deg": phi_range_deg,
        "omega_range_rad_s": omega_range_rad_s,
        "yaw_range_deg": yaw_range_deg,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_length": float(lengths.mean()),
        "std_length": float(lengths.std()),
        "success_rate": float(successes.mean()),
        "mean_position_rmse": float(
            position_rmses.mean()
        ),
        "mean_position_error": float(
            mean_position_errors.mean()
        ),
        "mean_max_roll_rad": float(
            max_rolls.mean()
        ),
        "mean_max_pitch_rad": float(
            max_pitches.mean()
        ),
        "mean_max_rate_norm": float(
            max_rate_norms.mean()
        ),
    }

    if verbose:
        print("\n" + "-" * 80)
        print(f"Model: {model_path}")
        print(
            "Test: "
            f"roll/pitch ∈ "
            f"[-{phi_range_deg}, +{phi_range_deg}] deg, "
            f"yaw ∈ "
            f"[-{yaw_range_deg}, +{yaw_range_deg}] deg, "
            f"angular-rate components ∈ "
            f"[-{omega_range_rad_s}, "
            f"+{omega_range_rad_s}] rad/s"
        )
        print(f"Episodes: {n_episodes}")
        print(f"Max steps: {max_steps}")
        print(
            f"Mean reward: "
            f"{result['mean_reward']:.3f} "
            f"+/- {result['std_reward']:.3f}"
        )
        print(
            f"Mean length: "
            f"{result['mean_length']:.2f} "
            f"+/- {result['std_length']:.2f}"
        )
        print(
            f"Success rate: "
            f"{result['success_rate'] * 100:.1f}%"
        )
        print(
            f"Mean position RMSE: "
            f"{result['mean_position_rmse']:.5f} m"
        )
        print(
            f"Mean position error: "
            f"{result['mean_position_error']:.5f} m"
        )
        print(
            f"Mean maximum |roll|: "
            f"{np.rad2deg(result['mean_max_roll_rad']):.3f} deg"
        )
        print(
            f"Mean maximum |pitch|: "
            f"{np.rad2deg(result['mean_max_pitch_rad']):.3f} deg"
        )
        print(
            f"Mean maximum rate norm: "
            f"{result['mean_max_rate_norm']:.5f} rad/s"
        )
        print("-" * 80)

    return result


def print_summary(results):
    """Print a comparison table."""

    print("\n=== SUMMARY TABLE ===")
    print(
        f"{'Model':<55} | "
        f"{'Reward':>10} | "
        f"{'Std':>8} | "
        f"{'Success':>9} | "
        f"{'Length':>9}"
    )
    print("-" * 105)

    for result in results:
        model_name = os.path.basename(
            os.path.dirname(result["model"])
        )

        print(
            f"{model_name:<55} | "
            f"{result['mean_reward']:>10.3f} | "
            f"{result['std_reward']:>8.3f} | "
            f"{result['success_rate'] * 100:>8.1f}% | "
            f"{result['mean_length']:>9.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PPO models on the actual "
            "PyBullet attitude-hover environment."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "Paths to best_model.zip or final_model.zip."
        ),
    )

    parser.add_argument(
        "--test_phi",
        type=float,
        required=True,
        help=(
            "Maximum absolute initial roll and pitch "
            "in degrees."
        ),
    )

    parser.add_argument(
        "--test_omega",
        type=float,
        default=0.0,
        help=(
            "Maximum absolute initial angular-rate "
            "component in rad/s."
        ),
    )

    parser.add_argument(
        "--test_yaw",
        type=float,
        default=180.0,
        help=(
            "Maximum absolute initial yaw angle "
            "in degrees."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=(
            "Maximum evaluation steps. Defaults to "
            "environment horizon plus one."
        ),
    )

    parser.add_argument(
        "--gui",
        action="store_true",
    )

    args = parser.parse_args()

    results = []

    for model_path in args.models:
        result = evaluate_model(
            model_path=model_path,
            phi_range_deg=args.test_phi,
            omega_range_rad_s=args.test_omega,
            yaw_range_deg=args.test_yaw,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            gui=args.gui,
            verbose=True,
        )

        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()