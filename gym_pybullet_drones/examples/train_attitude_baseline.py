"""Train a four-motor PPO hover controller.

This is the baseline experiment for the custom attitude environment.

The policy:
- receives kinematic observations,
- receives recent action history,
- outputs four independent motor actions,
- uses PyBullet physics,
- is trained with the position-only hover reward.

Run from the repository root:

    python -m gym_pybullet_drones.examples.train_attitude_baseline
"""

import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from gym_pybullet_drones.envs.AttitudeHoverAviary import (
    AttitudeHoverAviary,
)
from gym_pybullet_drones.utils.enums import (
    ActionType,
    ObservationType,
    Physics,
)


DEFAULT_OUTPUT_FOLDER = "results_attitude"
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_EVAL_FREQ = 5_000
DEFAULT_EPISODES = 5


def make_environment(gui=False):
    """Create one baseline attitude environment."""

    return AttitudeHoverAviary(
        gui=gui,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        physics=Physics.PYB,
        max_roll_deg=0.0,
        max_pitch_deg=0.0,
        max_yaw_deg=0.0,
        max_ang_vel_rad_s=0.0,
        episode_len_sec=8.0,
    )


def evaluate_saved_model(model_path, episodes=5):
    """Run deterministic evaluation of a saved model."""

    print(f"[INFO] Loading model for evaluation: {model_path}")

    model = PPO.load(model_path)

    eval_env = Monitor(
        make_environment(gui=False)
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=True,
    )

    print(
        f"[INFO] Evaluation reward: "
        f"{mean_reward:.3f} +/- {std_reward:.3f}"
    )

    eval_env.close()


def train(total_timesteps, eval_freq, output_folder):
    """Train and save the baseline PPO policy."""

    timestamp = datetime.now().strftime(
        "%m.%d.%Y_%H.%M.%S"
    )

    run_folder = os.path.join(
        output_folder,
        f"baseline_{timestamp}",
    )

    os.makedirs(run_folder, exist_ok=True)

    print("[INFO] Creating training environment.")

    train_env = make_vec_env(
        AttitudeHoverAviary,
        env_kwargs=dict(
            gui=False,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            physics=Physics.PYB,
            max_roll_deg=0.0,
            max_pitch_deg=0.0,
            max_yaw_deg=0.0,
            max_ang_vel_rad_s=0.0,
            episode_len_sec=8.0,
        ),
        n_envs=1,
        seed=0,
    )

    eval_env = Monitor(
        make_environment(gui=False)
    )

    print("[INFO] Training action space:")
    print(train_env.action_space)

    print("[INFO] Training observation space:")
    print(train_env.observation_space)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        seed=0,
        tensorboard_log=os.path.join(
            run_folder,
            "tensorboard",
        ),
    )

    print("[INFO] PPO policy architecture:")
    print(model.policy)

    # The baseline reward is approximately:
    # reward ≈ 2 per control step when hovering.
    # At 30 Hz for 8 seconds, the rough maximum is about 480.
    #
    # We do not stop early by default because this is a new task
    # with a different action space from the original example.
    callback = EvalCallback(
        eval_env,
        best_model_save_path=run_folder,
        log_path=run_folder,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    print(
        f"[INFO] Starting PPO training for "
        f"{total_timesteps} timesteps."
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    final_model_path = os.path.join(
        run_folder,
        "final_model",
    )

    model.save(final_model_path)

    print(
        f"[INFO] Final model saved to: "
        f"{final_model_path}.zip"
    )

    train_env.close()
    eval_env.close()

    best_model_path = os.path.join(
        run_folder,
        "best_model.zip",
    )

    if os.path.isfile(best_model_path):
        evaluate_saved_model(
            best_model_path,
            episodes=DEFAULT_EPISODES,
        )
    else:
        evaluate_saved_model(
            f"{final_model_path}.zip",
            episodes=DEFAULT_EPISODES,
        )

    return run_folder


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline four-motor PPO controller."
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Number of PPO training timesteps.",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=DEFAULT_EVAL_FREQ,
        help="Evaluation frequency in PPO timesteps.",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help="Folder for checkpoints and logs.",
    )

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    main()