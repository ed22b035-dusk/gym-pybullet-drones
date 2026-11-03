import argparse
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from gym_pybullet_drones.envs.AttitudeHoverStableAviary import (
    AttitudeHoverStableAviary,
)
from gym_pybullet_drones.utils.enums import (
    ActionType,
    ObservationType,
    Physics,
)


def make_environment(gui=False):
    return AttitudeHoverStableAviary(
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


def train(total_timesteps, eval_freq, output_folder):
    timestamp = datetime.now().strftime(
        "%m.%d.%Y_%H.%M.%S"
    )

    run_folder = os.path.join(
        output_folder,
        f"stable_{timestamp}",
    )

    os.makedirs(run_folder, exist_ok=True)

    train_env = make_vec_env(
        AttitudeHoverStableAviary,
        env_kwargs=dict(
            gui=False,
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

    print("[INFO] Action space:")
    print(train_env.action_space)

    print("[INFO] Observation space:")
    print(train_env.observation_space)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=0,
        tensorboard_log=os.path.join(
            run_folder,
            "tensorboard",
        ),
    )

    print("[INFO] Policy:")
    print(model.policy)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_folder,
        log_path=run_folder,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(
        os.path.join(
            run_folder,
            "final_model",
        )
    )

    train_env.close()
    eval_env.close()

    print(
        "[INFO] Saved stable-reward run to:",
        run_folder,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="results_attitude_stable",
    )

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    main()