import numpy as np

from gym_pybullet_drones.envs.AttitudeHoverAviary import (
    AttitudeHoverAviary,
)
from gym_pybullet_drones.utils.enums import (
    ActionType,
    ObservationType,
    Physics,
)


def main():
    env = AttitudeHoverAviary(
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        physics=Physics.PYB,
        max_roll_deg=0.0,
        max_pitch_deg=0.0,
        max_yaw_deg=0.0,
        max_ang_vel_rad_s=0.0,
        episode_len_sec=2.0,
    )

    actions_to_test = [
        np.array(
            [[0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        np.array(
            [[0.5, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        np.array(
            [[0.0, 0.5, 0.0, 0.0]],
            dtype=np.float32,
        ),
        np.array(
            [[0.0, 0.0, 0.5, 0.0]],
            dtype=np.float32,
        ),
        np.array(
            [[0.0, 0.0, 0.0, 0.5]],
            dtype=np.float32,
        ),
    ]

    try:
        print("[INFO] Action space:", env.action_space)
        print("[INFO] Observation space:", env.observation_space)

        for action_index, action in enumerate(actions_to_test):
            print("\n" + "=" * 80)
            print(
                f"[TEST {action_index}] "
                f"action={action.tolist()}"
            )
            print("=" * 80)

            obs, info = env.reset(seed=42)

            initial_roll_rate = info["roll_rate_rad_s"]
            initial_pitch_rate = info["pitch_rate_rad_s"]
            initial_yaw_rate = info["yaw_rate_rad_s"]

            print(
                "[INITIAL] "
                f"roll={info['roll_deg']:+.4f} deg, "
                f"pitch={info['pitch_deg']:+.4f} deg, "
                f"yaw={info['yaw_deg']:+.4f} deg, "
                f"rates=["
                f"{initial_roll_rate:+.6f}, "
                f"{initial_pitch_rate:+.6f}, "
                f"{initial_yaw_rate:+.6f}] rad/s"
            )

            # Apply the same motor pattern for only two control steps.
            # A short duration makes the torque direction easier to identify.
            for step in range(2):
                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                print(
                    f"[STEP {step + 1}] "
                    f"roll={info['roll_deg']:+.4f} deg, "
                    f"pitch={info['pitch_deg']:+.4f} deg, "
                    f"yaw={info['yaw_deg']:+.4f} deg, "
                    f"rates=["
                    f"{info['roll_rate_rad_s']:+.6f}, "
                    f"{info['pitch_rate_rad_s']:+.6f}, "
                    f"{info['yaw_rate_rad_s']:+.6f}] rad/s, "
                    f"reward={reward:.6f}, "
                    f"terminated={terminated}, "
                    f"truncated={truncated}"
                )

                if terminated or truncated:
                    print(
                        "[INFO] Episode ended before two "
                        "control steps."
                    )
                    break

            final_roll_rate = info["roll_rate_rad_s"]
            final_pitch_rate = info["pitch_rate_rad_s"]
            final_yaw_rate = info["yaw_rate_rad_s"]

            delta_rates = np.array(
                [
                    final_roll_rate - initial_roll_rate,
                    final_pitch_rate - initial_pitch_rate,
                    final_yaw_rate - initial_yaw_rate,
                ]
            )

            print(
                "[RESULT] Change in angular rates after test: "
                f"Δroll_rate={delta_rates[0]:+.6f}, "
                f"Δpitch_rate={delta_rates[1]:+.6f}, "
                f"Δyaw_rate={delta_rates[2]:+.6f} rad/s"
            )

    finally:
        env.close()

    print("\n[INFO] Motor mapping diagnostic complete.")


if __name__ == "__main__":
    main()