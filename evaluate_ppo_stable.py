import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.StableHoverAviary import StableHoverAviary

def evaluate(model_path, n_episodes=30):
    """Evaluate the trained model."""

    env = StableHoverAviary(
        obs='kin',
        act='rpm',
        ctrl_freq=30,
        gui=True,  # Enable GUI for visualization
    )
    model = PPO.load(model_path)

    rewards = []
    success_count = 0
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        episode_reward = 0
        steps = 0

        # Track all metrics for post-analysis
        all_errors = []
        all_velocities = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

            # Track all errors and velocities
            all_errors.append(info['position_error'])
            # Get velocity from state
            state = env._getDroneStateVector(0)
            vel = np.linalg.norm(state[10:13])
            all_velocities.append(vel)

        rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Success: remained stable in last 1 second (last 30 steps)
        if len(all_errors) >= 30:
            last_second_errors = all_errors[-30:]
            last_second_velocities = all_velocities[-30:]
        else:
            last_second_errors = all_errors
            last_second_velocities = all_velocities

        mean_error = np.mean(last_second_errors)
        mean_vel = np.mean(last_second_velocities)

        if mean_error < 0.05 and mean_vel < 0.05:
            success_count += 1

        print(f"Episode {ep+1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Steps={steps}, "
              f"Success={success_count}")

    # Print summary
    print(f"\n=== Evaluation Results ===")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"Success criteria: mean position error < 5cm, mean velocity < 5cm/s in last 1s")

if __name__ == '__main__':
    evaluate('results/stable_hover/final_model.zip', n_episodes=30)