import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.CurriculumHoverAviaryV4 import CurriculumHoverAviaryV4

MODEL_PATH = os.path.join("curriculum_checkpoints_v4", "v4_stage2_3m_ppo_model.zip")
LOG_DIR = "curriculum_checkpoints_v4"


def get_training_history():
    """Extracts training total timesteps and ep_rew_mean from monitor CSV files or logs if available."""
    monitor_files = glob.glob(os.path.join(LOG_DIR, "**", "*.monitor.csv"), recursive=True) + \
                    glob.glob("*.monitor.csv")
    
    if monitor_files:
        dfs = []
        for mf in monitor_files:
            try:
                df = pd.read_csv(mf, skiprows=1)
                dfs.append(df)
            except Exception:
                pass
        if dfs:
            full_df = pd.concat(dfs).sort_values("t")
            full_df["timesteps"] = full_df["l"].cumsum()
            full_df["ep_rew_mean"] = full_df["r"].rolling(window=20, min_periods=1).mean()
            return full_df["timesteps"].values, full_df["ep_rew_mean"].values

    # Fallback to logged checkpoint milestones from your 5.0M AQUA run logs
    timesteps_logged = np.array([
        622592, 688128, 753664, 1212416, 1277952, 1343488, 1376256,
        1998848, 2064384, 2129920, 4718592, 4816896, 4915200, 5013504
    ])
    ep_rew_logged = np.array([
        92.2, 145.0, 165.0, 270.0, 276.0, 280.0, 283.0,
        304.0, 303.0, 305.0, 302.0, 302.0, 298.0, 302.0
    ])
    return timesteps_logged, ep_rew_logged


def evaluate_episode_rewards(model, env_stage, num_episodes=20):
    """Evaluates step-by-step cumulative rewards across episodes."""
    env = CurriculumHoverAviaryV4(gui=False, stage=env_stage)
    
    all_cum_rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_flat = np.array(action, dtype=np.float32).flatten()
            
            obs, reward, terminated, truncated, info = env.step(action_flat)
            step_rewards.append(reward)
            
            done = terminated or truncated
            
        cum_reward = np.cumsum(step_rewards)
        all_cum_rewards.append(cum_reward)
        
    env.close()
    
    # Pad to equal length (240 steps) if any ended early
    max_len = 240
    padded_rewards = np.zeros((num_episodes, max_len))
    for i, cr in enumerate(all_cum_rewards):
        length = min(len(cr), max_len)
        padded_rewards[i, :length] = cr[:length]
        if length < max_len:
            padded_rewards[i, length:] = cr[-1]
            
    return padded_rewards


def plot_cumulative_rewards():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    print("=" * 75)
    print("   GENERATING V4 STAGE 2 CUMULATIVE REWARD DIAGNOSTIC PLOTS")
    print("=" * 75)

    model = PPO.load(MODEL_PATH)

    # 1. Evaluate step-by-step cumulative rewards across 20 episodes
    print("[1/2] Evaluating episode cumulative rewards over timesteps...")
    eval_cum_rewards = evaluate_episode_rewards(model, env_stage=2, num_episodes=20)
    
    mean_cum_reward = np.mean(eval_cum_rewards, axis=0)
    std_cum_reward = np.std(eval_cum_rewards, axis=0)
    
    timesteps_ep = np.arange(1, 241)  # 1 to 240 steps (8.0s)
    time_sec = timesteps_ep / 30.0    # 30 Hz control frequency

    # 2. Retrieve training curve data
    print("[2/2] Retrieving training cumulative return curve data...")
    train_timesteps, train_rewards = get_training_history()

    # Create 2-Panel Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("V4 Stage 2 — Episode Cumulative Reward Metrics", fontsize=15, fontweight='bold')

    # Panel 1: In-Episode Cumulative Reward vs Timesteps (0 - 240 steps)
    ax1.plot(timesteps_ep, mean_cum_reward, color='tab:blue', lw=2.5, label='Mean Cumulative Reward (20 Runs)')
    ax1.fill_between(timesteps_ep, mean_cum_reward - std_cum_reward, mean_cum_reward + std_cum_reward,
                     color='tab:blue', alpha=0.2, label='±1 Std Dev Variance')
    
    # Add individual episode samples
    for i in range(min(5, len(eval_cum_rewards))):
        ax1.plot(timesteps_ep, eval_cum_rewards[i], color='gray', alpha=0.3, lw=1.0,
                 label='Sample Episode' if i == 0 else "")

    ax1.axhline(336.0, color='r', linestyle='--', alpha=0.7, label='Theoretical Max Return (336.0)')
    ax1.set_title("A. In-Episode Cumulative Reward Progression (8.0s Flight)")
    ax1.set_xlabel("Episode Control Timesteps (30 Hz)")
    ax1.set_ylabel("Cumulative Episode Return G_t")
    ax1.set_xlim([0, 240])
    ax1.set_ylim([0, 350])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)

    # Twin axis for seconds on top
    ax1_top = ax1.twiny()
    ax1_top.set_xlim([0, 8.0])
    ax1_top.set_xlabel("Flight Time (Seconds)")

    # Panel 2: Training Cumulative Return vs Total Timesteps (0 - 5.0M)
    ax2.plot(train_timesteps / 1e6, train_rewards, color='tab:green', lw=2.5, marker='o', ms=5, label='Mean Return (ep_rew_mean)')
    ax2.axhline(336.0, color='r', linestyle='--', alpha=0.7, label='Theoretical Max Bound (336.0)')
    ax2.axhline(302.0, color='tab:blue', linestyle=':', alpha=0.8, label='Final Saturation Level (~302)')
    
    ax2.set_title("B. Training Cumulative Return vs Total Timesteps")
    ax2.set_xlabel("Total Training Timesteps (Millions)")
    ax2.set_ylabel("Mean Episode Cumulative Return")
    ax2.set_xlim([0, 5.2])
    ax2.set_ylim([0, 350])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    output_png = "v4_stage2_cumulative_reward_plot.png"
    plt.savefig(output_png, dpi=300)
    print(f"\n -> Plot saved successfully to: {output_png}")
    plt.show()


if __name__ == "__main__":
    plot_cumulative_rewards()