import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def find_evaluation_files(results_dir):
    """Find all EvalCallback evaluation files below a folder."""

    patterns = [
        os.path.join(
            results_dir,
            "**",
            "evaluations.npz",
        ),
        os.path.join(
            results_dir,
            "**",
            "evaluations_results.npz",
        ),
    ]

    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(
                pattern,
                recursive=True,
            )
        )

    return sorted(set(files))


def load_evaluations(path):
    """Load arrays saved by Stable-Baselines3 EvalCallback."""

    with np.load(path) as data:
        print(f"\n[INFO] File: {path}")
        print("[INFO] Available arrays:", data.files)

        timesteps = data["timesteps"]
        results = data["results"]
        ep_lengths = data["ep_lengths"]

    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    mean_lengths = np.mean(ep_lengths, axis=1)

    return (
        timesteps,
        mean_rewards,
        std_rewards,
        mean_lengths,
    )


def plot_file(path, show=True, save=True):
    """Create plots for one evaluation file."""

    (
        timesteps,
        mean_rewards,
        std_rewards,
        mean_lengths,
    ) = load_evaluations(path)

    folder = os.path.dirname(path)
    name = os.path.splitext(
        os.path.basename(path)
    )[0]

    safe_mean_rewards = np.maximum(
        mean_rewards,
        1e-12,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
    )

    axes[0].plot(
        timesteps,
        mean_rewards,
        marker="o",
        linewidth=2,
        label="Mean evaluation reward",
    )

    axes[0].fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.25,
        label="Reward ± standard deviation",
    )

    axes[0].set_ylabel("Evaluation reward")
    axes[0].set_title(
        f"Evaluation results: {os.path.basename(folder)}"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        timesteps,
        mean_lengths,
        marker="o",
        color="tab:orange",
        linewidth=2,
        label="Mean episode length",
    )

    axes[1].set_xlabel("Training timesteps")
    axes[1].set_ylabel("Episode length")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()

    if save:
        output_path = os.path.join(
            folder,
            f"{name}_plots.png",
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"[INFO] Saved plot: {output_path}"
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    fig_log, axes_log = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
    )

    axes_log[0].plot(
        timesteps,
        safe_mean_rewards,
        marker="o",
        linewidth=2,
    )

    axes_log[0].set_yscale("log")
    axes_log[0].set_ylabel(
        "Mean evaluation reward (log scale)"
    )
    axes_log[0].grid(True, which="both", alpha=0.3)

    axes_log[1].plot(
        timesteps,
        mean_lengths,
        marker="o",
        color="tab:orange",
        linewidth=2,
    )

    axes_log[1].set_xlabel("Training timesteps")
    axes_log[1].set_ylabel("Mean episode length")
    axes_log[1].grid(True, alpha=0.3)

    fig_log.tight_layout()

    if save:
        log_output_path = os.path.join(
            folder,
            f"{name}_log_plots.png",
        )

        fig_log.savefig(
            log_output_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"[INFO] Saved log plot: "
            f"{log_output_path}"
        )

    if show:
        plt.show()
    else:
        plt.close(fig_log)


def plot_all(results_dir, show=False):
    """Plot every evaluation file under results_dir."""

    files = find_evaluation_files(results_dir)

    if not files:
        raise FileNotFoundError(
            f"No evaluation files found under {results_dir}"
        )

    for path in files:
        plot_file(
            path,
            show=show,
            save=True,
        )


def compare_all(results_dir, output_name):
    """Compare all runs on one reward/episode-length figure."""

    files = find_evaluation_files(results_dir)

    if not files:
        raise FileNotFoundError(
            f"No evaluation files found under {results_dir}"
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=False,
    )

    for path in files:
        (
            timesteps,
            mean_rewards,
            std_rewards,
            mean_lengths,
        ) = load_evaluations(path)

        label = os.path.basename(
            os.path.dirname(path)
        )

        axes[0].plot(
            timesteps,
            mean_rewards,
            marker="o",
            linewidth=2,
            label=label,
        )

        axes[1].plot(
            timesteps,
            mean_lengths,
            marker="o",
            linewidth=2,
            label=label,
        )

    axes[0].set_title("Comparison of evaluation rewards")
    axes[0].set_xlabel("Training timesteps")
    axes[0].set_ylabel("Mean reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Comparison of episode lengths")
    axes[1].set_xlabel("Training timesteps")
    axes[1].set_ylabel("Mean episode length")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        output_name,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"[INFO] Saved comparison plot: {output_name}"
    )

    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        default="results_attitude",
        help="Root folder containing baseline run folders.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Create one comparison plot for all runs.",
    )

    args = parser.parse_args()

    if args.compare:
        compare_all(
            results_dir=args.results_dir,
            output_name=os.path.join(
                args.results_dir,
                "all_runs_comparison.png",
            ),
        )
    else:
        plot_all(
            results_dir=args.results_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()