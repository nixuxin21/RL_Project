"""
Noisy codebook feature robustness sweep.

This experiment keeps the physical channel and exact greedy oracle unchanged, but
adds Gaussian observation noise to the C-dimensional codebook quality features
seen by feature-based policies. It is intended to test whether the current
Feature Argmax / PowerTie advantage survives when those features are no longer
exact previews.
"""

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_policy_comparison import (
    codebook_index_to_action,
    ensure_parent_dir,
    evaluate_codebook_aware_sac_policy,
    evaluate_feature_argmax_irs_policy,
    evaluate_feature_argmax_power_tie_irs_policy,
    evaluate_fixed_action_policy,
    evaluate_greedy_irs_policy,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
    metric_mean_ci,
    physical_to_action,
)


def parse_float_list(value):
    """Parse a comma-separated float list such as '0,0.05,0.1'."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    """Parse noisy feature sweep parameters."""
    parser = argparse.ArgumentParser(
        description="Sweep Gaussian noise levels on codebook quality features."
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--noise-std-values", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--fixed-irs-index", type=int, default=7)
    parser.add_argument("--model-dir", default="./rl_models")
    parser.add_argument("--include-codebook-aware-sac", action="store_true")
    parser.add_argument("--codebook-aware-model-name", default="sac_codebook_aware_irs_selector.zip")
    parser.add_argument(
        "--codebook-aware-stats-name",
        default="vec_normalize_codebook_aware_irs_selector.pkl",
    )
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """Validate positive sizes and non-negative noise levels."""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    for name in ("num_nodes", "num_slots", "num_irs_elements", "num_codebook_states"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    args.noise_std_values = parse_float_list(args.noise_std_values)
    if not args.noise_std_values:
        raise ValueError("--noise-std-values must contain at least one value")
    if any(value < 0.0 for value in args.noise_std_values):
        raise ValueError("--noise-std-values must be non-negative")


def resolve_output_prefix(args):
    """Resolve the shared output prefix for CSV and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    suffix = f"ep{args.episodes}_runs{args.num_seeds}_{seed_label}"
    if args.include_codebook_aware_sac:
        suffix += "_cbsac"
    output_prefix = os.path.join("results", "noisy_features", f"noisy_feature_sweep_{suffix}")
    ensure_parent_dir(output_prefix)
    return output_prefix


def make_actions(args):
    """Build fixed transmission actions shared by the baseline policies."""
    fixed_index = int(np.clip(args.fixed_irs_index, 0, args.num_codebook_states - 1))
    args.fixed_irs_index = fixed_index
    g_action = physical_to_action(args.g_th, low=0.001, scale=0.05)
    alpha_action = physical_to_action(args.alpha_th, low=0.05, scale=0.05)
    fixed_irs_action = codebook_index_to_action(fixed_index, args.num_codebook_states)
    fixed_action = np.array([g_action, alpha_action, fixed_irs_action], dtype=np.float32)
    random_action = np.array([g_action, alpha_action, 0.0], dtype=np.float32)
    return fixed_action, random_action


def run_policy_suite_for_noise(args, episode_seeds, fixed_action, random_action):
    """Run the noisy-feature policy subset for one noise level and one run seed."""
    results = []
    if args.include_codebook_aware_sac:
        results.append(evaluate_codebook_aware_sac_policy(episode_seeds, args))

    results.append(evaluate_feature_argmax_irs_policy(episode_seeds, args, fixed_action))
    results.append(evaluate_feature_argmax_power_tie_irs_policy(episode_seeds, args, fixed_action))
    results.append(evaluate_greedy_irs_policy(episode_seeds, args, fixed_action))
    results.append(evaluate_fixed_action_policy("Random IRS", "random", random_action, episode_seeds, args))
    results.append(evaluate_fixed_action_policy("No IRS", "none", random_action, episode_seeds, args))
    return results


def seed_summary(result):
    """Summarize a policy result within one run seed."""
    keys = (
        "success_nodes",
        "avg_mse",
        "avg_power",
        "episode_reward",
        "slots_used",
        "total_energy",
        "decision_preview_calls_per_slot",
        "tie_candidates_mean",
    )
    return {key: float(np.mean(result[key])) for key in keys}


def aggregate_seed_results(seed_result_sets):
    """Aggregate policy results across run seeds."""
    if not seed_result_sets:
        return []

    aggregated_results = []
    result_count = len(seed_result_sets[0])
    for result_idx in range(result_count):
        parts = [seed_results[result_idx] for seed_results in seed_result_sets]
        aggregated = {"name": parts[0]["name"], "tx_per_slot": []}
        for key in seed_summary(parts[0]):
            aggregated[key] = np.concatenate([part[key] for part in parts])
        for part in parts:
            aggregated["tx_per_slot"].extend(part["tx_per_slot"])
        aggregated["seed_summaries"] = [seed_summary(part) for part in parts]
        aggregated_results.append(aggregated)
    return aggregated_results


def summarize_noise_results(args, noise_std, results):
    """Convert one noise level's aggregated results into CSV rows."""
    rows = []
    for result in results:
        success_mean, success_ci95 = metric_mean_ci(result, "success_nodes")
        slots_mean, slots_ci95 = metric_mean_ci(result, "slots_used")
        energy_mean, energy_ci95 = metric_mean_ci(result, "total_energy")
        rows.append(
            {
                "noise_std": noise_std,
                "policy": result["name"],
                "episodes": len(result["success_nodes"]),
                "num_seeds": args.num_seeds,
                "success_mean": success_mean,
                "success_ci95": success_ci95,
                "success_rate_mean": success_mean / args.num_nodes,
                "perfect_rate": float(np.mean(result["success_nodes"] == args.num_nodes) * 100.0),
                "slots_mean": slots_mean,
                "slots_ci95": slots_ci95,
                "avg_power": float(np.mean(result["avg_power"])),
                "total_energy_mean": energy_mean,
                "total_energy_ci95": energy_ci95,
                "decision_preview_calls_per_slot_mean": float(
                    np.mean(result["decision_preview_calls_per_slot"])
                ),
                "tie_candidates_mean": float(np.mean(result["tie_candidates_mean"])),
                "avg_reward": float(np.mean(result["episode_reward"])),
            }
        )
    return rows


def write_csv(path, rows):
    """Write the noisy sweep summary CSV."""
    ensure_parent_dir(path)
    fieldnames = [
        "noise_std",
        "policy",
        "episodes",
        "num_seeds",
        "success_mean",
        "success_ci95",
        "success_rate_mean",
        "perfect_rate",
        "slots_mean",
        "slots_ci95",
        "avg_power",
        "total_energy_mean",
        "total_energy_ci95",
        "decision_preview_calls_per_slot_mean",
        "tie_candidates_mean",
        "avg_reward",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def plot_results(rows, args, output_prefix):
    """Plot success rate, latency, and energy against feature noise std."""
    policies = []
    for row in rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.get_cmap("tab10")
    colors = {policy: cmap(idx % 10) for idx, policy in enumerate(policies)}

    for policy in policies:
        policy_rows = sorted(
            [row for row in rows if row["policy"] == policy],
            key=lambda row: row["noise_std"],
        )
        x = [row["noise_std"] for row in policy_rows]
        axes[0].plot(
            x,
            [row["success_rate_mean"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[1].plot(
            x,
            [row["slots_mean"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[2].plot(
            x,
            [row["total_energy_mean"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )

    axes[0].set_title("Success Rate vs Feature Noise")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0.0, 1.03)
    axes[1].set_title("Latency vs Feature Noise")
    axes[1].set_ylabel("Slots Used")
    axes[1].set_ylim(0.0, args.num_slots + 1)
    axes[2].set_title("Energy vs Feature Noise")
    axes[2].set_ylabel("Total Energy Proxy")

    for ax in axes:
        ax.set_xlabel("Codebook Feature Noise Std")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = f"{output_prefix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(rows):
    """Print a compact noisy sweep summary."""
    print("=" * 126)
    print("Noisy Codebook Feature Sweep Summary")
    print("=" * 126)
    print(
        f"{'Noise':>7} {'Policy':<30} {'Success':>9} {'Rate':>7} "
        f"{'Perfect%':>9} {'Slots':>8} {'Energy':>10} {'Preview':>9} {'Tie':>7}"
    )
    for row in rows:
        print(
            f"{row['noise_std']:>7.3f} {row['policy']:<30} "
            f"{row['success_mean']:>9.3f} {row['success_rate_mean']:>7.3f} "
            f"{row['perfect_rate']:>8.2f}% {row['slots_mean']:>8.3f} "
            f"{row['total_energy_mean']:>10.3f} "
            f"{row['decision_preview_calls_per_slot_mean']:>9.2f} "
            f"{row['tie_candidates_mean']:>7.2f}"
        )


def main():
    """Run the noisy codebook feature sweep."""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)
    fixed_action, random_action = make_actions(args)
    run_seeds = make_run_seeds(args)
    episode_seed_sets = [make_episode_seeds(args, run_seed) for run_seed in run_seeds]

    print("=" * 96)
    print(
        f"Noisy feature sweep: episodes={args.episodes}, num_seeds={args.num_seeds}, "
        f"base_seed={args.seed}, noise={args.noise_std_values}"
    )
    print(f"Output prefix: {output_prefix}")
    print("=" * 96)

    rows = []
    for noise_std in args.noise_std_values:
        args.codebook_feature_noise_std = float(noise_std)
        print("=" * 96)
        print(f"Feature noise std: {noise_std:g} ({format_float_for_suffix(noise_std)})")
        print("=" * 96)
        seed_result_sets = []
        for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
            print(f"Run seed [{run_idx}/{len(run_seeds)}]: {run_seeds[run_idx - 1]}")
            seed_result_sets.append(
                run_policy_suite_for_noise(args, episode_seeds, fixed_action, random_action)
            )
        rows.extend(summarize_noise_results(args, noise_std, aggregate_seed_results(seed_result_sets)))

    print_summary(rows)
    write_csv(f"{output_prefix}.csv", rows)
    if not args.no_plots:
        plot_results(rows, args, output_prefix)


if __name__ == "__main__":
    main()
