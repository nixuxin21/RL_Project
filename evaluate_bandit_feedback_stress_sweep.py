"""
Stress-scenario sweep for bandit-feedback IRS-assisted MS-AirComp.

This wrapper reuses the strict aggregate-feedback evaluator while varying the
physical task difficulty. It reports both raw scheduling metrics and a simple
node-equivalent utility that charges for latency and probe calls:

    utility = success - slot_cost * slots - probe_cost * total_probes

The goal is to expose settings where limited-feedback IRS selection creates a
clearer tradeoff than the default environment.
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evaluate_bandit_feedback_ms_aircomp as bandit
from evaluate_policy_comparison import ensure_parent_dir, format_float_for_suffix


BASE_SCENARIO = {
    "num_nodes": 50,
    "num_slots": 10,
    "num_irs_elements": 64,
    "num_codebook_states": 16,
}


SCENARIO_PRESETS = {
    "default": {
        "label": "Default",
        "description": "K=50, N=10, M=64, C=16",
        "overrides": {},
    },
    "short_slots": {
        "label": "Short Slots",
        "description": "Tighter deadline with K=50, N=5, M=64, C=16",
        "overrides": {"num_slots": 5},
    },
    "many_nodes": {
        "label": "Many Nodes",
        "description": "Larger backlog with K=80, N=10, M=64, C=16",
        "overrides": {"num_nodes": 80},
    },
    "small_irs": {
        "label": "Small IRS",
        "description": "Weaker IRS aperture with K=50, N=10, M=32, C=16",
        "overrides": {"num_irs_elements": 32},
    },
    "small_codebook": {
        "label": "Small Codebook",
        "description": "Coarser IRS search space with K=50, N=10, M=64, C=8",
        "overrides": {"num_codebook_states": 8},
    },
    "compound_hard": {
        "label": "Compound Hard",
        "description": "Combined stress with K=80, N=5, M=32, C=8",
        "overrides": {
            "num_nodes": 80,
            "num_slots": 5,
            "num_irs_elements": 32,
            "num_codebook_states": 8,
        },
    },
}


BASELINE_POLICIES = {
    "no_irs": bandit.POLICY_NO_IRS,
    "fixed": bandit.POLICY_FIXED_IRS,
    "random_irs": bandit.POLICY_RANDOM_IRS,
    "oracle": bandit.POLICY_ORACLE_FULL,
    "full_feedback": bandit.POLICY_FULL_FEEDBACK,
}


PROBE_POLICIES = {
    "random": bandit.POLICY_RANDOM_PROBE,
    "rotating": bandit.POLICY_ROTATING_GRID,
    "ucb": bandit.POLICY_UCB_PROBE,
    "thompson": bandit.POLICY_THOMPSON_PROBE,
}


CSV_FIELDS = [
    "scenario",
    "scenario_label",
    "scenario_description",
    "num_nodes",
    "num_slots",
    "num_irs_elements",
    "num_codebook_states",
    "g_th",
    "alpha_th",
    "slot_cost",
    "probe_cost",
    "utility_mean",
    "total_probe_calls_mean",
    "feedback_noise_std",
    "probe_budget",
    "budget_fraction",
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
    "probe_calls_per_slot_mean",
    "oracle_match_rate",
    "oracle_tx_gap_mean",
    "observed_score_mean",
    "observed_tx_fraction_mean",
    "avg_reward",
]


def parse_csv_items(value):
    """Parse a comma-separated list, preserving non-empty strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    """Parse stress-sweep arguments."""
    parser = argparse.ArgumentParser(
        description="Run stress scenarios for aggregate-feedback IRS probing."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--scenarios", default="default,short_slots,small_codebook,compound_hard")
    parser.add_argument("--probe-budgets", default="1,2,4")
    parser.add_argument("--feedback-noise-std-values", default="0.2")
    parser.add_argument("--baseline-policies", default="no_irs,fixed,random_irs,oracle,full_feedback")
    parser.add_argument("--probe-policies", default="random,rotating,ucb,thompson")
    parser.add_argument("--slot-cost", type=float, default=0.10)
    parser.add_argument("--probe-cost", type=float, default=0.005)
    parser.add_argument("--power-feedback-noise-std", type=float, default=0.0)
    parser.add_argument("--feedback-power-weight", type=float, default=0.05)
    parser.add_argument("--ucb-coeff", type=float, default=0.25)
    parser.add_argument("--thompson-std", type=float, default=0.20)
    parser.add_argument("--bandit-lr", type=float, default=0.60)
    parser.add_argument("--bandit-prior", type=float, default=0.0)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--fixed-irs-index", type=int, default=7)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_stress_args(args):
    """Validate stress-sweep specific arguments."""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if args.slot_cost < 0.0:
        raise ValueError("--slot-cost must be non-negative")
    if args.probe_cost < 0.0:
        raise ValueError("--probe-cost must be non-negative")

    if args.scenarios == "all":
        args.scenarios = list(SCENARIO_PRESETS)
    else:
        args.scenarios = parse_csv_items(args.scenarios)
    unknown_scenarios = [name for name in args.scenarios if name not in SCENARIO_PRESETS]
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {unknown_scenarios}")

    args.baseline_policies = parse_csv_items(args.baseline_policies)
    unknown_baselines = [name for name in args.baseline_policies if name not in BASELINE_POLICIES]
    if unknown_baselines:
        raise ValueError(f"Unknown baseline policies: {unknown_baselines}")

    args.probe_policies = parse_csv_items(args.probe_policies)
    unknown_probe_policies = [name for name in args.probe_policies if name not in PROBE_POLICIES]
    if unknown_probe_policies:
        raise ValueError(f"Unknown probe policies: {unknown_probe_policies}")


def resolve_output_prefix(args):
    """Resolve output prefix for stress CSVs and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    scenario_label = "-".join(args.scenarios)
    budget_label = "-".join(format_float_for_suffix(value) for value in bandit.parse_int_list(args.probe_budgets))
    noise_label = "-".join(
        format_float_for_suffix(value) for value in bandit.parse_float_list(args.feedback_noise_std_values)
    )
    suffix = (
        f"ep{args.episodes}_runs{args.num_seeds}_{seed_label}_"
        f"{scenario_label}_b{budget_label}_fbnoise{noise_label}_"
        f"slot{format_float_for_suffix(args.slot_cost)}_probe{format_float_for_suffix(args.probe_cost)}"
    )
    output_prefix = os.path.join("results", "bandit_feedback", f"bandit_feedback_stress_{suffix}")
    ensure_parent_dir(output_prefix)
    return output_prefix


def scenario_config(name):
    """Return merged physical parameters for a scenario preset."""
    preset = SCENARIO_PRESETS[name]
    config = dict(BASE_SCENARIO)
    config.update(preset["overrides"])
    return config


def build_bandit_args(args, scenario_name):
    """Build the argument namespace expected by the bandit evaluator."""
    config = scenario_config(scenario_name)
    bandit_args = argparse.Namespace(
        episodes=args.episodes,
        seed=args.seed,
        num_seeds=args.num_seeds,
        seed_stride=args.seed_stride,
        probe_budgets=args.probe_budgets,
        feedback_noise_std_values=args.feedback_noise_std_values,
        power_feedback_noise_std=args.power_feedback_noise_std,
        feedback_power_weight=args.feedback_power_weight,
        ucb_coeff=args.ucb_coeff,
        thompson_std=args.thompson_std,
        bandit_lr=args.bandit_lr,
        bandit_prior=args.bandit_prior,
        num_nodes=config["num_nodes"],
        num_slots=config["num_slots"],
        num_irs_elements=config["num_irs_elements"],
        num_codebook_states=config["num_codebook_states"],
        g_th=args.g_th,
        alpha_th=args.alpha_th,
        fixed_irs_index=args.fixed_irs_index,
        output_prefix=None,
        no_plots=args.no_plots,
    )
    bandit.validate_args(bandit_args)
    return bandit_args


def evaluate_selected_suite(args, bandit_args, episode_seed_sets, feedback_noise_std, base_action):
    """Run the selected baseline and probe policies for one scenario/noise pair."""
    seed_result_sets = []
    baseline_policy_names = [BASELINE_POLICIES[name] for name in args.baseline_policies]
    probe_policy_names = [PROBE_POLICIES[name] for name in args.probe_policies]

    for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
        run_seed = bandit.make_run_seeds(bandit_args)[run_idx - 1]
        print(f"Run seed [{run_idx}/{len(episode_seed_sets)}]: {run_seed}")
        seed_results = []

        for policy_name in baseline_policy_names:
            budget = bandit_args.num_codebook_states if policy_name in {
                bandit.POLICY_ORACLE_FULL,
                bandit.POLICY_FULL_FEEDBACK,
            } else 0
            seed_results.append(
                bandit.evaluate_policy(
                    episode_seeds,
                    bandit_args,
                    feedback_noise_std,
                    policy_name,
                    budget,
                    base_action,
                )
            )

        for budget in bandit_args.probe_budgets:
            for policy_name in probe_policy_names:
                seed_results.append(
                    bandit.evaluate_policy(
                        episode_seeds,
                        bandit_args,
                        feedback_noise_std,
                        policy_name,
                        budget,
                        base_action,
                    )
                )

        seed_result_sets.append(seed_results)

    return bandit.summarize_results(bandit_args, bandit.aggregate_seed_results(seed_result_sets))


def attach_stress_metadata(rows, args, scenario_name, bandit_args):
    """Add scenario metadata and utility columns to bandit summary rows."""
    preset = SCENARIO_PRESETS[scenario_name]
    for row in rows:
        total_probe_calls = float(row["probe_calls_per_slot_mean"]) * float(row["slots_mean"])
        utility = (
            float(row["success_mean"])
            - float(args.slot_cost) * float(row["slots_mean"])
            - float(args.probe_cost) * total_probe_calls
        )
        row.update(
            {
                "scenario": scenario_name,
                "scenario_label": preset["label"],
                "scenario_description": preset["description"],
                "num_nodes": bandit_args.num_nodes,
                "num_slots": bandit_args.num_slots,
                "num_irs_elements": bandit_args.num_irs_elements,
                "num_codebook_states": bandit_args.num_codebook_states,
                "g_th": bandit_args.g_th,
                "alpha_th": bandit_args.alpha_th,
                "slot_cost": args.slot_cost,
                "probe_cost": args.probe_cost,
                "total_probe_calls_mean": total_probe_calls,
                "utility_mean": utility,
            }
        )
    return rows


def write_csv(path, rows):
    """Write the stress summary CSV."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def is_oracle(row):
    """Return whether a row is an offline oracle diagnostic."""
    return row["policy"] == bandit.POLICY_ORACLE_FULL


def compact_policy_label(row):
    """Return compact labels for console and plots."""
    label = bandit.policy_label(row)
    return label.replace(" Feedback Probe", "").replace(" Feedback", "")


def print_stress_summary(rows):
    """Print best non-oracle policy per scenario and feedback noise level."""
    print("=" * 132)
    print("Bandit-Feedback Stress Summary")
    print("=" * 132)
    print(
        f"{'Scenario':<16} {'Noise':>6} {'Best non-oracle':<34} {'Utility':>9} "
        f"{'Success':>9} {'Perfect%':>9} {'Slots':>7} {'Probes':>8} {'Oracle':>9}"
    )
    groups = sorted({(row["scenario"], row["feedback_noise_std"]) for row in rows})
    for scenario_name, noise in groups:
        subset = [
            row
            for row in rows
            if row["scenario"] == scenario_name and row["feedback_noise_std"] == noise
        ]
        non_oracle = [row for row in subset if not is_oracle(row)]
        oracle_rows = [row for row in subset if is_oracle(row)]
        best = max(non_oracle, key=lambda row: row["utility_mean"])
        oracle_success = oracle_rows[0]["success_mean"] if oracle_rows else float("nan")
        print(
            f"{scenario_name:<16} {noise:>6.3f} {compact_policy_label(best):<34} "
            f"{best['utility_mean']:>9.3f} {best['success_mean']:>9.3f} "
            f"{best['perfect_rate']:>8.2f}% {best['slots_mean']:>7.3f} "
            f"{best['total_probe_calls_mean']:>8.2f} {oracle_success:>9.3f}"
        )


def plot_results(rows, output_prefix):
    """Plot latency/probe tradeoffs for each stress scenario."""
    scenarios = [name for name in SCENARIO_PRESETS if any(row["scenario"] == name for row in rows)]
    if not scenarios:
        return

    labels = []
    for row in rows:
        if is_oracle(row):
            continue
        label = compact_policy_label(row)
        if label not in labels:
            labels.append(label)

    cols = 2
    plot_rows = int(np.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(plot_rows, cols, figsize=(15, 5 * plot_rows), squeeze=False)
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(idx % 20) for idx, label in enumerate(labels)}

    for ax, scenario_name in zip(axes.ravel(), scenarios):
        subset = [row for row in rows if row["scenario"] == scenario_name and not is_oracle(row)]
        for row in subset:
            label = compact_policy_label(row)
            ax.scatter(
                row["total_probe_calls_mean"],
                row["slots_mean"],
                s=36 + row["perfect_rate"] * 0.35,
                color=colors[label],
                alpha=0.78,
                label=label,
            )
        best = max(subset, key=lambda row: row["utility_mean"])
        ax.scatter(
            best["total_probe_calls_mean"],
            best["slots_mean"],
            s=190,
            facecolors="none",
            edgecolors="black",
            linewidths=1.7,
        )
        ax.set_title(SCENARIO_PRESETS[scenario_name]["label"])
        ax.set_xlabel("Total Probe Calls / Episode")
        ax.set_ylabel("Slots Used")
        ax.grid(True, linestyle="--", alpha=0.35)

    for ax in axes.ravel()[len(scenarios):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="lower center",
        ncol=min(4, max(1, len(dedup))),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = f"{output_prefix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    """Run stress scenarios for aggregate-feedback IRS probing."""
    args = parse_args()
    validate_stress_args(args)
    output_prefix = resolve_output_prefix(args)

    print("=" * 96)
    print(
        f"Bandit-feedback stress sweep: episodes={args.episodes}, num_seeds={args.num_seeds}, "
        f"scenarios={args.scenarios}, feedback_noise={args.feedback_noise_std_values}, "
        f"budgets={args.probe_budgets}"
    )
    print(
        f"Utility: success - {args.slot_cost:g} * slots - {args.probe_cost:g} * total_probes; "
        f"output_prefix={output_prefix}"
    )
    print("=" * 96)

    all_rows = []
    for scenario_name in args.scenarios:
        bandit_args = build_bandit_args(args, scenario_name)
        base_action = bandit.make_base_action(bandit_args)
        run_seeds = bandit.make_run_seeds(bandit_args)
        episode_seed_sets = [bandit.make_episode_seeds(bandit_args, run_seed) for run_seed in run_seeds]
        print("=" * 96)
        print(f"Scenario: {scenario_name} ({SCENARIO_PRESETS[scenario_name]['description']})")
        print("=" * 96)

        for feedback_noise_std in bandit_args.feedback_noise_std_values:
            print("=" * 96)
            print(f"Aggregate feedback noise std={feedback_noise_std:g}")
            print("=" * 96)
            rows = evaluate_selected_suite(
                args,
                bandit_args,
                episode_seed_sets,
                feedback_noise_std,
                base_action,
            )
            all_rows.extend(attach_stress_metadata(rows, args, scenario_name, bandit_args))

    print_stress_summary(all_rows)
    write_csv(f"{output_prefix}.csv", all_rows)
    if not args.no_plots:
        plot_results(all_rows, output_prefix)


if __name__ == "__main__":
    main()
