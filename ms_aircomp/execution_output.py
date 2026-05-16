"""Output naming, progress, summary, and plots for execution mismatch runs."""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ms_aircomp.experiment_utils import ensure_parent_dir, format_float_for_suffix
from ms_aircomp.execution_policy_registry import (
    POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
    POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
    POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
    POLICY_CHOICES,
    POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
    POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
    POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
    POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
    POLICY_SPARSE_TOPK_FEEDBACK_GRID,
)

__all__ = [
    "plot_results",
    "print_progress",
    "print_summary",
    "resolve_output_prefix",
]


def resolve_output_prefix(args):
    """Resolve output prefix for CSV and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    budget_label = "-".join(format_float_for_suffix(value) for value in args.probe_budgets)
    model_label = "-".join(args.mismatch_models)
    rho_label = "-".join(format_float_for_suffix(value) for value in args.channel_rho_values)
    delay_label = "-".join(str(value) for value in args.csi_delay_slots)
    decision_label = "-".join(format_float_for_suffix(value) for value in args.decision_error_std_values)
    execution_label = "-".join(format_float_for_suffix(value) for value in args.execution_error_std_values)
    sparse_label = ""
    if any(
        POLICY_CHOICES.get(alias)
        in {
            POLICY_SPARSE_TOPK_FEEDBACK_GRID,
            POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
            POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
            POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
            POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
            POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
            POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
            POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
        }
        for alias in args.policies
    ):
        sparse_seed_label = "-".join(
            format_float_for_suffix(value) for value in args.sparse_topk_seed_multipliers
        )
        sparse_fraction_label = "-".join(
            format_float_for_suffix(value) for value in args.sparse_topk_fractions
        )
        sparse_label = f"_sparse{sparse_seed_label}_topfrac{sparse_fraction_label}"
    if any(
        POLICY_CHOICES.get(alias)
        in {
            POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
            POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
        }
        for alias in args.policies
    ):
        coverage_label = "-".join(
            format_float_for_suffix(value) for value in args.coverage_sparse_weights
        )
        coverage_power_label = "-".join(
            format_float_for_suffix(value) for value in args.coverage_sparse_power_weights
        )
        sparse_label += f"_coveragecw{coverage_label}_cpw{coverage_power_label}"
    if any(
        POLICY_CHOICES.get(alias)
        in {
            POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
            POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
            POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
            POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
            POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
        }
        for alias in args.policies
    ):
        margin_label = "-".join(
            format_float_for_suffix(value) for value in args.adaptive_sparse_margin_thresholds
        )
        sparse_label += f"_adaptmargin{margin_label}"
    if any(
        POLICY_CHOICES.get(alias) == POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID
        for alias in args.policies
    ):
        preview_cost_label = "-".join(
            format_float_for_suffix(value) for value in args.adaptive_sparse_v2_preview_costs
        )
        sparse_label += f"_v2pc{preview_cost_label}"
    if any(
        POLICY_CHOICES.get(alias)
        in {
            POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
            POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
        }
        for alias in args.policies
    ):
        sparse_label += (
            f"_v3nr{args.adaptive_sparse_v3_neighbor_radius}"
            f"_nc{args.adaptive_sparse_v3_neighbor_count}"
            f"_hc{args.adaptive_sparse_v3_history_count}"
        )
    if any(
        POLICY_CHOICES.get(alias) == POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID
        for alias in args.policies
    ):
        extra_label = "-".join(str(value) for value in args.learned_shortlist_extra_counts)
        sparse_label += f"_learnedextra{extra_label}"
    if any(
        POLICY_CHOICES.get(alias) == POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID
        for alias in args.policies
    ):
        extra_label = "-".join(str(value) for value in args.learned_set_extra_counts)
        sparse_label += f"_learnedsetextra{extra_label}"
    suffix = (
        f"ep{args.episodes}_runs{args.num_seeds}_{seed_label}_b{budget_label}_"
        f"model{model_label}_rho{rho_label}_delay{delay_label}_"
        f"decerr{decision_label}_execerr{execution_label}{sparse_label}"
    )
    output_prefix = os.path.join("results", "execution_mismatch", f"execution_mismatch_{suffix}")
    ensure_parent_dir(output_prefix)
    return output_prefix


def print_progress(name, decision_error_std, execution_error_std, ep, episodes, success_nodes, num_nodes):
    """Print progress at 10 percent intervals."""
    interval = max(episodes // 10, 1)
    if ep % interval == 0 or ep == episodes:
        recent = np.mean(success_nodes[-interval:])
        print(
            f"  {name} decerr={decision_error_std:g} execerr={execution_error_std:g}: "
            f"[{ep:04d}/{episodes:04d}] recent success {recent:.2f}/{num_nodes}"
        )


def print_summary(rows):
    """Print a compact execution mismatch summary."""
    print("=" * 184)
    print("Execution Channel Mismatch Summary")
    print("=" * 184)
    print(
        f"{'Model':>12} {'Rho':>5} {'Delay':>5} {'DecErr':>6} {'ExecErr':>7} "
        f"{'Policy':<44} {'Success':>9} {'Perfect%':>9} "
        f"{'Slots':>7} {'Fail':>8} {'MissOpp':>8} {'Preview':>8} {'Gap':>7}"
    )
    for row in rows:
        print(
            f"{row['mismatch_model']:>12} {row['channel_rho']:>5.2f} "
            f"{row['csi_delay_slots']:>5} {row['decision_error_std']:>6.3f} "
            f"{row['execution_error_std']:>7.3f} "
            f"{row['policy']:<44} {row['success_mean']:>9.3f} "
            f"{row['perfect_rate']:>8.2f}% {row['slots_mean']:>7.3f} "
            f"{row['failed_nodes_mean']:>8.2f} {row['missed_opportunities_mean']:>8.2f} "
            f"{row['decision_preview_calls_per_slot_mean']:>8.2f} "
            f"{row['oracle_tx_gap_mean']:>7.3f}"
        )


def plot_results(rows, args, output_prefix):
    """Plot success, failed invitations, and oracle gap vs execution error."""
    policies = []
    for row in rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    scenario_keys = sorted(
        {
            (
                row["mismatch_model"],
                row["channel_rho"],
                row["csi_delay_slots"],
                row["decision_error_std"],
            )
            for row in rows
        },
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    for mismatch_model, channel_rho, csi_delay_slots, decision_error_std in scenario_keys:
        subset = [
            row
            for row in rows
            if row["mismatch_model"] == mismatch_model
            and row["channel_rho"] == channel_rho
            and row["csi_delay_slots"] == csi_delay_slots
            and row["decision_error_std"] == decision_error_std
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cmap = plt.get_cmap("tab20")
        colors = {policy: cmap(idx % 20) for idx, policy in enumerate(policies)}
        for policy in policies:
            policy_rows = sorted(
                [row for row in subset if row["policy"] == policy],
                key=lambda row: row["execution_error_std"],
            )
            if not policy_rows:
                continue
            x = [row["execution_error_std"] for row in policy_rows]
            axes[0].plot(
                x,
                [row["success_mean"] for row in policy_rows],
                marker="o",
                linewidth=1.5,
                label=policy,
                color=colors[policy],
            )
            axes[1].plot(
                x,
                [row["failed_nodes_mean"] for row in policy_rows],
                marker="o",
                linewidth=1.5,
                label=policy,
                color=colors[policy],
            )
            axes[2].plot(
                x,
                [row["oracle_tx_gap_mean"] for row in policy_rows],
                marker="o",
                linewidth=1.5,
                label=policy,
                color=colors[policy],
            )

        scenario_title = (
            f"{mismatch_model}, rho={channel_rho:g}, delay={int(csi_delay_slots)}, "
            f"decision error={decision_error_std:g}"
        )
        axes[0].set_title(f"Success, {scenario_title}")
        axes[0].set_ylabel("Successful Nodes")
        axes[0].set_ylim(0.0, args.num_nodes + 1)
        axes[1].set_title("Failed Invited Nodes")
        axes[1].set_ylabel("Failed Nodes / Episode")
        axes[2].set_title("Per-Slot Execution Oracle Gap")
        axes[2].set_ylabel("Missed Tx Count")
        for ax in axes:
            ax.set_xlabel("Execution Channel Error Std")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(fontsize=7)
        fig.tight_layout()
        decision_suffix = format_float_for_suffix(decision_error_std)
        rho_suffix = format_float_for_suffix(channel_rho)
        path = (
            f"{output_prefix}_{mismatch_model}_rho{rho_suffix}_"
            f"delay{int(csi_delay_slots)}_decerr{decision_suffix}.png"
        )
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved: {path}")
