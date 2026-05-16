"""
Post-hoc probing cost tradeoff analysis.

The partial probing experiments report coverage, latency, and preview budget.
This script converts those results into a cost-aware utility:

    utility = success_mean - slot_cost * slots_mean - preview_cost * total_preview_calls_mean

where total_preview_calls_mean is approximated as
decision_preview_calls_per_slot_mean * slots_mean. The goal is to identify when
full Greedy's latency advantage is worth its extra preview calls, and when
lower-budget probing policies are preferable.
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

from evaluate_policy_comparison import ensure_parent_dir, format_float_for_suffix


DEFAULT_PARTIAL_SUMMARY = (
    "results/partial_probing/"
    "partial_probing_sweep_ep1000_runs5_seed2026_b1-2-4-8.csv"
)
DEFAULT_LEARNED_SUMMARY = (
    "results/learned_probing/"
    "learned_probing_train5000_eval1000_seed2026_rotatinggrid_b4_evalb1-2-4-8_eval_summary.csv"
)


NUMERIC_COLUMNS = {
    "probe_budget",
    "budget_fraction",
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
    "candidate_count_mean",
    "oracle_match_rate",
    "oracle_tx_gap_mean",
    "avg_reward",
}


def parse_float_list(value):
    """Parse a comma-separated float list such as '0,0.001,0.01'."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    """Parse cost tradeoff analysis parameters."""
    parser = argparse.ArgumentParser(
        description="Analyze probing policy utility under explicit slot and preview costs."
    )
    parser.add_argument(
        "--summary-csv",
        action="append",
        default=None,
        help=(
            "Input probing summary CSV. Can be passed multiple times. "
            "Defaults to the formal partial and learned probing summaries."
        ),
    )
    parser.add_argument(
        "--slot-cost-values",
        default="0,0.05,0.1,0.2",
        help="Comma-separated node-equivalent cost per used communication slot.",
    )
    parser.add_argument(
        "--preview-cost-values",
        default="0,0.0005,0.001,0.002,0.005,0.01,0.02,0.05",
        help="Comma-separated node-equivalent cost per preview call.",
    )
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """Validate cost lists and input CSV paths."""
    if args.summary_csv is None:
        args.summary_csv = [DEFAULT_PARTIAL_SUMMARY, DEFAULT_LEARNED_SUMMARY]
    args.slot_cost_values = parse_float_list(args.slot_cost_values)
    args.preview_cost_values = parse_float_list(args.preview_cost_values)
    if not args.slot_cost_values:
        raise ValueError("--slot-cost-values must contain at least one value")
    if not args.preview_cost_values:
        raise ValueError("--preview-cost-values must contain at least one value")
    if any(value < 0.0 for value in args.slot_cost_values):
        raise ValueError("--slot-cost-values must be non-negative")
    if any(value < 0.0 for value in args.preview_cost_values):
        raise ValueError("--preview-cost-values must be non-negative")
    for path in args.summary_csv:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Summary CSV not found: {path}")


def resolve_output_prefix(args):
    """Resolve shared output prefix for CSVs and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    slot_label = "-".join(format_float_for_suffix(value) for value in args.slot_cost_values)
    preview_label = "-".join(format_float_for_suffix(value) for value in args.preview_cost_values)
    output_prefix = os.path.join(
        "results",
        "probing_cost",
        f"probing_cost_tradeoff_slot{slot_label}_preview{preview_label}",
    )
    ensure_parent_dir(output_prefix)
    return output_prefix


def read_summary_csv(path):
    """Read and type-convert one probing summary CSV."""
    rows = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for raw in reader:
            row = dict(raw)
            for key in NUMERIC_COLUMNS:
                if key in row and row[key] != "":
                    value = float(row[key])
                    row[key] = int(value) if key in {"probe_budget", "episodes", "num_seeds"} else value
            row["source_csv"] = path
            rows.append(row)
    return rows


def merge_candidate_rows(paths):
    """
    Merge candidate rows from multiple summaries.

    Duplicate policy-budget rows are kept from the first CSV. Learned Probe rows
    are added from the learned summary while Random/Rotating/Greedy duplicates
    remain from the partial probing summary.
    """
    merged = {}
    for path in paths:
        for row in read_summary_csv(path):
            key = (row["policy"], int(row["probe_budget"]))
            if key not in merged:
                merged[key] = row
    rows = list(merged.values())
    rows.sort(key=lambda row: (int(row["probe_budget"]), row["policy"]))
    return rows


def enrich_candidate_row(row):
    """Add cost-relevant derived metrics to one candidate row."""
    enriched = dict(row)
    slots = float(enriched["slots_mean"])
    preview_per_slot = float(enriched["decision_preview_calls_per_slot_mean"])
    total_preview = preview_per_slot * slots
    success_rate = float(enriched.get("success_rate_mean", 0.0))
    num_nodes = float(enriched["success_mean"]) / success_rate if success_rate > 0.0 else 50.0
    enriched["total_preview_calls_mean"] = total_preview
    enriched["success_shortfall_mean"] = num_nodes - float(enriched["success_mean"])
    return enriched


def is_dominated(row, others):
    """
    Return True if another candidate is no worse on all main dimensions and
    strictly better on at least one.
    """
    for other in others:
        if other is row:
            continue
        no_worse = (
            float(other["success_mean"]) >= float(row["success_mean"])
            and float(other["perfect_rate"]) >= float(row["perfect_rate"])
            and float(other["slots_mean"]) <= float(row["slots_mean"])
            and float(other["total_preview_calls_mean"]) <= float(row["total_preview_calls_mean"])
        )
        strictly_better = (
            float(other["success_mean"]) > float(row["success_mean"])
            or float(other["perfect_rate"]) > float(row["perfect_rate"])
            or float(other["slots_mean"]) < float(row["slots_mean"])
            or float(other["total_preview_calls_mean"]) < float(row["total_preview_calls_mean"])
        )
        if no_worse and strictly_better:
            return True
    return False


def utility(row, slot_cost, preview_cost):
    """Compute cost-aware node-equivalent utility for one candidate row."""
    return (
        float(row["success_mean"])
        - slot_cost * float(row["slots_mean"])
        - preview_cost * float(row["total_preview_calls_mean"])
    )


def candidate_label(row):
    """Compact policy-budget label for tables and plots."""
    return f"{row['policy']} B={int(row['probe_budget'])}"


def rank_key(row):
    """Tie-break utility winners by coverage, then latency, then preview calls."""
    return (
        float(row["utility"]),
        float(row["perfect_rate"]),
        -float(row["slots_mean"]),
        -float(row["total_preview_calls_mean"]),
    )


def build_tradeoff_rows(candidates, slot_cost_values, preview_cost_values):
    """Build all candidate-cost utility rows and one winner row per cost pair."""
    utility_rows = []
    winner_rows = []
    for slot_cost in slot_cost_values:
        for preview_cost in preview_cost_values:
            cost_rows = []
            for candidate in candidates:
                row = dict(candidate)
                row["slot_cost"] = slot_cost
                row["preview_cost"] = preview_cost
                row["utility"] = utility(row, slot_cost, preview_cost)
                row["label"] = candidate_label(row)
                cost_rows.append(row)
            best = max(cost_rows, key=rank_key)
            for row in cost_rows:
                row["is_winner"] = int(row["label"] == best["label"])
            utility_rows.extend(cost_rows)
            winner_rows.append({key: best[key] for key in winner_fieldnames()})
    return utility_rows, winner_rows


def candidate_fieldnames():
    """Candidate CSV field order."""
    return [
        "probe_budget",
        "policy",
        "success_mean",
        "perfect_rate",
        "slots_mean",
        "decision_preview_calls_per_slot_mean",
        "total_preview_calls_mean",
        "total_energy_mean",
        "oracle_tx_gap_mean",
        "success_shortfall_mean",
        "is_pareto",
        "source_csv",
    ]


def utility_fieldnames():
    """Utility CSV field order."""
    return [
        "slot_cost",
        "preview_cost",
        "label",
        "probe_budget",
        "policy",
        "utility",
        "is_winner",
        "success_mean",
        "perfect_rate",
        "slots_mean",
        "total_preview_calls_mean",
        "total_energy_mean",
        "oracle_tx_gap_mean",
    ]


def winner_fieldnames():
    """Winner CSV field order."""
    return [
        "slot_cost",
        "preview_cost",
        "label",
        "probe_budget",
        "policy",
        "utility",
        "success_mean",
        "perfect_rate",
        "slots_mean",
        "total_preview_calls_mean",
        "total_energy_mean",
        "oracle_tx_gap_mean",
    ]


def write_csv(path, rows, fieldnames):
    """Write rows to CSV."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def print_winners(winner_rows):
    """Print a compact winner table."""
    print("=" * 132)
    print("Probing Cost Tradeoff Winners")
    print("=" * 132)
    print(
        f"{'SlotCost':>8} {'PreviewCost':>11} {'Winner':<34} "
        f"{'Utility':>9} {'Success':>9} {'Perfect%':>9} {'Slots':>8} {'Previews':>9}"
    )
    for row in winner_rows:
        print(
            f"{float(row['slot_cost']):>8.4f} {float(row['preview_cost']):>11.4f} "
            f"{row['label']:<34} {float(row['utility']):>9.3f} "
            f"{float(row['success_mean']):>9.3f} {float(row['perfect_rate']):>8.2f}% "
            f"{float(row['slots_mean']):>8.3f} {float(row['total_preview_calls_mean']):>9.3f}"
        )


def plot_frontier(candidates, output_prefix):
    """Plot latency vs total preview calls with candidate labels."""
    fig, ax = plt.subplots(figsize=(11, 7))
    pareto = [row for row in candidates if row["is_pareto"]]
    dominated = [row for row in candidates if not row["is_pareto"]]

    for rows, marker, alpha, label in (
        (dominated, "o", 0.35, "Dominated"),
        (pareto, "D", 0.9, "Pareto"),
    ):
        if not rows:
            continue
        ax.scatter(
            [row["total_preview_calls_mean"] for row in rows],
            [row["slots_mean"] for row in rows],
            s=[max(30.0, row["perfect_rate"] * 1.2) for row in rows],
            marker=marker,
            alpha=alpha,
            label=label,
        )

    for row in pareto:
        ax.annotate(
            candidate_label(row),
            (row["total_preview_calls_mean"], row["slots_mean"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )

    ax.set_title("Coverage-Latency-Preview Frontier")
    ax.set_xlabel("Total Preview Calls per Episode (mean)")
    ax.set_ylabel("Slots Used (mean)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path = f"{output_prefix}_frontier.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def short_label(label):
    """Shorten policy labels for heatmap cells."""
    replacements = {
        "Rotating Grid Probe": "RotGrid",
        "Random Probe": "Random",
        "Greedy Full Preview": "Greedy",
        "Learned Probe": "Learned",
        "Hybrid Local+Grid Probe": "Hybrid",
        "Fixed Grid Probe": "Fixed",
        "Local Probe": "Local",
    }
    output = label
    for source, target in replacements.items():
        output = output.replace(source, target)
    return output


def plot_winner_heatmap(winner_rows, slot_cost_values, preview_cost_values, output_prefix):
    """Plot categorical winner heatmap over slot/preview costs."""
    labels = sorted({row["label"] for row in winner_rows})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(slot_cost_values), len(preview_cost_values)), dtype=int)
    text = [["" for _ in preview_cost_values] for _ in slot_cost_values]

    for row in winner_rows:
        y = slot_cost_values.index(float(row["slot_cost"]))
        x = preview_cost_values.index(float(row["preview_cost"]))
        matrix[y, x] = label_to_id[row["label"]]
        text[y][x] = short_label(row["label"])

    fig, ax = plt.subplots(figsize=(max(9, len(preview_cost_values) * 1.3), max(4, len(slot_cost_values) * 1.0)))
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(len(labels) - 1, 1))

    ax.set_xticks(np.arange(len(preview_cost_values)))
    ax.set_yticks(np.arange(len(slot_cost_values)))
    ax.set_xticklabels([f"{value:g}" for value in preview_cost_values], rotation=45, ha="right")
    ax.set_yticklabels([f"{value:g}" for value in slot_cost_values])
    ax.set_xlabel("Preview Cost")
    ax.set_ylabel("Slot Cost")
    ax.set_title("Best Policy by Cost-Aware Utility")

    for y in range(len(slot_cost_values)):
        for x in range(len(preview_cost_values)):
            ax.text(x, y, text[y][x], ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    path = f"{output_prefix}_winners.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    """Run probing cost tradeoff analysis."""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)

    candidates = [enrich_candidate_row(row) for row in merge_candidate_rows(args.summary_csv)]
    for row in candidates:
        row["is_pareto"] = int(not is_dominated(row, candidates))

    utility_rows, winner_rows = build_tradeoff_rows(
        candidates,
        args.slot_cost_values,
        args.preview_cost_values,
    )

    print_winners(winner_rows)
    write_csv(f"{output_prefix}_candidates.csv", candidates, candidate_fieldnames())
    write_csv(f"{output_prefix}_utilities.csv", utility_rows, utility_fieldnames())
    write_csv(f"{output_prefix}_winners.csv", winner_rows, winner_fieldnames())
    if not args.no_plots:
        plot_frontier(candidates, output_prefix)
        plot_winner_heatmap(
            winner_rows,
            args.slot_cost_values,
            args.preview_cost_values,
            output_prefix,
        )


if __name__ == "__main__":
    main()
