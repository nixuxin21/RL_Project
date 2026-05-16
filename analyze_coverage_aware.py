"""
Generate method-focused analysis for Coverage-Aware Sparse-TopK.

The script reads existing CSV results only. It does not rerun simulations.
Outputs:
- per-scenario coverage ablation CSV with deltas vs Sparse-TopK
- coverage-weight ablation plot
- markdown method analysis note
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULTS_DIR = os.path.join("results", "execution_mismatch")
DEFAULT_CSV_OUTPUT = os.path.join(
    DEFAULT_RESULTS_DIR,
    "coverage_aware_ablation_analysis.csv",
)
DEFAULT_WEIGHT_PLOT = os.path.join(
    DEFAULT_RESULTS_DIR,
    "coverage_aware_weight_ablation.png",
)
DEFAULT_POWER_PLOT = os.path.join(
    DEFAULT_RESULTS_DIR,
    "coverage_aware_power_ablation.png",
)
DEFAULT_MD_OUTPUT = os.path.join("docs", "COVERAGE_AWARE_ANALYSIS.md")

MAIN_COVERAGE_WEIGHT = 0.5
MAIN_POWER_WEIGHT = 0.0

SPARSE_FRONTIER_FILE = (
    "sparse_topk_frontier_ep300_runs3_"
    "rho0p7-0p9-0p98_delay1-2-3_b4-8_sm2-3_tf0p75.csv"
)
COVERAGE_ABLATION_FILE = (
    "coverage_sparse_topk_ablation_ep300_runs3_"
    "rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0-0p25-0p5-1-2_cpw0.csv"
)
POWER_ABLATION_FILE = (
    "coverage_sparse_power_ablation_ep300_runs3_"
    "rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_"
    "cw0p5_cpw0-0p02-0p05-0p1-0p2.csv"
)
SPARSE_POLICY = "Sparse-TopK Feedback Grid B=4 sm=3 tf=0.75"
COVERAGE_POLICY_PREFIX = "Coverage-Aware Sparse-TopK Feedback Grid B=4 sm=3 tf=0.75"
BUDGET_SPLIT_SPECS = [
    (
        "B=3 sm=4.1",
        "coverage_budget_split_selected_ep300_runs3_"
        "rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0.csv",
        "Coverage-Aware Sparse-TopK Feedback Grid B=3 sm=4.1 "
        "tf=0.75 cw=0.5 cpw=0",
    ),
    (
        "B=5 sm=2.2",
        "coverage_budget_split_selected_ep300_runs3_"
        "rho0p7-0p9-0p98_delay1-2-3_b5_sm2p2_tf0p75_cw0p5_cpw0.csv",
        "Coverage-Aware Sparse-TopK Feedback Grid B=5 sm=2.2 "
        "tf=0.75 cw=0.5 cpw=0",
    ),
    (
        "B=6 sm=1.6",
        "coverage_budget_split_selected_ep300_runs3_"
        "rho0p7-0p9-0p98_delay1-2-3_b6_sm1p6_tf0p75_cw0p5_cpw0.csv",
        "Coverage-Aware Sparse-TopK Feedback Grid B=6 sm=1.6 "
        "tf=0.75 cw=0.5 cpw=0",
    ),
    (
        "B=8 sm=1",
        "coverage_budget_split_selected_ep300_runs3_"
        "rho0p7-0p9-0p98_delay1-2-3_b8_sm1_tf0p75_cw0p5_cpw0.csv",
        "Coverage-Aware Sparse-TopK Feedback Grid B=8 sm=1 tf=0.75 "
        "cw=0.5 cpw=0",
    ),
]
NEIGHBOR_COVERAGE_SPECS = [
    (
        "nr=1 nc=1",
        "neighbor_coverage_pilot_ep100_runs2_"
        "rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_nr1_nc1.csv",
        "Neighbor-Coverage Sparse-TopK Feedback Grid B=3 sm=4.1 "
        "tf=0.75 cw=0.5 cpw=0 nr=1 nc=1",
    ),
    (
        "nr=1 nc=2",
        "neighbor_coverage_pilot_ep100_runs2_"
        "rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_nr1_nc2.csv",
        "Neighbor-Coverage Sparse-TopK Feedback Grid B=3 sm=4.1 "
        "tf=0.75 cw=0.5 cpw=0 nr=1 nc=2",
    ),
    (
        "nr=1 nc=3",
        "neighbor_coverage_pilot_ep100_runs2_"
        "rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_nr1_nc3.csv",
        "Neighbor-Coverage Sparse-TopK Feedback Grid B=3 sm=4.1 "
        "tf=0.75 cw=0.5 cpw=0 nr=1 nc=3",
    ),
    (
        "nr=2 nc=3",
        "neighbor_coverage_pilot_ep100_runs2_"
        "rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_nr2_nc3.csv",
        "Neighbor-Coverage Sparse-TopK Feedback Grid B=3 sm=4.1 "
        "tf=0.75 cw=0.5 cpw=0 nr=2 nc=3",
    ),
]

NUMERIC_FIELDS = [
    "success_mean",
    "perfect_rate",
    "slots_mean",
    "failed_nodes_mean",
    "missed_opportunities_mean",
    "true_opportunities_mean",
    "decision_preview_calls_per_slot_mean",
    "oracle_tx_gap_mean",
    "coverage_sparse_selected_marginal_fraction_mean",
    "coverage_sparse_selected_overlap_fraction_mean",
]

OUTPUT_FIELDS = [
    "label",
    "coverage_sparse_weight",
    "coverage_sparse_power_weight",
    "scenario_key",
    "channel_rho",
    "csi_delay_slots",
    "episodes",
    "num_seeds",
    "success_mean",
    "perfect_rate",
    "slots_mean",
    "failed_nodes_mean",
    "missed_opportunities_mean",
    "true_opportunities_mean",
    "decision_preview_calls_per_slot_mean",
    "oracle_tx_gap_mean",
    "coverage_sparse_selected_marginal_fraction_mean",
    "coverage_sparse_selected_overlap_fraction_mean",
    "slots_delta_vs_sparse",
    "failed_delta_vs_sparse",
    "missed_delta_vs_sparse",
    "gap_delta_vs_sparse",
]


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Coverage-Aware Sparse-TopK weight ablation."
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--weight-plot", default=DEFAULT_WEIGHT_PLOT)
    parser.add_argument("--power-plot", default=DEFAULT_POWER_PLOT)
    parser.add_argument("--md-output", default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def ensure_parent_dir(path):
    """Create the parent directory for a file path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def read_csv(path):
    """Read a CSV file into dict rows."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input CSV: {path}")
    with open(path, newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def to_float(row, field, default=0.0):
    """Parse a floating-point CSV field."""
    value = row.get(field, "")
    if value == "":
        return float(default)
    return float(value)


def to_int(row, field, default=0):
    """Parse an integer-like CSV field."""
    value = row.get(field, "")
    if value == "":
        return int(default)
    return int(float(value))


def format_float(value, digits=3):
    """Format a float for markdown tables."""
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def format_weight(value):
    """Format a coverage weight compactly."""
    return f"{float(value):g}"


def scenario_key(row):
    """Return a stable rho/delay scenario key."""
    rho = to_float(row, "channel_rho")
    delay = to_int(row, "csi_delay_slots")
    return f"rho={rho:g}, delay={delay}"


def mean(rows, field):
    """Return the mean of a numeric field."""
    values = [float(row[field]) for row in rows]
    if not values:
        raise ValueError(f"No values for {field}")
    return sum(values) / len(values)


def mean_optional(rows, field, default=0.0):
    """Return the mean of an optional numeric field."""
    values = [
        to_float(row, field, default)
        for row in rows
        if row.get(field, "") != ""
    ]
    if not values:
        return float(default)
    return sum(values) / len(values)


def markdown_table(headers, rows):
    """Render a simple markdown table."""
    output = []
    output.append("| " + " | ".join(headers) + " |")
    output.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        output.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(output)


def load_analysis_rows(results_dir):
    """Load coverage ablation rows and attach per-scenario sparse deltas."""
    sparse_path = os.path.join(results_dir, SPARSE_FRONTIER_FILE)
    coverage_path = os.path.join(results_dir, COVERAGE_ABLATION_FILE)
    sparse_source = [
        row for row in read_csv(sparse_path) if row.get("policy") == SPARSE_POLICY
    ]
    coverage_source = [
        row
        for row in read_csv(coverage_path)
        if row.get("policy", "").startswith(COVERAGE_POLICY_PREFIX)
    ]
    if not sparse_source:
        raise ValueError(f"No Sparse-TopK rows found in {sparse_path}")
    if not coverage_source:
        raise ValueError(f"No Coverage-Aware rows found in {coverage_path}")

    sparse_by_scenario = {scenario_key(row): row for row in sparse_source}
    analysis_rows = []
    for row in coverage_source:
        key = scenario_key(row)
        sparse = sparse_by_scenario.get(key)
        if sparse is None:
            raise ValueError(f"Missing Sparse-TopK baseline for {key}")
        weight = to_float(row, "coverage_sparse_weight")
        power_weight = to_float(row, "coverage_sparse_power_weight")
        output = {
            "label": (
                f"Coverage-Aware cw={format_weight(weight)} "
                f"cpw={format_weight(power_weight)}"
            ),
            "coverage_sparse_weight": weight,
            "coverage_sparse_power_weight": power_weight,
            "scenario_key": key,
            "channel_rho": to_float(row, "channel_rho"),
            "csi_delay_slots": to_int(row, "csi_delay_slots"),
            "episodes": to_int(row, "episodes"),
            "num_seeds": to_int(row, "num_seeds"),
        }
        for field in NUMERIC_FIELDS:
            output[field] = to_float(row, field)
        output["slots_delta_vs_sparse"] = (
            output["slots_mean"] - to_float(sparse, "slots_mean")
        )
        output["failed_delta_vs_sparse"] = (
            output["failed_nodes_mean"] - to_float(sparse, "failed_nodes_mean")
        )
        output["missed_delta_vs_sparse"] = (
            output["missed_opportunities_mean"]
            - to_float(sparse, "missed_opportunities_mean")
        )
        output["gap_delta_vs_sparse"] = (
            output["oracle_tx_gap_mean"] - to_float(sparse, "oracle_tx_gap_mean")
        )
        analysis_rows.append(output)
    return sparse_source, analysis_rows


def aggregate_sparse(sparse_rows):
    """Aggregate the Sparse-TopK baseline across scenarios."""
    first = sparse_rows[0]
    return {
        "label": "Sparse-TopK B=4 sm=3",
        "coverage_sparse_weight": "",
        "episodes": to_int(first, "episodes"),
        "num_seeds": to_int(first, "num_seeds"),
        "scenario_count": len(sparse_rows),
        "success_mean": mean(sparse_rows, "success_mean"),
        "perfect_rate": mean(sparse_rows, "perfect_rate"),
        "slots_mean": mean(sparse_rows, "slots_mean"),
        "failed_nodes_mean": mean(sparse_rows, "failed_nodes_mean"),
        "missed_opportunities_mean": mean(sparse_rows, "missed_opportunities_mean"),
        "true_opportunities_mean": mean(sparse_rows, "true_opportunities_mean"),
        "decision_preview_calls_per_slot_mean": mean(
            sparse_rows,
            "decision_preview_calls_per_slot_mean",
        ),
        "oracle_tx_gap_mean": mean(sparse_rows, "oracle_tx_gap_mean"),
        "coverage_sparse_selected_marginal_fraction_mean": "",
        "coverage_sparse_selected_overlap_fraction_mean": "",
        "slots_delta_vs_sparse": 0.0,
        "failed_delta_vs_sparse": 0.0,
        "missed_delta_vs_sparse": 0.0,
        "gap_delta_vs_sparse": 0.0,
    }


def aggregate_coverage(analysis_rows):
    """Aggregate coverage-aware rows by coverage weight."""
    groups = {}
    for row in analysis_rows:
        groups.setdefault(float(row["coverage_sparse_weight"]), []).append(row)

    summaries = []
    for weight in sorted(groups):
        group = groups[weight]
        first = group[0]
        item = {
            "label": (
                f"Coverage-Aware cw={format_weight(weight)} "
                f"cpw={format_weight(mean(group, 'coverage_sparse_power_weight'))}"
            ),
            "coverage_sparse_weight": weight,
            "coverage_sparse_power_weight": mean(group, "coverage_sparse_power_weight"),
            "episodes": int(first["episodes"]),
            "num_seeds": int(first["num_seeds"]),
            "scenario_count": len(group),
        }
        for field in NUMERIC_FIELDS:
            item[field] = mean(group, field)
        for field in [
            "slots_delta_vs_sparse",
            "failed_delta_vs_sparse",
            "missed_delta_vs_sparse",
            "gap_delta_vs_sparse",
        ]:
            item[field] = mean(group, field)
        summaries.append(item)
    return summaries


def load_power_summaries(results_dir):
    """Load and aggregate coverage sparse power penalty ablation rows."""
    power_path = os.path.join(results_dir, POWER_ABLATION_FILE)
    if not os.path.exists(power_path):
        return []

    power_source = [
        row
        for row in read_csv(power_path)
        if row.get("policy", "").startswith(COVERAGE_POLICY_PREFIX)
    ]
    groups = {}
    for row in power_source:
        groups.setdefault(to_float(row, "coverage_sparse_power_weight"), []).append(row)

    summaries = []
    for power_weight in sorted(groups):
        group = groups[power_weight]
        first = group[0]
        item = {
            "label": (
                f"Coverage-Aware cw={format_weight(mean(group, 'coverage_sparse_weight'))} "
                f"cpw={format_weight(power_weight)}"
            ),
            "coverage_sparse_weight": mean(group, "coverage_sparse_weight"),
            "coverage_sparse_power_weight": power_weight,
            "episodes": to_int(first, "episodes"),
            "num_seeds": to_int(first, "num_seeds"),
            "scenario_count": len(group),
        }
        for field in NUMERIC_FIELDS:
            item[field] = mean(group, field)
        summaries.append(item)
    return summaries


def load_budget_split_summaries(results_dir):
    """Load available formal near-preview-16 budget split results."""
    summaries = []
    for label, file_name, policy in BUDGET_SPLIT_SPECS:
        budget_path = os.path.join(results_dir, file_name)
        if not os.path.exists(budget_path):
            continue

        rows = [
            row
            for row in read_csv(budget_path)
            if row.get("policy", "") == policy
        ]
        if not rows:
            continue

        first = rows[0]
        item = {
            "label": label,
            "coverage_sparse_weight": MAIN_COVERAGE_WEIGHT,
            "coverage_sparse_power_weight": MAIN_POWER_WEIGHT,
            "episodes": to_int(first, "episodes"),
            "num_seeds": to_int(first, "num_seeds"),
            "scenario_count": len(rows),
        }
        for field in NUMERIC_FIELDS:
            item[field] = mean(rows, field)
        summaries.append(item)
    return summaries


def load_neighbor_coverage_summaries(results_dir):
    """Load optional Neighbor-Coverage diagnostic pilot summaries."""
    summaries = []
    for label, file_name, policy in NEIGHBOR_COVERAGE_SPECS:
        neighbor_path = os.path.join(results_dir, file_name)
        if not os.path.exists(neighbor_path):
            continue

        rows = [
            row
            for row in read_csv(neighbor_path)
            if row.get("policy", "") == policy
        ]
        if not rows:
            continue

        first = rows[0]
        item = {
            "label": label,
            "coverage_sparse_weight": MAIN_COVERAGE_WEIGHT,
            "coverage_sparse_power_weight": MAIN_POWER_WEIGHT,
            "episodes": to_int(first, "episodes"),
            "num_seeds": to_int(first, "num_seeds"),
            "scenario_count": len(rows),
        }
        for field in NUMERIC_FIELDS:
            item[field] = mean(rows, field)
        item["adaptive_sparse_v3_neighbor_extra_preview_mean"] = mean_optional(
            rows,
            "adaptive_sparse_v3_neighbor_extra_preview_mean",
        )
        item["adaptive_sparse_v3_selected_extra_preview_mean"] = mean_optional(
            rows,
            "adaptive_sparse_v3_selected_extra_preview_mean",
        )
        summaries.append(item)
    return summaries


def write_analysis_csv(path, rows):
    """Write per-scenario coverage ablation analysis rows."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_weight_ablation(summaries, path):
    """Plot coverage-weight effects on the key tradeoff metrics."""
    ensure_parent_dir(path)
    weights = [float(item["coverage_sparse_weight"]) for item in summaries]
    if not weights:
        raise ValueError("No coverage summaries available to plot")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    plot_specs = [
        ("oracle_tx_gap_mean", "Oracle gap", "#1f77b4"),
        ("missed_opportunities_mean", "Missed opportunities", "#ff7f0e"),
        ("failed_nodes_mean", "Failed invitations", "#2ca02c"),
    ]
    for ax, (field, title, color) in zip(axes, plot_specs):
        values = [float(item[field]) for item in summaries]
        ax.plot(weights, values, marker="o", color=color)
        ax.set_title(title)
        ax.set_xlabel("coverage weight")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("equal-scenario mean")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_power_ablation(summaries, path):
    """Plot power penalty effects on key tradeoff metrics."""
    if not summaries:
        return
    ensure_parent_dir(path)
    powers = [float(item["coverage_sparse_power_weight"]) for item in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    plot_specs = [
        ("oracle_tx_gap_mean", "Oracle gap", "#1f77b4"),
        ("missed_opportunities_mean", "Missed opportunities", "#ff7f0e"),
        ("failed_nodes_mean", "Failed invitations", "#2ca02c"),
    ]
    for ax, (field, title, color) in zip(axes, plot_specs):
        values = [float(item[field]) for item in summaries]
        ax.plot(powers, values, marker="o", color=color)
        ax.set_title(title)
        ax.set_xlabel("power penalty weight")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("equal-scenario mean")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(
    path,
    csv_output,
    weight_plot,
    power_plot,
    sparse_summary,
    coverage_summaries,
    power_summaries,
    budget_split_summaries,
    neighbor_coverage_summaries,
):
    """Write Coverage-Aware method analysis markdown."""
    ensure_parent_dir(path)
    best_gap = min(
        coverage_summaries,
        key=lambda row: (
            row["oracle_tx_gap_mean"],
            row["missed_opportunities_mean"],
            row["failed_nodes_mean"],
        ),
    )
    main_setting = next(
        (
            item
            for item in coverage_summaries
            if abs(float(item["coverage_sparse_weight"]) - MAIN_COVERAGE_WEIGHT) < 1e-12
            and abs(float(item["coverage_sparse_power_weight"]) - MAIN_POWER_WEIGHT) < 1e-12
        ),
        best_gap,
    )
    best_power = None
    if power_summaries:
        best_power = min(
            power_summaries,
            key=lambda row: (
                row["oracle_tx_gap_mean"],
                row["missed_opportunities_mean"],
                row["slots_mean"],
                row["coverage_sparse_power_weight"],
            ),
        )

    overall_rows = []
    for item in [sparse_summary] + coverage_summaries:
        overall_rows.append(
            [
                item["label"],
                item["scenario_count"],
                item["episodes"],
                item["num_seeds"],
                format_float(item["slots_mean"]),
                format_float(item["perfect_rate"], 2),
                format_float(item["failed_nodes_mean"]),
                format_float(item["missed_opportunities_mean"]),
                format_float(item["decision_preview_calls_per_slot_mean"], 2),
                format_float(item["oracle_tx_gap_mean"]),
                format_float(item["coverage_sparse_selected_marginal_fraction_mean"]),
                format_float(item["coverage_sparse_selected_overlap_fraction_mean"]),
            ]
        )

    delta_rows = []
    for item in coverage_summaries:
        delta_rows.append(
            [
                format_weight(item["coverage_sparse_weight"]),
                format_float(item["slots_delta_vs_sparse"]),
                format_float(item["failed_delta_vs_sparse"]),
                format_float(item["missed_delta_vs_sparse"]),
                format_float(item["gap_delta_vs_sparse"]),
                format_float(item["coverage_sparse_selected_marginal_fraction_mean"]),
                format_float(item["coverage_sparse_selected_overlap_fraction_mean"]),
            ]
        )

    power_rows = []
    for item in power_summaries:
        power_rows.append(
            [
                format_weight(item["coverage_sparse_power_weight"]),
                format_float(item["slots_mean"]),
                format_float(item["perfect_rate"], 2),
                format_float(item["failed_nodes_mean"]),
                format_float(item["missed_opportunities_mean"]),
                format_float(item["decision_preview_calls_per_slot_mean"], 2),
                format_float(item["oracle_tx_gap_mean"]),
                format_float(item["coverage_sparse_selected_marginal_fraction_mean"]),
                format_float(item["coverage_sparse_selected_overlap_fraction_mean"]),
            ]
        )

    power_section = ""
    if power_rows and best_power is not None:
        power_section = f"""
## Power Penalty Ablation

This sweep fixes `cw={format_weight(MAIN_COVERAGE_WEIGHT)}` and varies the stale power penalty `cpw`. Lower gap and lower missed opportunities are better.

{markdown_table(
            [
                "cpw",
                "Slots",
                "Perfect %",
                "Failed",
                "Missed",
                "Preview",
                "Gap",
                "Marginal",
                "Overlap",
            ],
            power_rows,
        )}

The selected main setting uses `cpw={format_weight(MAIN_POWER_WEIGHT)}`. The formal sweep shows that `cpw=0` and `cpw=0.02` are tied on the main metrics, while larger power penalties reduce failed invitations slightly but increase missed opportunities, slots, and oracle gap. This is why the current mainline removes the stale power penalty.

The power-ablation figure is `{power_plot}`.
"""

    budget_split_section = ""
    selected_budget_split = None
    if budget_split_summaries:
        budget_split_items = [main_setting] + budget_split_summaries
        best_budget_split = min(
            budget_split_items,
            key=lambda row: (
                row["oracle_tx_gap_mean"],
                row["missed_opportunities_mean"],
                row["slots_mean"],
            ),
        )
        selected_budget_split = best_budget_split
        budget_split_rows = []
        for item in budget_split_items:
            setting = "B=4 sm=3" if item is main_setting else item["label"]
            budget_split_rows.append(
                [
                    setting,
                    format_float(item["slots_mean"]),
                    format_float(item["perfect_rate"], 2),
                    format_float(item["failed_nodes_mean"]),
                    format_float(item["missed_opportunities_mean"]),
                    format_float(
                        item["decision_preview_calls_per_slot_mean"],
                        2,
                    ),
                    format_float(item["oracle_tx_gap_mean"]),
                ]
            )
        budget_split_section = f"""
## Budget Split Selection

The weight/power ablations above use the original `B=4 sm=3` split. A follow-up budget-split sweep tests whether the same preview budget should spend more calls on stale candidate breadth or on current feedback confirmation. The settings below all stay near preview `16` by trading off final feedback probes against stale seed-pool breadth.

{markdown_table(
            [
                "Setting",
                "Slots",
                "Perfect %",
                "Failed",
                "Missed",
                "Preview",
                "Gap",
            ],
            budget_split_rows,
        )}

The selected main setting is `{best_budget_split["label"]}`. It has the lowest formal gap in the near-preview-16 budget split table, with missed opportunities `{format_float(best_budget_split["missed_opportunities_mean"])}` and failed invitations `{format_float(best_budget_split["failed_nodes_mean"])}`.
"""

    neighbor_section = ""
    if neighbor_coverage_summaries:
        current_b3 = next(
            (
                item
                for item in budget_split_summaries
                if item["label"] == "B=3 sm=4.1"
            ),
            selected_budget_split,
        )
        best_neighbor = min(
            neighbor_coverage_summaries,
            key=lambda row: (
                row["oracle_tx_gap_mean"],
                row["missed_opportunities_mean"],
                row["slots_mean"],
            ),
        )
        neighbor_rows = []
        for item in neighbor_coverage_summaries:
            neighbor_rows.append(
                [
                    item["label"],
                    format_float(item["slots_mean"]),
                    format_float(item["perfect_rate"], 2),
                    format_float(item["failed_nodes_mean"]),
                    format_float(item["missed_opportunities_mean"]),
                    format_float(item["decision_preview_calls_per_slot_mean"], 2),
                    format_float(item["oracle_tx_gap_mean"]),
                    format_float(
                        item["adaptive_sparse_v3_neighbor_extra_preview_mean"],
                    ),
                    format_float(
                        item["adaptive_sparse_v3_selected_extra_preview_mean"],
                    ),
                ]
            )
        baseline_sentence = ""
        if current_b3 is not None:
            baseline_sentence = (
                "The current selected B=3 setting remains better: "
                f"gap `{format_float(current_b3['oracle_tx_gap_mean'])}` and "
                f"slots `{format_float(current_b3['slots_mean'])}` versus the best "
                f"neighbor diagnostic gap `{format_float(best_neighbor['oracle_tx_gap_mean'])}` "
                f"and slots `{format_float(best_neighbor['slots_mean'])}`."
            )
        neighbor_section = f"""
## Neighbor-Coverage Diagnostic

This pilot keeps the same total preview budget as `B=3 sm=4.1` but reallocates part of the stale preview pool from uniform grid seeds to local neighbors around the strongest stale candidates.

{markdown_table(
            [
                "Setting",
                "Slots",
                "Perfect %",
                "Failed",
                "Missed",
                "Preview",
                "Gap",
                "Neighbor previews",
                "Selected neighbors",
            ],
            neighbor_rows,
        )}

{baseline_sentence} The fixed local-neighbor reallocation is therefore a negative diagnostic, not a new mainline method.
"""

    content = f"""# Coverage-Aware Sparse-TopK Analysis

Generated by `analyze_coverage_aware.py` from existing execution-mismatch CSV files.

This note isolates the current Coverage-Aware method family:

> Coverage-Aware Sparse-TopK Feedback keeps top stale anchors from a sparse stale preview pool, fills the remaining feedback probes by marginal device-coverage gain, and then uses current aggregate feedback to confirm the final IRS state.

The per-scenario CSV is `{csv_output}`. The weight-ablation figure is `{weight_plot}`.

## Formal Weight Ablation

All values are equal-weight averages across the 9 temporal AR(1) rho/delay scenarios. `Samples` is the aggregate episode count in the source CSV row.

{markdown_table(
        [
            "Method",
            "Scenarios",
            "Samples",
            "Seeds",
            "Slots",
            "Perfect %",
            "Failed",
            "Missed",
            "Preview",
            "Gap",
            "Marginal",
            "Overlap",
        ],
        overall_rows,
    )}

{power_section}

{budget_split_section}

{neighbor_section}

## Delta Vs Sparse-TopK

Negative deltas are improvements over `Sparse-TopK B=4 sm=3` at the same preview budget. `Marginal` and `Overlap` are stale-candidate diagnostics for the selected B-candidate set.

{markdown_table(
        [
            "cw",
            "Delta slots",
            "Delta failed",
            "Delta missed",
            "Delta gap",
            "Marginal",
            "Overlap",
        ],
        delta_rows,
    )}

## Interpretation

`Coverage-Aware cw={format_weight(main_setting["coverage_sparse_weight"])} cpw={format_weight(main_setting["coverage_sparse_power_weight"])}` is the calibrated B=4 report setting. It changes gap from `{format_float(sparse_summary["oracle_tx_gap_mean"])}` to `{format_float(main_setting["oracle_tx_gap_mean"])}`, failed invitations from `{format_float(sparse_summary["failed_nodes_mean"])}` to `{format_float(main_setting["failed_nodes_mean"])}`, and missed opportunities from `{format_float(sparse_summary["missed_opportunities_mean"])}` to `{format_float(main_setting["missed_opportunities_mean"])}` at the same `{format_float(main_setting["decision_preview_calls_per_slot_mean"], 2)}` preview calls per slot.

The best numeric gap in the weight ablation is `cw={format_weight(best_gap["coverage_sparse_weight"])}` with gap `{format_float(best_gap["oracle_tx_gap_mean"])}` and missed opportunities `{format_float(best_gap["missed_opportunities_mean"])}`. In this run, `cw=0`, `cw=0.25`, and `cw=0.5` are effectively tied on the main metrics; `cw=0.5` is retained as the report setting because it keeps the intended marginal-coverage fill interpretation without changing preview cost. Future replacements should beat the selected budget split near preview `16`, not only the original B=4 calibration point.

The mechanism is narrow and useful: the method does not change stale/limited CSI, the AirComp objective, or the current-feedback confirmation step. It only changes how non-anchor IRS candidates are generated inside the same sparse preview budget, and the selected budget split shifts more of that budget to stale candidate breadth. This is a clean candidate-generation improvement rather than a separate research branch.
"""
    with open(path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write(content)


def main():
    """Run coverage-aware analysis generation."""
    args = parse_args()
    sparse_rows, analysis_rows = load_analysis_rows(args.results_dir)
    sparse_summary = aggregate_sparse(sparse_rows)
    coverage_summaries = aggregate_coverage(analysis_rows)
    power_summaries = load_power_summaries(args.results_dir)
    budget_split_summaries = load_budget_split_summaries(args.results_dir)
    neighbor_coverage_summaries = load_neighbor_coverage_summaries(args.results_dir)
    write_analysis_csv(args.csv_output, analysis_rows)
    plot_weight_ablation(coverage_summaries, args.weight_plot)
    plot_power_ablation(power_summaries, args.power_plot)
    write_markdown(
        args.md_output,
        args.csv_output,
        args.weight_plot,
        args.power_plot,
        sparse_summary,
        coverage_summaries,
        power_summaries,
        budget_split_summaries,
        neighbor_coverage_summaries,
    )
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.weight_plot}")
    if power_summaries:
        print(f"Wrote {args.power_plot}")
    print(f"Wrote {args.md_output}")


if __name__ == "__main__":
    main()
