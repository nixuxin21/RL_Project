"""
Generate paper-ready analysis artifacts for the execution-mismatch frontier.

The script reads existing CSV results only. It does not rerun simulations.
Outputs:
- per-scenario analysis CSV
- cost-vs-gap plot
- failed/missed tradeoff plot
- markdown analysis note
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
    "main_frontier_analysis.csv",
)
DEFAULT_PREVIEW_GAP_PLOT = os.path.join(
    DEFAULT_RESULTS_DIR,
    "main_frontier_preview_gap.png",
)
DEFAULT_FAILED_MISSED_PLOT = os.path.join(
    DEFAULT_RESULTS_DIR,
    "main_frontier_failed_missed.png",
)
DEFAULT_MD_OUTPUT = os.path.join("docs", "MAIN_RESULTS_ANALYSIS.md")

FRONTIER_FILE = (
    "sparse_topk_frontier_ep300_runs3_"
    "rho0p7-0p9-0p98_delay1-2-3_b4-8_sm2-3_tf0p75.csv"
)
ADAPTIVE_V2_FILE = (
    "adaptive_sparse_topk_v2_pilot_ep100_runs2_"
    "rho0p7-0p9-0p98_delay1-2-3_b4_mt0p02-0p05_pc0-0p002-0p005-0p01.csv"
)
COVERAGE_SPARSE_FILE = (
    "coverage_sparse_topk_frontier_ep300_runs3_"
    "rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0p5_cpw0.csv"
)
COVERAGE_BUDGET_SPLIT_FILE = (
    "coverage_budget_split_selected_ep300_runs3_"
    "rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0.csv"
)
INVITATION_MASK_CORRECTION_FILE = (
    "invitation_mask_correction_formal_ep300_runs3_"
    "rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1.csv"
)

METHOD_SPECS = [
    {
        "label": "Rotating B=4",
        "short_label": "Rot B4",
        "role": "low-budget reference",
        "source_file": FRONTIER_FILE,
        "policy": "Estimated Rotating Grid B=4",
    },
    {
        "label": "Rotating B=8",
        "short_label": "Rot B8",
        "role": "low-cost deployment baseline",
        "source_file": FRONTIER_FILE,
        "policy": "Estimated Rotating Grid B=8",
    },
    {
        "label": "Adaptive V2 pc=0.005",
        "short_label": "Adap pc=.005",
        "role": "adaptive continuum point",
        "source_file": ADAPTIVE_V2_FILE,
        "policy": (
            "Adaptive Sparse-TopK V2 Feedback Grid B=4 bm=2 em=3 "
            "mt=0.05 pc=0.005 tf=0.75"
        ),
    },
    {
        "label": "Adaptive V2 pc=0.002",
        "short_label": "Adap pc=.002",
        "role": "adaptive high-quality point",
        "source_file": ADAPTIVE_V2_FILE,
        "policy": (
            "Adaptive Sparse-TopK V2 Feedback Grid B=4 bm=2 em=3 "
            "mt=0.05 pc=0.002 tf=0.75"
        ),
    },
    {
        "label": "Sparse-TopK B=4 sm=3",
        "short_label": "Sparse sm3",
        "role": "reportable medium-cost baseline",
        "source_file": FRONTIER_FILE,
        "policy": "Sparse-TopK Feedback Grid B=4 sm=3 tf=0.75",
    },
    {
        "label": "Coverage-Aware B=4 cw=0.5 cpw=0",
        "short_label": "Cov B4",
        "role": "same-cost B=4 coverage reference",
        "source_file": COVERAGE_SPARSE_FILE,
        "policy": "Coverage-Aware Sparse-TopK Feedback Grid B=4 sm=3 tf=0.75 cw=0.5 cpw=0",
    },
    {
        "label": "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0",
        "short_label": "Cov B3",
        "role": "current budget-split refinement",
        "source_file": COVERAGE_BUDGET_SPLIT_FILE,
        "policy": "Coverage-Aware Sparse-TopK Feedback Grid B=3 sm=4.1 tf=0.75 cw=0.5 cpw=0",
    },
    {
        "label": "Mask-Corrected Coverage-Aware B=3 mc=1",
        "short_label": "MaskCorr",
        "role": "current best same-preview method",
        "source_file": INVITATION_MASK_CORRECTION_FILE,
        "policy": "Mask-Corrected Coverage-Aware B=3 mc=1",
    },
    {
        "label": "Stale-TopK B=4",
        "short_label": "Stale",
        "role": "high-cost positive reference",
        "source_file": FRONTIER_FILE,
        "policy": "Stale-TopK Feedback Grid B=4",
    },
    {
        "label": "Temporal Deviation Oracle B=4",
        "short_label": "TD Oracle",
        "role": "hidden-info upper bound",
        "source_file": FRONTIER_FILE,
        "policy": "Temporal Deviation Oracle B=4",
    },
]

NUMERIC_FIELDS = [
    "success_mean",
    "perfect_rate",
    "slots_mean",
    "failed_nodes_mean",
    "missed_opportunities_mean",
    "true_opportunities_mean",
    "failure_slot_rate",
    "decision_preview_calls_per_slot_mean",
    "oracle_tx_gap_mean",
    "scheduled_nodes_mean",
    "adaptive_sparse_expansion_rate",
]

OUTPUT_FIELDS = [
    "label",
    "role",
    "source_file",
    "policy",
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
    "failed_plus_missed_mean",
    "true_opportunities_mean",
    "failure_slot_rate",
    "decision_preview_calls_per_slot_mean",
    "oracle_tx_gap_mean",
    "scheduled_nodes_mean",
    "adaptive_sparse_expansion_rate",
    "slots_delta_vs_rotating_b8",
    "failed_delta_vs_rotating_b8",
    "missed_delta_vs_rotating_b8",
    "preview_delta_vs_rotating_b8",
    "gap_delta_vs_rotating_b8",
]


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze the main execution-mismatch frontier."
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--preview-gap-plot", default=DEFAULT_PREVIEW_GAP_PLOT)
    parser.add_argument("--failed-missed-plot", default=DEFAULT_FAILED_MISSED_PLOT)
    parser.add_argument("--md-output", default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def ensure_parent_dir(path):
    """Create the parent directory for path if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_csv(path):
    """Read CSV rows."""
    with open(path, newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def to_float(row, field, default=0.0):
    """Parse a numeric field with a default for missing values."""
    value = row.get(field, "")
    if value == "":
        return float(default)
    return float(value)


def to_int(row, field, default=0):
    """Parse an int-like field."""
    value = row.get(field, "")
    if value == "":
        return int(default)
    return int(float(value))


def scenario_key(row):
    """Return a stable rho/delay scenario key."""
    rho = to_float(row, "channel_rho")
    delay = to_int(row, "csi_delay_slots")
    return f"rho={rho:g}, delay={delay}"


def load_method_rows(results_dir):
    """Load selected rows for each main frontier method."""
    cache = {}
    method_rows = []
    for spec in METHOD_SPECS:
        source_path = os.path.join(results_dir, spec["source_file"])
        if source_path not in cache:
            cache[source_path] = read_csv(source_path)
        selected = [
            row
            for row in cache[source_path]
            if row.get("policy") == spec["policy"]
        ]
        if not selected:
            raise ValueError(f"No rows found for policy: {spec['policy']}")
        for row in selected:
            output = {
                "label": spec["label"],
                "role": spec["role"],
                "source_file": spec["source_file"],
                "policy": spec["policy"],
                "scenario_key": scenario_key(row),
                "channel_rho": to_float(row, "channel_rho"),
                "csi_delay_slots": to_int(row, "csi_delay_slots"),
                "episodes": to_int(row, "episodes"),
                "num_seeds": to_int(row, "num_seeds"),
            }
            for field in NUMERIC_FIELDS:
                output[field] = to_float(row, field)
            output["failed_plus_missed_mean"] = (
                output["failed_nodes_mean"] + output["missed_opportunities_mean"]
            )
            method_rows.append(output)
    return method_rows


def add_rotating_b8_deltas(rows):
    """Add per-scenario deltas relative to Rotating B=8."""
    baseline_by_scenario = {
        row["scenario_key"]: row for row in rows if row["label"] == "Rotating B=8"
    }
    for row in rows:
        baseline = baseline_by_scenario.get(row["scenario_key"])
        if baseline is None:
            raise ValueError(f"Missing Rotating B=8 for {row['scenario_key']}")
        row["slots_delta_vs_rotating_b8"] = (
            row["slots_mean"] - baseline["slots_mean"]
        )
        row["failed_delta_vs_rotating_b8"] = (
            row["failed_nodes_mean"] - baseline["failed_nodes_mean"]
        )
        row["missed_delta_vs_rotating_b8"] = (
            row["missed_opportunities_mean"]
            - baseline["missed_opportunities_mean"]
        )
        row["preview_delta_vs_rotating_b8"] = (
            row["decision_preview_calls_per_slot_mean"]
            - baseline["decision_preview_calls_per_slot_mean"]
        )
        row["gap_delta_vs_rotating_b8"] = (
            row["oracle_tx_gap_mean"] - baseline["oracle_tx_gap_mean"]
        )


def mean(rows, field):
    """Mean over rows for a field."""
    return sum(float(row[field]) for row in rows) / len(rows)


def aggregate_by_label(rows):
    """Return equal-scenario averages by method label."""
    groups = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)

    aggregated = []
    order = [spec["label"] for spec in METHOD_SPECS]
    for label in order:
        group = groups[label]
        first = group[0]
        aggregated.append(
            {
                "label": label,
                "role": first["role"],
                "source_file": first["source_file"],
                "episodes": first["episodes"],
                "num_seeds": first["num_seeds"],
                "scenario_count": len(group),
                "success_mean": mean(group, "success_mean"),
                "perfect_rate": mean(group, "perfect_rate"),
                "slots_mean": mean(group, "slots_mean"),
                "failed_nodes_mean": mean(group, "failed_nodes_mean"),
                "missed_opportunities_mean": mean(
                    group, "missed_opportunities_mean"
                ),
                "failed_plus_missed_mean": mean(
                    group, "failed_plus_missed_mean"
                ),
                "decision_preview_calls_per_slot_mean": mean(
                    group, "decision_preview_calls_per_slot_mean"
                ),
                "oracle_tx_gap_mean": mean(group, "oracle_tx_gap_mean"),
                "adaptive_sparse_expansion_rate": mean(
                    group, "adaptive_sparse_expansion_rate"
                ),
                "slots_delta_vs_rotating_b8": mean(
                    group, "slots_delta_vs_rotating_b8"
                ),
                "failed_delta_vs_rotating_b8": mean(
                    group, "failed_delta_vs_rotating_b8"
                ),
                "missed_delta_vs_rotating_b8": mean(
                    group, "missed_delta_vs_rotating_b8"
                ),
                "preview_delta_vs_rotating_b8": mean(
                    group, "preview_delta_vs_rotating_b8"
                ),
                "gap_delta_vs_rotating_b8": mean(
                    group, "gap_delta_vs_rotating_b8"
                ),
            }
        )
    return aggregated


def write_analysis_csv(rows, path):
    """Write per-scenario analysis rows."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def plot_preview_gap(summary, path):
    """Plot preview cost vs oracle gap."""
    ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    colors = {
        "Rotating B=4": "#7f7f7f",
        "Rotating B=8": "#1f77b4",
        "Adaptive V2 pc=0.005": "#2ca02c",
        "Adaptive V2 pc=0.002": "#8c564b",
        "Sparse-TopK B=4 sm=3": "#ff7f0e",
        "Coverage-Aware B=4 cw=0.5 cpw=0": "#bcbd22",
        "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0": "#17becf",
        "Mask-Corrected Coverage-Aware B=3 mc=1": "#e377c2",
        "Stale-TopK B=4": "#d62728",
        "Temporal Deviation Oracle B=4": "#9467bd",
    }
    x_values = []
    y_values = []
    for item in summary:
        label = item["label"]
        x = item["decision_preview_calls_per_slot_mean"]
        y = item["oracle_tx_gap_mean"]
        x_values.append(x)
        y_values.append(y)
        marker = "D" if label == "Temporal Deviation Oracle B=4" else "o"
        ax.scatter(
            x,
            y,
            s=76,
            marker=marker,
            color=colors.get(label, "#333333"),
            zorder=3,
        )
        short_label = next(
            spec["short_label"] for spec in METHOD_SPECS if spec["label"] == label
        )
        offset = {
            "Coverage-Aware B=4 cw=0.5 cpw=0": (6, 12),
            "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0": (6, -12),
            "Mask-Corrected Coverage-Aware B=3 mc=1": (6, -28),
            "Stale-TopK B=4": (-38, 5),
        }.get(label, (6, 5))
        ax.annotate(
            short_label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )
    ordered = sorted(
        [
            item
            for item in summary
            if item["label"] != "Temporal Deviation Oracle B=4"
        ],
        key=lambda item: item["decision_preview_calls_per_slot_mean"],
    )
    ax.plot(
        [item["decision_preview_calls_per_slot_mean"] for item in ordered],
        [item["oracle_tx_gap_mean"] for item in ordered],
        color="#bbbbbb",
        linewidth=1,
        zorder=1,
    )
    ax.set_xlabel("Average preview calls per slot")
    ax.set_ylabel("Oracle tx gap")
    ax.set_title("Execution Mismatch Frontier: Preview Cost vs Oracle Gap")
    ax.set_xlim(min(x_values) - 0.6, max(x_values) + 1.0)
    ax.set_ylim(min(y_values) - 0.05, max(y_values) + 0.08)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_failed_missed(summary, path):
    """Plot failed and missed components for each method."""
    ensure_parent_dir(path)
    labels = [
        next(spec["short_label"] for spec in METHOD_SPECS if spec["label"] == item["label"])
        for item in summary
    ]
    failed = [item["failed_nodes_mean"] for item in summary]
    missed = [item["missed_opportunities_mean"] for item in summary]
    x_positions = list(range(len(summary)))

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.bar(x_positions, failed, label="Failed invitations", color="#d62728")
    ax.bar(
        x_positions,
        missed,
        bottom=failed,
        label="Missed opportunities",
        color="#1f77b4",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Average nodes per episode")
    ax.set_title("Execution Mismatch Frontier: Failed vs Missed Tradeoff")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def format_float(value, digits=3):
    """Format a float for markdown tables."""
    return f"{float(value):.{digits}f}"


def markdown_table(headers, rows):
    """Build a markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(summary, csv_output, preview_gap_plot, failed_missed_plot, path):
    """Write the paper-ready analysis markdown note."""
    ensure_parent_dir(path)
    overall_rows = []
    for item in summary:
        overall_rows.append(
            [
                item["label"],
                item["role"],
                item["scenario_count"],
                item["episodes"],
                item["num_seeds"],
                format_float(item["slots_mean"]),
                format_float(item["perfect_rate"], 2),
                format_float(item["failed_nodes_mean"]),
                format_float(item["missed_opportunities_mean"]),
                format_float(item["decision_preview_calls_per_slot_mean"], 2),
                format_float(item["oracle_tx_gap_mean"]),
            ]
        )

    delta_rows = []
    for item in summary:
        if item["label"] == "Rotating B=8":
            continue
        delta_rows.append(
            [
                item["label"],
                format_float(item["slots_delta_vs_rotating_b8"], 3),
                format_float(item["failed_delta_vs_rotating_b8"], 3),
                format_float(item["missed_delta_vs_rotating_b8"], 3),
                format_float(item["preview_delta_vs_rotating_b8"], 2),
                format_float(item["gap_delta_vs_rotating_b8"], 3),
            ]
        )

    rotating = next(item for item in summary if item["label"] == "Rotating B=8")
    sparse = next(item for item in summary if item["label"] == "Sparse-TopK B=4 sm=3")
    coverage_b4 = next(item for item in summary if item["label"] == "Coverage-Aware B=4 cw=0.5 cpw=0")
    coverage_b3 = next(item for item in summary if item["label"] == "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0")
    mask_corrected = next(item for item in summary if item["label"] == "Mask-Corrected Coverage-Aware B=3 mc=1")
    stale = next(item for item in summary if item["label"] == "Stale-TopK B=4")
    adaptive = next(item for item in summary if item["label"] == "Adaptive V2 pc=0.005")

    content = f"""# Main Results Analysis

Generated by `analyze_main_frontier.py` from existing execution-mismatch CSV files.

This note is the paper-facing analysis layer for the current main story:

> IRS-assisted multi-slot AirComp under stale/limited CSI and execution-channel mismatch, using low-cost IRS candidate generation plus current aggregate feedback.

The per-scenario CSV is `{csv_output}`. Figures are `{preview_gap_plot}` and `{failed_missed_plot}`.

## Overall Frontier

All values are equal-weight averages across the 9 temporal AR(1) rho/delay scenarios. `Samples` records the aggregate episode samples in the source CSV row, and `Seeds` records the run-seed count. Adaptive V2 currently comes from its pilot sweep; Rotating/Sparse/Coverage-Aware/Stale/Oracle come from formal frontier sweeps.

{markdown_table(
        [
            "Method",
            "Role",
            "Scenarios",
            "Samples",
            "Seeds",
            "Slots",
            "Perfect %",
            "Failed",
            "Missed",
            "Preview",
            "Gap",
        ],
        overall_rows,
    )}

## Delta Vs Rotating B=8

Negative slot/gap deltas are improvements over the strongest low-cost deployment baseline. Negative failed deltas mean fewer failed invitations; positive missed deltas mean the method gives up more currently feasible opportunities.

{markdown_table(
        [
            "Method",
            "Delta slots",
            "Delta failed",
            "Delta missed",
            "Delta preview",
            "Delta gap",
        ],
        delta_rows,
    )}

## Mechanism Interpretation

`Rotating B=8` remains the most important low-cost deployment baseline: it reaches slots `{format_float(rotating["slots_mean"])}` with only `{format_float(rotating["decision_preview_calls_per_slot_mean"], 2)}` previews and gap `{format_float(rotating["oracle_tx_gap_mean"])}`.

`Sparse-TopK B=4 sm=3` is the current reportable medium-cost positive point. Relative to `Rotating B=8`, it reduces the oracle gap by `{format_float(-sparse["gap_delta_vs_rotating_b8"])}` and failed invitations by `{format_float(-sparse["failed_delta_vs_rotating_b8"])}`, but increases preview by `{format_float(sparse["preview_delta_vs_rotating_b8"], 2)}` and missed opportunities by `{format_float(sparse["missed_delta_vs_rotating_b8"])}`.

`Coverage-Aware B=4 cw=0.5 cpw=0` is the first same-cost refinement of `Sparse-TopK B=4 sm=3`: at the same `{format_float(coverage_b4["decision_preview_calls_per_slot_mean"], 2)}` preview calls per slot, it changes gap from `{format_float(sparse["oracle_tx_gap_mean"])}` to `{format_float(coverage_b4["oracle_tx_gap_mean"])}` and missed opportunities from `{format_float(sparse["missed_opportunities_mean"])}` to `{format_float(coverage_b4["missed_opportunities_mean"])}` in the formal frontier sweep. The separate power ablation supports removing the stale power penalty; the coverage-weight ablation shows low sensitivity among `cw=0`, `cw=0.25`, and `cw=0.5`.

`Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` is the current main budget-split refinement. It keeps total preview at `{format_float(coverage_b3["decision_preview_calls_per_slot_mean"], 2)}` but shifts the split from `4` current feedback probes plus `12` stale seeds to `3` current feedback probes plus about `13` stale seeds. This lowers slots to `{format_float(coverage_b3["slots_mean"])}`, gap to `{format_float(coverage_b3["oracle_tx_gap_mean"])}`, and missed opportunities to `{format_float(coverage_b3["missed_opportunities_mean"])}`. The tradeoff is higher failed invitations than the B=4 coverage setting: `{format_float(coverage_b3["failed_nodes_mean"])}` versus `{format_float(coverage_b4["failed_nodes_mean"])}`.

`Mask-Corrected Coverage-Aware B=3 mc=1` is the strongest same-preview method. It keeps the B3 candidate-generation and current aggregate confirmation steps, then uses the confirmed IRS aggregate feedback count to correct the stale invitation mask cardinality. At the same `{format_float(mask_corrected["decision_preview_calls_per_slot_mean"], 2)}` preview calls, it lowers slots/gap from `{format_float(coverage_b3["slots_mean"])}/{format_float(coverage_b3["oracle_tx_gap_mean"])}` to `{format_float(mask_corrected["slots_mean"])}/{format_float(mask_corrected["oracle_tx_gap_mean"])}` and lowers failed/missed from `{format_float(coverage_b3["failed_nodes_mean"])}/{format_float(coverage_b3["missed_opportunities_mean"])}` to `{format_float(mask_corrected["failed_nodes_mean"])}/{format_float(mask_corrected["missed_opportunities_mean"])}`.

`Stale-TopK B=4` is still a high-cost positive reference, but it costs `{format_float(stale["decision_preview_calls_per_slot_mean"], 2)}` previews per slot. After mask correction, the same-preview B3 method now beats this stale-ranking reference on slots and oracle gap.

`Adaptive V2 pc=0.005` is best read as a cost-quality continuum point, not as the final method. It uses `{format_float(adaptive["decision_preview_calls_per_slot_mean"], 2)}` previews on average and lowers the gap versus `Rotating B=8` by `{format_float(-adaptive["gap_delta_vs_rotating_b8"])}`.

The main mechanism is now clear: stale shortlist generation plus current aggregate confirmation fixes part of candidate selection, but the residual B3 gap is dominated by stale invitation-mask mismatch. Mask correction directly targets that mismatch using only aggregate current feedback count, which explains why it improves failed invitations and missed opportunities simultaneously.

The separate invitation-mask noise-aware sweep shows the boundary of that mechanism. Direct `mc=1` remains the best gap setting through feedback-noise std `0.05`. At std `0.1`, clipped `mc=1 clip=2` improves direct `mc=1` on gap, failed invitations, and missed opportunities (`0.818/3.275/0.599` versus `0.856/5.264/0.734`). The reportable claim should therefore state the aggregate-count reliability assumption and include clipped target-count correction as the high-noise robustness variant.

## Generated Figures

- `{preview_gap_plot}`: cost-quality frontier, with preview cost on the x-axis and oracle gap on the y-axis.
- `{failed_missed_plot}`: stacked failed/missed components, showing the tradeoff behind each method.

## Research Implication

Do not continue broad SAC/imitation/bandit branches as main work. The invitation-mask correction result package is now the paper-facing entry point: direct `Mask-Corrected Coverage-Aware B=3 mc=1` is the reliable-feedback main point, and `mc=1 clip=2` is the high-noise robustness variant. Next work should be writing the paper table and narrative, not adding another heuristic branch.
"""
    with open(path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write(content)


def main():
    """Run analysis generation."""
    args = parse_args()
    rows = load_method_rows(args.results_dir)
    add_rotating_b8_deltas(rows)
    summary = aggregate_by_label(rows)
    write_analysis_csv(rows, args.csv_output)
    plot_preview_gap(summary, args.preview_gap_plot)
    plot_failed_missed(summary, args.failed_missed_plot)
    write_markdown(
        summary,
        args.csv_output,
        args.preview_gap_plot,
        args.failed_missed_plot,
        args.md_output,
    )
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.preview_gap_plot}")
    print(f"Wrote {args.failed_missed_plot}")
    print(f"Wrote {args.md_output}")


if __name__ == "__main__":
    main()
