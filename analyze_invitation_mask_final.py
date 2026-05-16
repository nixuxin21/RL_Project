"""
Generate final invitation-mask correction analysis artifacts.

The script reads existing CSV results only. It does not rerun simulations.
Outputs:
- final invitation-mask summary CSV
- gap-vs-feedback-noise plot
- failed/missed-vs-feedback-noise plot
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
    "final_invitation_mask_analysis.csv",
)
DEFAULT_GAP_PLOT = os.path.join(
    DEFAULT_RESULTS_DIR,
    "final_invitation_mask_gap_noise.png",
)
DEFAULT_FAILED_MISSED_PLOT = os.path.join(
    DEFAULT_RESULTS_DIR,
    "final_invitation_mask_failed_missed_noise.png",
)
DEFAULT_MD_OUTPUT = os.path.join("docs", "FINAL_INVITATION_MASK_ANALYSIS.md")
PAPER_FIGURE4_POINTS = os.path.join(
    "results",
    "paper",
    "figure4_invitation_mask_noise_points.csv",
)
PAPER_FIGURE4_GAP = os.path.join(
    "results",
    "paper",
    "figure4_invitation_mask_gap_noise.png",
)
PAPER_FIGURE4_FAILED_MISSED = os.path.join(
    "results",
    "paper",
    "figure4_invitation_mask_failed_missed_noise.png",
)

NOISE_AWARE_FILE = (
    "invitation_mask_correction_noise_aware_formal_ep300_runs3_"
    "rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_"
    "mc0-0p75-1_clipinf-2_fbn0-0p02-0p05-0p1.csv"
)

METHOD_SPECS = [
    {
        "label": "Coverage-Aware B=3",
        "short_label": "B3",
        "role": "uncorrected baseline",
        "policy": "Coverage-Aware B=3 sm=4.1",
    },
    {
        "label": "Direct Mask Correction mc=1",
        "short_label": "Direct",
        "role": "reliable-feedback main method",
        "policy": "Mask-Corrected Coverage-Aware B=3 mc=1",
    },
    {
        "label": "Clipped Mask Correction mc=1 clip=2",
        "short_label": "Clip2",
        "role": "high-noise robustness variant",
        "policy": "Noise-Aware Mask-Corrected Coverage-Aware B=3 mc=1 clip=2",
    },
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
    "mask_correction_added_mean",
    "mask_correction_pruned_mean",
    "mask_correction_target_delta_mean",
    "mask_correction_applied_rate",
]

OUTPUT_FIELDS = [
    "label",
    "short_label",
    "role",
    "policy",
    "feedback_noise_std",
    "scenario_count",
    "episodes_per_scenario",
    "num_seeds",
    "success_mean",
    "perfect_rate",
    "slots_mean",
    "failed_nodes_mean",
    "missed_opportunities_mean",
    "failed_plus_missed_mean",
    "true_opportunities_mean",
    "decision_preview_calls_per_slot_mean",
    "oracle_tx_gap_mean",
    "mask_correction_added_mean",
    "mask_correction_pruned_mean",
    "mask_correction_target_delta_mean",
    "mask_correction_applied_rate",
    "slots_delta_vs_b3",
    "failed_delta_vs_b3",
    "missed_delta_vs_b3",
    "gap_delta_vs_b3",
    "slots_delta_vs_direct",
    "failed_delta_vs_direct",
    "missed_delta_vs_direct",
    "gap_delta_vs_direct",
]


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze final invitation-mask correction results."
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--gap-plot", default=DEFAULT_GAP_PLOT)
    parser.add_argument("--failed-missed-plot", default=DEFAULT_FAILED_MISSED_PLOT)
    parser.add_argument("--md-output", default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def ensure_parent_dir(path):
    """Create the parent directory for a path if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_csv(path):
    """Read CSV rows."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input CSV: {path}")
    with open(path, newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def to_float(row, field, default=0.0):
    """Parse a numeric field."""
    value = row.get(field, "")
    if value == "":
        return float(default)
    return float(value)


def to_int(row, field, default=0):
    """Parse an integer-like field."""
    value = row.get(field, "")
    if value == "":
        return int(default)
    return int(float(value))


def mean(rows, field, default=0.0):
    """Return the mean of a numeric field."""
    if not rows:
        return float(default)
    return sum(to_float(row, field, default) for row in rows) / len(rows)


def first_int(rows, field):
    """Return the first int-like field from rows."""
    for row in rows:
        value = row.get(field, "")
        if value != "":
            return int(float(value))
    return 0


def format_float(value, digits=3):
    """Format a float for markdown."""
    return f"{float(value):.{digits}f}"


def format_noise(value):
    """Format a feedback-noise value compactly."""
    return f"{float(value):g}"


def markdown_table(headers, rows):
    """Render a markdown table."""
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        output.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(output)


def aggregate_method(rows, spec, noise_level):
    """Aggregate one method at one feedback-noise level."""
    selected = [
        row
        for row in rows
        if row.get("policy") == spec["policy"]
        and abs(to_float(row, "confirmation_feedback_noise_std") - noise_level) < 1e-12
    ]
    if not selected:
        raise ValueError(
            f"No rows found for policy={spec['policy']} noise={noise_level}"
        )
    item = {
        "label": spec["label"],
        "short_label": spec["short_label"],
        "role": spec["role"],
        "policy": spec["policy"],
        "feedback_noise_std": float(noise_level),
        "scenario_count": len(selected),
        "episodes_per_scenario": first_int(selected, "episodes"),
        "num_seeds": first_int(selected, "num_seeds"),
    }
    for field in NUMERIC_FIELDS:
        item[field] = mean(selected, field)
    item["failed_plus_missed_mean"] = (
        item["failed_nodes_mean"] + item["missed_opportunities_mean"]
    )
    return item


def add_deltas(summary):
    """Add deltas against B3 and direct mask correction at each noise level."""
    by_noise_label = {
        (float(item["feedback_noise_std"]), item["label"]): item
        for item in summary
    }
    for item in summary:
        noise = float(item["feedback_noise_std"])
        b3 = by_noise_label[(noise, "Coverage-Aware B=3")]
        direct = by_noise_label[(noise, "Direct Mask Correction mc=1")]
        item["slots_delta_vs_b3"] = item["slots_mean"] - b3["slots_mean"]
        item["failed_delta_vs_b3"] = (
            item["failed_nodes_mean"] - b3["failed_nodes_mean"]
        )
        item["missed_delta_vs_b3"] = (
            item["missed_opportunities_mean"] - b3["missed_opportunities_mean"]
        )
        item["gap_delta_vs_b3"] = item["oracle_tx_gap_mean"] - b3["oracle_tx_gap_mean"]
        item["slots_delta_vs_direct"] = item["slots_mean"] - direct["slots_mean"]
        item["failed_delta_vs_direct"] = (
            item["failed_nodes_mean"] - direct["failed_nodes_mean"]
        )
        item["missed_delta_vs_direct"] = (
            item["missed_opportunities_mean"]
            - direct["missed_opportunities_mean"]
        )
        item["gap_delta_vs_direct"] = (
            item["oracle_tx_gap_mean"] - direct["oracle_tx_gap_mean"]
        )


def load_summary(results_dir):
    """Load and aggregate the final invitation-mask methods."""
    source_path = os.path.join(results_dir, NOISE_AWARE_FILE)
    rows = read_csv(source_path)
    policies = {spec["policy"] for spec in METHOD_SPECS}
    noise_levels = sorted(
        {
            to_float(row, "confirmation_feedback_noise_std")
            for row in rows
            if row.get("policy") in policies
        }
    )
    summary = []
    for noise_level in noise_levels:
        for spec in METHOD_SPECS:
            summary.append(aggregate_method(rows, spec, noise_level))
    add_deltas(summary)
    return summary, source_path


def write_summary_csv(path, summary):
    """Write aggregated summary CSV."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for item in summary:
            writer.writerow({field: item.get(field, "") for field in OUTPUT_FIELDS})


def series_by_label(summary, label):
    """Return summary items for a label ordered by feedback-noise level."""
    return sorted(
        [item for item in summary if item["label"] == label],
        key=lambda item: item["feedback_noise_std"],
    )


def plot_gap_vs_noise(summary, path):
    """Plot oracle gap against feedback noise."""
    ensure_parent_dir(path)
    colors = {
        "Coverage-Aware B=3": "#7f7f7f",
        "Direct Mask Correction mc=1": "#e377c2",
        "Clipped Mask Correction mc=1 clip=2": "#2ca02c",
    }
    markers = {
        "Coverage-Aware B=3": "o",
        "Direct Mask Correction mc=1": "s",
        "Clipped Mask Correction mc=1 clip=2": "^",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for spec in METHOD_SPECS:
        label = spec["label"]
        items = series_by_label(summary, label)
        ax.plot(
            [item["feedback_noise_std"] for item in items],
            [item["oracle_tx_gap_mean"] for item in items],
            marker=markers[label],
            linewidth=2.0,
            markersize=6,
            color=colors[label],
            label=spec["short_label"],
        )
    ax.set_xlabel("Feedback noise std")
    ax.set_ylabel("Oracle tx gap")
    ax.set_title("Invitation Mask Correction: Gap vs Feedback Noise")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_failed_missed_vs_noise(summary, path):
    """Plot failed invitations and missed opportunities against feedback noise."""
    ensure_parent_dir(path)
    colors = {
        "Coverage-Aware B=3": "#7f7f7f",
        "Direct Mask Correction mc=1": "#e377c2",
        "Clipped Mask Correction mc=1 clip=2": "#2ca02c",
    }
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True)
    for spec in METHOD_SPECS:
        label = spec["label"]
        items = series_by_label(summary, label)
        x_values = [item["feedback_noise_std"] for item in items]
        axes[0].plot(
            x_values,
            [item["failed_nodes_mean"] for item in items],
            marker="o",
            linewidth=1.9,
            color=colors[label],
            label=spec["short_label"],
        )
        axes[1].plot(
            x_values,
            [item["missed_opportunities_mean"] for item in items],
            marker="s",
            linewidth=1.9,
            color=colors[label],
            label=spec["short_label"],
        )
    axes[0].set_ylabel("Failed invitations")
    axes[1].set_ylabel("Missed opportunities")
    axes[1].set_xlabel("Feedback noise std")
    axes[0].set_title("Invitation Mask Correction: Failed and Missed vs Noise")
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    axes[0].legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def best_by_noise(summary):
    """Return the lowest-gap method at each feedback-noise level."""
    noise_levels = sorted({item["feedback_noise_std"] for item in summary})
    best = []
    for noise in noise_levels:
        items = [item for item in summary if item["feedback_noise_std"] == noise]
        best.append(
            min(
                items,
                key=lambda item: (
                    item["oracle_tx_gap_mean"],
                    item["slots_mean"],
                    item["failed_nodes_mean"],
                ),
            )
        )
    return best


def find_item(summary, label, noise_level):
    """Find one summary item."""
    for item in summary:
        if item["label"] == label and abs(item["feedback_noise_std"] - noise_level) < 1e-12:
            return item
    raise ValueError(f"Missing summary item: {label} noise={noise_level}")


def write_markdown(path, source_path, csv_path, gap_plot, failed_missed_plot, summary):
    """Write final invitation-mask markdown analysis."""
    ensure_parent_dir(path)
    reliable = find_item(summary, "Direct Mask Correction mc=1", 0.0)
    high_noise_direct = find_item(summary, "Direct Mask Correction mc=1", 0.1)
    high_noise_clip = find_item(summary, "Clipped Mask Correction mc=1 clip=2", 0.1)
    b3_zero = find_item(summary, "Coverage-Aware B=3", 0.0)
    b3_high = find_item(summary, "Coverage-Aware B=3", 0.1)

    headline_rows = [
        [
            "Reliable feedback main",
            "Direct mc=1",
            format_noise(reliable["feedback_noise_std"]),
            format_float(reliable["slots_mean"]),
            format_float(reliable["failed_nodes_mean"]),
            format_float(reliable["missed_opportunities_mean"]),
            format_float(reliable["oracle_tx_gap_mean"]),
            format_float(reliable["gap_delta_vs_b3"]),
        ],
        [
            "High-noise robust",
            "mc=1 clip=2",
            format_noise(high_noise_clip["feedback_noise_std"]),
            format_float(high_noise_clip["slots_mean"]),
            format_float(high_noise_clip["failed_nodes_mean"]),
            format_float(high_noise_clip["missed_opportunities_mean"]),
            format_float(high_noise_clip["oracle_tx_gap_mean"]),
            format_float(high_noise_clip["gap_delta_vs_b3"]),
        ],
    ]

    noise_rows = []
    for noise_item in best_by_noise(summary):
        b3 = find_item(summary, "Coverage-Aware B=3", noise_item["feedback_noise_std"])
        direct = find_item(
            summary,
            "Direct Mask Correction mc=1",
            noise_item["feedback_noise_std"],
        )
        clip = find_item(
            summary,
            "Clipped Mask Correction mc=1 clip=2",
            noise_item["feedback_noise_std"],
        )
        noise_rows.append(
            [
                format_noise(noise_item["feedback_noise_std"]),
                format_float(b3["oracle_tx_gap_mean"]),
                format_float(direct["oracle_tx_gap_mean"]),
                format_float(clip["oracle_tx_gap_mean"]),
                noise_item["short_label"],
                format_float(noise_item["oracle_tx_gap_mean"]),
            ]
        )

    high_noise_rows = [
        [
            "Direct mc=1",
            format_float(high_noise_direct["slots_mean"]),
            format_float(high_noise_direct["failed_nodes_mean"]),
            format_float(high_noise_direct["missed_opportunities_mean"]),
            format_float(high_noise_direct["oracle_tx_gap_mean"]),
        ],
        [
            "mc=1 clip=2",
            format_float(high_noise_clip["slots_mean"]),
            format_float(high_noise_clip["failed_nodes_mean"]),
            format_float(high_noise_clip["missed_opportunities_mean"]),
            format_float(high_noise_clip["oracle_tx_gap_mean"]),
        ],
        [
            "Clip2 delta",
            format_float(high_noise_clip["slots_delta_vs_direct"]),
            format_float(high_noise_clip["failed_delta_vs_direct"]),
            format_float(high_noise_clip["missed_delta_vs_direct"]),
            format_float(high_noise_clip["gap_delta_vs_direct"]),
        ],
    ]

    content = f"""# Final Invitation Mask Analysis

Generated by `analyze_invitation_mask_final.py` from existing CSV results.

Source CSV: `{source_path}`

Generated artifacts:

- `{csv_path}`
- `{gap_plot}`
- `{failed_missed_plot}`

Paper-facing Figure 4 artifacts are generated from `{csv_path}` by `make paper-figures`:

- `{PAPER_FIGURE4_POINTS}`
- `{PAPER_FIGURE4_GAP}`
- `{PAPER_FIGURE4_FAILED_MISSED}`

Use the `results/paper/` files for the paper main text. The `results/execution_mismatch/` plots above remain the analysis-layer trace.

## Headline Results

{markdown_table(
        [
            "Use",
            "Method",
            "Noise std",
            "Slots",
            "Failed",
            "Missed",
            "Gap",
            "Gap delta vs B3",
        ],
        headline_rows,
    )}

The reliable-feedback main result is direct `mc=1`: at the same preview cost `16`, it lowers the no-noise B3 gap from `{format_float(b3_zero["oracle_tx_gap_mean"])}` to `{format_float(reliable["oracle_tx_gap_mean"])}`. The high-noise robustness result is `mc=1 clip=2`: at feedback-noise std `0.1`, it lowers the B3 gap from `{format_float(b3_high["oracle_tx_gap_mean"])}` to `{format_float(high_noise_clip["oracle_tx_gap_mean"])}`.

## Noise Sweep

{markdown_table(
        [
            "Noise std",
            "B3 gap",
            "Direct gap",
            "Clip2 gap",
            "Best",
            "Best gap",
        ],
        noise_rows,
    )}

Direct `mc=1` remains the lowest-gap method through feedback-noise std `0.05`. At std `0.1`, clipping the per-slot target-count correction to at most two nodes becomes better.

## High-Noise Tradeoff

{markdown_table(
        ["Method", "Slots", "Failed", "Missed", "Gap"],
        high_noise_rows,
    )}

The clipped variant is not the low-noise main method: it slightly worsens low-noise gap. Its value is high-noise robustness because it reduces over-invitation when the aggregate feedback count is noisy.

## Paper Claim

The final contribution should be stated as invitation-mask correction after IRS confirmation. The method does not change candidate generation or preview cost. It uses aggregate current feedback count to repair the stale invitation mask for the confirmed IRS. Reliable aggregate feedback uses exact direct correction (`mc=1`); high-noise aggregate feedback uses clipped target-count correction (`mc=1 clip=2`).
"""
    with open(path, "w", encoding="utf-8") as mdfile:
        mdfile.write(content)


def main():
    """Run final invitation-mask analysis."""
    args = parse_args()
    summary, source_path = load_summary(args.results_dir)
    write_summary_csv(args.csv_output, summary)
    plot_gap_vs_noise(summary, args.gap_plot)
    plot_failed_missed_vs_noise(summary, args.failed_missed_plot)
    write_markdown(
        args.md_output,
        source_path,
        args.csv_output,
        args.gap_plot,
        args.failed_missed_plot,
        summary,
    )
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.gap_plot}")
    print(f"Wrote {args.failed_missed_plot}")
    print(f"Wrote {args.md_output}")


if __name__ == "__main__":
    main()
