"""从冻结结果 CSV 生成论文图 2、图 3 和图 4 所需的绘图点与 PNG。"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_paper_tables import TABLE1_METHOD_ORDER


DEFAULT_SOURCE = Path("results/execution_mismatch/final_execution_baseline_summary.csv")
DEFAULT_INVITATION_SOURCE = Path("results/execution_mismatch/final_invitation_mask_analysis.csv")
DEFAULT_POINTS_CSV = Path("results/paper/figure2_figure3_points.csv")
DEFAULT_FIGURE2 = Path("results/paper/figure2_preview_gap_frontier.png")
DEFAULT_FIGURE3 = Path("results/paper/figure3_failed_missed_tradeoff.png")
DEFAULT_FIGURE4_POINTS_CSV = Path("results/paper/figure4_invitation_mask_noise_points.csv")
DEFAULT_FIGURE4_GAP = Path("results/paper/figure4_invitation_mask_gap_noise.png")
DEFAULT_FIGURE4_FAILED_MISSED = Path(
    "results/paper/figure4_invitation_mask_failed_missed_noise.png"
)

POINT_COLUMNS = [
    "Method",
    "ShortLabel",
    "Role",
    "Preview",
    "Gap",
    "Failed",
    "Missed",
    "IsProposed",
    "IsOracle",
]

FIGURE4_COLUMNS = [
    "Label",
    "ShortLabel",
    "Role",
    "FeedbackNoiseStd",
    "Gap",
    "Failed",
    "Missed",
    "IsNoNoiseCorrectionTradeoff",
    "IsHighNoiseGapBest",
]

SHORT_LABELS = {
    "Rotating B=8": "Rot B8",
    "Sparse-TopK B=4 sm=3": "Sparse",
    "Coverage-Aware B=4 cw=0.5 cpw=0": "Cov B4",
    "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0": "Cov B3",
    "Mask-Corrected Coverage-Aware B=3 mc=1": "MaskCorr",
    "Stale-TopK B=4": "Stale",
    "Temporal Deviation Oracle B=4": "Oracle",
}

PROPOSED_METHOD = "Mask-Corrected Coverage-Aware B=3 mc=1"
ORACLE_METHOD = "Temporal Deviation Oracle B=4"

COLORS = {
    "Rotating B=8": "#4c78a8",
    "Sparse-TopK B=4 sm=3": "#72b7b2",
    "Coverage-Aware B=4 cw=0.5 cpw=0": "#59a14f",
    "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0": "#8cd17d",
    PROPOSED_METHOD: "#e15759",
    "Stale-TopK B=4": "#b07aa1",
    ORACLE_METHOD: "#6b7280",
}

FIGURE4_LABEL_ORDER = [
    "Coverage-Aware B=3",
    "Direct Mask Correction mc=1",
    "Clipped Mask Correction mc=1 clip=2",
]

FIGURE4_NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1]

FIGURE4_COLORS = {
    "Coverage-Aware B=3": "#4c78a8",
    "Direct Mask Correction mc=1": "#e15759",
    "Clipped Mask Correction mc=1 clip=2": "#59a14f",
}

FIGURE4_MARKERS = {
    "Coverage-Aware B=3": "o",
    "Direct Mask Correction mc=1": "s",
    "Clipped Mask Correction mc=1 clip=2": "^",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def row_by_label(rows, label):
    matches = [row for row in rows if row.get("label") == label]
    if not matches:
        raise ValueError(f"missing required paper figure method: {label}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous paper figure method: {label}")
    return matches[0]


def row_by_label_noise(rows, label, noise):
    matches = [
        row
        for row in rows
        if row.get("label") == label
        and abs(float(row.get("feedback_noise_std", "nan")) - noise) < 1e-12
    ]
    if not matches:
        raise ValueError(f"missing required Figure 4 row: {label}, noise={noise:g}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous Figure 4 row: {label}, noise={noise:g}")
    return matches[0]


def format_point(row):
    method = row["label"]
    return {
        "Method": method,
        "ShortLabel": SHORT_LABELS[method],
        "Role": row["role"],
        "Preview": f"{float(row['decision_preview_calls_per_slot_mean']):.2f}",
        "Gap": f"{float(row['oracle_tx_gap_mean']):.3f}",
        "Failed": f"{float(row['failed_nodes_mean']):.3f}",
        "Missed": f"{float(row['missed_opportunities_mean']):.3f}",
        "IsProposed": str(method == PROPOSED_METHOD),
        "IsOracle": str(method == ORACLE_METHOD),
    }


def build_points(source_rows):
    return [format_point(row_by_label(source_rows, label)) for label in TABLE1_METHOD_ORDER]


def format_figure4_point(row):
    label = row["label"]
    noise = float(row["feedback_noise_std"])
    return {
        "Label": label,
        "ShortLabel": row["short_label"],
        "Role": row["role"],
        "FeedbackNoiseStd": f"{noise:g}",
        "Gap": f"{float(row['oracle_tx_gap_mean']):.3f}",
        "Failed": f"{float(row['failed_nodes_mean']):.3f}",
        "Missed": f"{float(row['missed_opportunities_mean']):.3f}",
        "IsNoNoiseCorrectionTradeoff": str(
            label == "Direct Mask Correction mc=1" and abs(noise) < 1e-12
        ),
        "IsHighNoiseGapBest": str(
            label == "Direct Mask Correction mc=1" and abs(noise - 0.1) < 1e-12
        ),
    }


def build_figure4_points(source_rows):
    return [
        format_figure4_point(row_by_label_noise(source_rows, label, noise))
        for noise in FIGURE4_NOISE_LEVELS
        for label in FIGURE4_LABEL_ORDER
    ]


def write_points_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def point_style(row):
    method = row["Method"]
    if method == PROPOSED_METHOD:
        return {"marker": "*", "s": 190, "linewidth": 1.2, "edgecolor": "black"}
    if method == ORACLE_METHOD:
        return {"marker": "X", "s": 120, "linewidth": 1.0, "edgecolor": "black"}
    return {"marker": "o", "s": 88, "linewidth": 0.8, "edgecolor": "white"}


def annotate(ax, x, y, label, dx=0.05, dy=0.015):
    ax.annotate(
        label,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset fontsize",
        fontsize=8,
        ha="left",
        va="bottom",
    )


def setup_axes(ax):
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#374151", labelsize=9)


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_preview_gap(rows, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for row in rows:
        x = float(row["Preview"])
        y = float(row["Gap"])
        style = point_style(row)
        ax.scatter(x, y, color=COLORS[row["Method"]], **style)
        annotate(ax, x, y, row["ShortLabel"])

    ax.set_xlabel("Decision preview calls per slot", fontsize=10)
    ax.set_ylabel("Oracle tx gap", fontsize=10)
    ax.set_title("Preview cost vs oracle gap", fontsize=11, pad=10)
    ax.set_xlim(2.5, 21.5)
    ax.set_ylim(0.20, 0.80)
    setup_axes(ax)
    save_figure(fig, path)


def plot_failed_missed(rows, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for row in rows:
        x = float(row["Failed"])
        y = float(row["Missed"])
        style = point_style(row)
        ax.scatter(x, y, color=COLORS[row["Method"]], **style)
        annotate(ax, x, y, row["ShortLabel"])

    ax.set_xlabel("Failed invited devices", fontsize=10)
    ax.set_ylabel("Missed opportunities", fontsize=10)
    ax.set_title("Failed invitations vs missed opportunities", fontsize=11, pad=10)
    ax.set_xlim(0.20, 1.50)
    ax.set_ylim(0.20, 1.45)
    setup_axes(ax)
    save_figure(fig, path)


def figure4_series(rows, label):
    return sorted(
        [row for row in rows if row["Label"] == label],
        key=lambda row: float(row["FeedbackNoiseStd"]),
    )


def plot_figure4_gap(rows, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label in FIGURE4_LABEL_ORDER:
        series = figure4_series(rows, label)
        ax.plot(
            [float(row["FeedbackNoiseStd"]) for row in series],
            [float(row["Gap"]) for row in series],
            label=series[0]["ShortLabel"],
            color=FIGURE4_COLORS[label],
            marker=FIGURE4_MARKERS[label],
            linewidth=1.8,
            markersize=6,
        )

    ax.set_xlabel("Aggregate-feedback noise std", fontsize=10)
    ax.set_ylabel("Oracle tx gap", fontsize=10)
    ax.set_title("Invitation-mask correction noise robustness", fontsize=11, pad=10)
    ax.set_xlim(-0.005, 0.105)
    ax.set_xticks(FIGURE4_NOISE_LEVELS)
    ax.set_xticklabels([f"{noise:g}" for noise in FIGURE4_NOISE_LEVELS])
    ax.legend(frameon=False, fontsize=8)
    setup_axes(ax)
    save_figure(fig, path)


def plot_figure4_failed_missed(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True)
    panels = [
        ("Failed", "Failed invited devices", "Failed invitations"),
        ("Missed", "Missed opportunities", "Missed opportunities"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels):
        for label in FIGURE4_LABEL_ORDER:
            series = figure4_series(rows, label)
            ax.plot(
                [float(row["FeedbackNoiseStd"]) for row in series],
                [float(row[metric]) for row in series],
                label=series[0]["ShortLabel"],
                color=FIGURE4_COLORS[label],
                marker=FIGURE4_MARKERS[label],
                linewidth=1.8,
                markersize=5,
            )
        ax.set_xlabel("Feedback noise std", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlim(-0.005, 0.105)
        ax.set_xticks(FIGURE4_NOISE_LEVELS)
        ax.set_xticklabels([f"{noise:g}" for noise in FIGURE4_NOISE_LEVELS])
        setup_axes(ax)

    axes[1].legend(frameon=False, fontsize=8)
    save_figure(fig, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--invitation-source", type=Path, default=DEFAULT_INVITATION_SOURCE)
    parser.add_argument("--points-csv", type=Path, default=DEFAULT_POINTS_CSV)
    parser.add_argument("--figure2", type=Path, default=DEFAULT_FIGURE2)
    parser.add_argument("--figure3", type=Path, default=DEFAULT_FIGURE3)
    parser.add_argument("--figure4-points-csv", type=Path, default=DEFAULT_FIGURE4_POINTS_CSV)
    parser.add_argument("--figure4-gap", type=Path, default=DEFAULT_FIGURE4_GAP)
    parser.add_argument("--figure4-failed-missed", type=Path, default=DEFAULT_FIGURE4_FAILED_MISSED)
    return parser.parse_args()


def main():
    args = parse_args()
    source_rows = read_rows(args.source)
    points = build_points(source_rows)
    write_points_csv(args.points_csv, points, POINT_COLUMNS)
    plot_preview_gap(points, args.figure2)
    plot_failed_missed(points, args.figure3)
    invitation_rows = read_rows(args.invitation_source)
    figure4_points = build_figure4_points(invitation_rows)
    write_points_csv(args.figure4_points_csv, figure4_points, FIGURE4_COLUMNS)
    plot_figure4_gap(figure4_points, args.figure4_gap)
    plot_figure4_failed_missed(figure4_points, args.figure4_failed_missed)
    print(f"wrote {args.points_csv}")
    print(f"wrote {args.figure2}")
    print(f"wrote {args.figure3}")
    print(f"wrote {args.figure4_points_csv}")
    print(f"wrote {args.figure4_gap}")
    print(f"wrote {args.figure4_failed_missed}")


if __name__ == "__main__":
    main()
