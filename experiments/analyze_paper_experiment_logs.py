"""Analyze raw paper-suite logs with paired statistics and simple figures."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ms_aircomp.execution_result_summary import result_metadata


DEFAULT_OUTPUT_DIR = Path("results") / "paper_experiment_analysis"
BOOTSTRAP_SEED = 20260519
BOOTSTRAP_SAMPLES = 2000
TIE_TOLERANCE = 1e-9

REQUIRED_SCENARIO_COLUMNS = {
    "config_name",
    "method_name",
    "policy_name",
    "run_index",
    "run_seed",
    "episode_seed",
    "mismatch_model",
    "rho",
    "csi_delay",
    "K",
    "num_slots",
    "num_irs_elements",
    "num_codebook_states",
    "probe_budget",
    "completion",
    "perfect_indicator",
    "slots_used",
    "failed_invitations",
    "missed_opportunities",
    "achieved_tx_count",
    "oracle_gap",
    "oracle_gap_mean",
    "nmse",
    "aircomp_raw_mse",
    "energy",
    "energy_per_success",
    "total_probe_calls",
    "total_protocol_cost",
}

REQUIRED_SLOT_COLUMNS = {
    "config_name",
    "method_name",
    "run_index",
    "run_seed",
    "episode_seed",
    "slot_idx",
    "failed_invited_count",
    "missed_feasible_count",
    "oracle_gap",
    "nmse",
    "total_protocol_cost",
}

METRICS = [
    ("completion", "Completion", "higher"),
    ("perfect_indicator", "Perfect", "higher"),
    ("achieved_tx_count", "AchievedTx", "higher"),
    ("slots_used", "Slots", "lower"),
    ("failed_invitations", "FailedInvites", "lower"),
    ("missed_opportunities", "MissedOpps", "lower"),
    ("oracle_gap", "OracleGap", "lower"),
    ("oracle_gap_mean", "OracleGapPerSlot", "lower"),
    ("nmse", "NMSE", "lower"),
    ("aircomp_raw_mse", "RawMSE", "lower"),
    ("energy", "Energy", "lower"),
    ("energy_per_success", "EnergyPerSuccess", "lower"),
    ("total_protocol_cost", "ProtocolCost", "lower"),
    ("total_probe_calls", "ProbeCalls", "lower"),
]

FIGURE_METHOD_LIMIT = 12


def parse_args():
    """Parse analysis command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze raw paper experiment suite logs.")
    parser.add_argument("--input-dir", required=True, help="Suite run directory or parent directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--analysis-name", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--baseline-methods",
        default="Random Same-Budget,Coverage-Aware Sparse-TopK,Count-Only Mask-Corrected",
        help="Comma-separated method-name substrings used as paired baselines.",
    )
    return parser.parse_args()


def csv_items(value):
    """Parse a comma-separated string."""
    return [item.strip() for item in str(value).split(",") if item.strip()]


def read_csv(path):
    """Read a CSV file into a list of dictionaries."""
    with open(path, newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def validate_columns(path, fieldnames, required):
    """Fail clearly when a raw log is missing required columns."""
    missing = sorted(required - set(fieldnames or []))
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")


def discover_structured_logs(input_dir):
    """Find scenario/slot/metadata log triplets under an input directory."""
    root = Path(input_dir)
    scenario_paths = sorted(root.rglob("structured_logs/scenario_summary.csv"))
    if not scenario_paths:
        raise ValueError(f"no structured scenario logs found under {root}")
    jobs = []
    for scenario_path in scenario_paths:
        log_dir = scenario_path.parent
        slot_path = log_dir / "slot_records.csv"
        metadata_path = log_dir / "run_metadata.jsonl"
        if not slot_path.exists():
            raise ValueError(f"missing slot log for {scenario_path}: {slot_path}")
        if not metadata_path.exists():
            raise ValueError(f"missing run metadata for {scenario_path}: {metadata_path}")
        jobs.append((scenario_path, slot_path, metadata_path))
    return jobs


def metadata_key(row):
    """Build a key shared by run metadata and raw rows."""
    return (
        row.get("config_name", ""),
        row.get("method_name", ""),
        str(row.get("run_index", "")),
    )


def load_metadata(path):
    """Load run metadata keyed by config/method/run index."""
    metadata = {}
    with open(path, encoding="utf-8") as jsonfile:
        for line in jsonfile:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata[metadata_key(row)] = row
    return metadata


def attach_metadata(row, metadata, source_path):
    """Attach method hyperparameters and source path to one raw row."""
    meta = metadata.get(metadata_key(row), {})
    hyper = meta.get("method_hyperparameters", {})
    row = dict(row)
    row["source_path"] = str(source_path)
    row["feedback_noise_std"] = str(hyper.get("confirmation_feedback_noise_std", ""))
    row["probing_policy"] = str(hyper.get("probing_policy", ""))
    row["posterior_probe_objective"] = str(hyper.get("posterior_probe_objective", ""))
    row["posterior_cardinality_policy"] = str(hyper.get("posterior_cardinality_policy", ""))
    row["result_role"] = result_metadata({"name": row.get("method_name", "")})["result_role"]
    row["inference_uses_hidden_current_csi"] = result_metadata(
        {"name": row.get("method_name", "")}
    )["inference_uses_hidden_current_csi"]
    row["uses_hidden_training_labels"] = result_metadata(
        {"name": row.get("method_name", "")}
    )["uses_hidden_training_labels"]
    return row


def load_raw_logs(input_dir):
    """Load all raw scenario and slot records from a suite run."""
    scenario_rows = []
    slot_rows = []
    for scenario_path, slot_path, metadata_path in discover_structured_logs(input_dir):
        metadata = load_metadata(metadata_path)
        scenario_records = read_csv(scenario_path)
        slot_records = read_csv(slot_path)
        with open(scenario_path, newline="", encoding="utf-8") as csvfile:
            validate_columns(scenario_path, csv.DictReader(csvfile).fieldnames, REQUIRED_SCENARIO_COLUMNS)
        with open(slot_path, newline="", encoding="utf-8") as csvfile:
            validate_columns(slot_path, csv.DictReader(csvfile).fieldnames, REQUIRED_SLOT_COLUMNS)
        scenario_rows.extend(attach_metadata(row, metadata, scenario_path) for row in scenario_records)
        slot_rows.extend(attach_metadata(row, metadata, slot_path) for row in slot_records)
    return scenario_rows, slot_rows


def to_float(row, field, default=math.nan):
    """Convert raw CSV values, booleans included, to float."""
    value = row.get(field, "")
    if value in ("", None):
        return float(default)
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text.lower() == "true":
        return 1.0
    if text.lower() == "false":
        return 0.0
    return float(text)


def finite_values(rows, field):
    """Return finite numeric values for a metric field."""
    values = []
    for row in rows:
        try:
            value = to_float(row, field)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean(values):
    """Return a finite mean or NaN for empty data."""
    return float(np.mean(values)) if values else math.nan


def bootstrap_ci(values, rng, samples):
    """Compute a nonparametric bootstrap CI for the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return math.nan, math.nan, "insufficient"
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = np.mean(values[draws], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), "ok"


def grouped(rows, key_fn):
    """Group rows by an arbitrary key function."""
    groups = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return groups


def scenario_key(row, include_seed=False):
    """Build a scenario key. Seed keys add run/episode identity for paired tests."""
    fields = [
        "config_name",
        "mismatch_model",
        "rho",
        "csi_delay",
        "K",
        "num_slots",
        "num_irs_elements",
        "num_codebook_states",
        "probe_budget",
        "feedback_noise_std",
    ]
    if include_seed:
        fields += ["run_seed", "episode_seed"]
    return tuple(row.get(field, "") for field in fields)


def scenario_label(row):
    """Human-readable scenario label for tables."""
    return (
        f"K={row.get('K')},S={row.get('num_slots')},C={row.get('num_codebook_states')},"
        f"M={row.get('num_irs_elements')},rho={row.get('rho')},delay={row.get('csi_delay')},"
        f"fb={row.get('feedback_noise_std')},B={row.get('probe_budget')}"
    )


def method_names(rows):
    """Return methods in first-seen order."""
    names = []
    seen = set()
    for row in rows:
        name = row["method_name"]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def metric_lookup():
    """Return metric metadata keyed by raw field name."""
    return {field: {"label": label, "direction": direction} for field, label, direction in METRICS}


def write_csv(path, rows, fieldnames):
    """Write a CSV with a stable schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def fmt(value, digits=6):
    """Format finite numeric values for CSV/Markdown/LaTeX."""
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def build_main_table(rows, rng, samples):
    """Build method-level mean and bootstrap CI rows."""
    output = []
    by_method = grouped(rows, lambda row: row["method_name"])
    for method in method_names(rows):
        method_rows = by_method[method]
        metadata = result_metadata({"name": method})
        for field, label, direction in METRICS:
            values = finite_values(method_rows, field)
            lo, hi, status = bootstrap_ci(values, rng, samples)
            output.append(
                {
                    "method_name": method,
                    "metric": field,
                    "metric_label": label,
                    "direction": direction,
                    "mean": fmt(mean(values)),
                    "ci95_low": fmt(lo),
                    "ci95_high": fmt(hi),
                    "sample_count": len(values),
                    "scenario_count": len({scenario_key(row) for row in method_rows}),
                    "seed_count": len({row.get("episode_seed", "") for row in method_rows}),
                    "ci_status": status,
                    **metadata,
                }
            )
    return output


def mean_by_pair(rows, metric):
    """Map seed-level paired keys to method metric values."""
    output = {}
    for row in rows:
        try:
            value = to_float(row, metric)
        except ValueError:
            continue
        if math.isfinite(value):
            output[(row["method_name"], scenario_key(row, include_seed=True))] = value
    return output


def find_baselines(methods, patterns):
    """Resolve baseline substrings to concrete method names."""
    baselines = []
    for pattern in patterns:
        matches = [method for method in methods if pattern in method]
        if matches:
            baselines.append(matches[0])
    return baselines


def paired_deltas(rows, baseline_patterns, rng, samples):
    """Compute paired bootstrap CIs for candidate minus baseline."""
    methods = method_names(rows)
    baselines = find_baselines(methods, baseline_patterns)
    metric_info = metric_lookup()
    output = []
    value_maps = {field: mean_by_pair(rows, field) for field, _label, _direction in METRICS}
    for baseline in baselines:
        for candidate in methods:
            if candidate == baseline:
                continue
            for field, _label, _direction in METRICS:
                value_map = value_maps[field]
                paired_keys = sorted(
                    key
                    for method, key in value_map
                    if method == baseline and (candidate, key) in value_map
                )
                deltas = [value_map[(candidate, key)] - value_map[(baseline, key)] for key in paired_keys]
                lo, hi, status = bootstrap_ci(deltas, rng, samples)
                output.append(
                    {
                        "baseline_method": baseline,
                        "candidate_method": candidate,
                        "metric": field,
                        "metric_label": metric_info[field]["label"],
                        "direction": metric_info[field]["direction"],
                        "paired_sample_count": len(deltas),
                        "mean_delta": fmt(mean(deltas)),
                        "ci95_low": fmt(lo),
                        "ci95_high": fmt(hi),
                        "ci_status": status,
                        "claim_status": claim_status(deltas, lo, hi, metric_info[field]["direction"], status),
                    }
                )
    return output


def claim_status(deltas, lo, hi, direction, ci_status):
    """Classify paired CI conservatively without overclaiming."""
    if ci_status != "ok" or len(deltas) < 2:
        return "insufficient_data"
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0.0 <= hi:
        return "inconclusive"
    if direction == "lower":
        return "candidate_better" if hi < 0.0 else "candidate_worse"
    return "candidate_better" if lo > 0.0 else "candidate_worse"


def per_scenario_deltas(rows, baseline_patterns):
    """Compute per-scenario paired mean deltas against key baselines."""
    methods = method_names(rows)
    baselines = find_baselines(methods, baseline_patterns)
    by_scenario_method = grouped(rows, lambda row: (scenario_key(row), row["method_name"]))
    scenario_representative = {}
    for row in rows:
        scenario_representative.setdefault(scenario_key(row), row)
    output = []
    for baseline in baselines:
        for candidate in methods:
            if candidate == baseline:
                continue
            scenario_keys = sorted(
                key
                for key in scenario_representative
                if (key, baseline) in by_scenario_method and (key, candidate) in by_scenario_method
            )
            for key in scenario_keys:
                rep = scenario_representative[key]
                row = {
                    "baseline_method": baseline,
                    "candidate_method": candidate,
                    "scenario": scenario_label(rep),
                }
                for field, _label, _direction in METRICS:
                    base_mean = mean(finite_values(by_scenario_method[(key, baseline)], field))
                    cand_mean = mean(finite_values(by_scenario_method[(key, candidate)], field))
                    row[f"{field}_baseline_mean"] = fmt(base_mean)
                    row[f"{field}_candidate_mean"] = fmt(cand_mean)
                    row[f"{field}_delta"] = fmt(cand_mean - base_mean)
                output.append(row)
    return output


def win_tie_loss(rows, baseline_patterns):
    """Count scenario-level wins/ties/losses for each paired comparison."""
    deltas = per_scenario_deltas(rows, baseline_patterns)
    output = []
    for baseline, candidate, metric in sorted(
        {
            (row["baseline_method"], row["candidate_method"], field)
            for row in deltas
            for field, _label, _direction in METRICS
        }
    ):
        info = metric_lookup()[metric]
        metric_deltas = [
            float(row[f"{metric}_delta"])
            for row in deltas
            if row["baseline_method"] == baseline
            and row["candidate_method"] == candidate
            and row.get(f"{metric}_delta", "") != ""
        ]
        wins = ties = losses = 0
        for delta in metric_deltas:
            if abs(delta) <= TIE_TOLERANCE:
                ties += 1
            elif (delta < 0.0 and info["direction"] == "lower") or (
                delta > 0.0 and info["direction"] == "higher"
            ):
                wins += 1
            else:
                losses += 1
        output.append(
            {
                "baseline_method": baseline,
                "candidate_method": candidate,
                "metric": metric,
                "metric_label": info["label"],
                "direction": info["direction"],
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "scenario_count": len(metric_deltas),
            }
        )
    return output


def deployable_oracle_table(main_rows):
    """Build a compact deployable-vs-diagnostic table from main metric rows."""
    selected_metrics = {
        "completion": "completion_mean",
        "slots_used": "slots_mean",
        "oracle_gap": "oracle_gap_mean",
        "nmse": "nmse_mean",
        "total_protocol_cost": "protocol_cost_mean",
    }
    by_method = {}
    for row in main_rows:
        method = row["method_name"]
        by_method.setdefault(
            method,
            {
                "method_name": method,
                "result_role": row["result_role"],
                "inference_uses_hidden_current_csi": row["inference_uses_hidden_current_csi"],
                "uses_hidden_training_labels": row["uses_hidden_training_labels"],
            },
        )
        if row["metric"] in selected_metrics:
            by_method[method][selected_metrics[row["metric"]]] = row["mean"]
    return list(by_method.values())


def main_metric_wide(main_rows):
    """Create a wide table useful for plotting and Markdown summaries."""
    output = {}
    for row in main_rows:
        method = row["method_name"]
        output.setdefault(method, {"method_name": method, "result_role": row["result_role"]})
        output[method][row["metric"]] = row["mean"]
        output[method][f"{row['metric']}_ci95_low"] = row["ci95_low"]
        output[method][f"{row['metric']}_ci95_high"] = row["ci95_high"]
    return list(output.values())


def sort_methods_for_figures(rows):
    """Keep figures readable by limiting to the most relevant methods."""
    methods = method_names(rows)
    if len(methods) <= FIGURE_METHOD_LIMIT:
        return methods
    priority_words = (
        "Coverage-Aware",
        "Count-Only",
        "Count-Conditioned",
        "Posterior-Greedy",
        "Random Same-Budget",
        "Full Current Oracle",
    )
    selected = []
    for word in priority_words:
        selected.extend([method for method in methods if word in method and method not in selected])
    selected.extend(method for method in methods if method not in selected)
    return selected[:FIGURE_METHOD_LIMIT]


def setup_axes(ax):
    """Apply a simple reproducible plotting style."""
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#374151", labelsize=8)


def save_figure(fig, path):
    """Save PNG/PDF siblings for a figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.with_suffix('.png')}")
    print(f"Saved: {path.with_suffix('.pdf')}")


def analysis_title(analysis_name, rows):
    """Return a compact figure title suffix with config, seed and method counts."""
    seeds = len({row.get("episode_seed", "") for row in rows})
    scenarios = len({scenario_key(row) for row in rows})
    configs = len({row.get("config_name", "") for row in rows})
    methods = len({row.get("method_name", "") for row in rows})
    return f"{analysis_name} | configs={configs}, scenarios={scenarios}, seeds={seeds}, methods={methods}"


def plot_frontier(rows, output_dir, analysis_name, y_metric, y_label, filename):
    """Plot protocol cost against a selected quality metric."""
    by_method = grouped(rows, lambda row: row["method_name"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for method in sort_methods_for_figures(rows):
        method_rows = by_method[method]
        x = mean(finite_values(method_rows, "total_protocol_cost"))
        y = mean(finite_values(method_rows, y_metric))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        role = result_metadata({"name": method})["result_role"]
        marker = "X" if "oracle" in role else "o"
        ax.scatter(x, y, marker=marker, s=80)
        ax.annotate(short_method(method), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Total protocol cost")
    ax.set_ylabel(y_label)
    ax.set_title(f"{filename.replace('_', ' ')}\n{analysis_title(analysis_name, rows)}", fontsize=10)
    setup_axes(ax)
    save_figure(fig, output_dir / filename)


def short_method(method):
    """Compact method label for plots."""
    replacements = {
        "Coverage-Aware Sparse-TopK Feedback Grid": "Coverage",
        "Count-Only Mask-Corrected Coverage-Aware Grid": "CountOnly",
        "Count-Conditioned Invitation Feedback Grid": "PosteriorInv",
        "Posterior-Greedy Probing + Count-Conditioned Invitation Grid": "PostProbe+Inv",
        "Posterior-Greedy Probing Feedback Grid": "PostProbe",
        "Random Same-Budget Feedback Grid": "Random",
        "Rotating Same-Budget Feedback Grid": "Rotating",
        "Sparse-TopK Same-Budget Feedback Grid": "Sparse",
        "Full Current Oracle": "Oracle",
        "Full Stale Exhaustive": "FullStale",
        "Estimated No IRS": "NoIRS",
    }
    label = method
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label[:28]


def plot_codebook_scaling(rows, output_dir, analysis_name):
    """Plot oracle gap against codebook size."""
    methods = sort_methods_for_figures(rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for method in methods:
        points = []
        for codebook, group_rows in grouped(
            [row for row in rows if row["method_name"] == method],
            lambda row: int(float(row["num_codebook_states"])),
        ).items():
            points.append((codebook, mean(finite_values(group_rows, "oracle_gap"))))
        if not points:
            continue
        points.sort()
        ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=short_method(method))
    ax.set_xlabel("Codebook size C")
    ax.set_ylabel("Oracle gap")
    ax.set_title(f"Codebook scaling\n{analysis_title(analysis_name, rows)}", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    setup_axes(ax)
    save_figure(fig, output_dir / "codebook_size_scaling")


def plot_feedback_noise(rows, output_dir, analysis_name):
    """Plot oracle gap against aggregate feedback noise."""
    methods = sort_methods_for_figures(rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for method in methods:
        points = []
        method_rows = [row for row in rows if row["method_name"] == method]
        for noise, group_rows in grouped(method_rows, lambda row: to_float(row, "feedback_noise_std", 0.0)).items():
            if math.isfinite(noise):
                points.append((noise, mean(finite_values(group_rows, "oracle_gap"))))
        if not points:
            continue
        points.sort()
        ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=short_method(method))
    ax.set_xlabel("Aggregate feedback noise std")
    ax.set_ylabel("Oracle gap")
    ax.set_title(f"Feedback-noise robustness\n{analysis_title(analysis_name, rows)}", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    setup_axes(ax)
    save_figure(fig, output_dir / "feedback_noise_robustness")


def plot_rho_delay_heatmap(rows, output_dir, analysis_name):
    """Plot a rho/delay heatmap for best deployable mean oracle gap."""
    deployable = [
        row
        for row in rows
        if result_metadata({"name": row["method_name"]})["inference_uses_hidden_current_csi"] == "false"
    ]
    groups = grouped(deployable, lambda row: (to_float(row, "rho"), int(float(row["csi_delay"]))))
    rhos = sorted({key[0] for key in groups if math.isfinite(key[0])})
    delays = sorted({key[1] for key in groups})
    if not rhos or not delays:
        return
    matrix = np.full((len(rhos), len(delays)), np.nan)
    for i, rho in enumerate(rhos):
        for j, delay in enumerate(delays):
            scenario_rows = groups.get((rho, delay), [])
            per_method = [
                mean(finite_values(group_rows, "oracle_gap"))
                for group_rows in grouped(scenario_rows, lambda row: row["method_name"]).values()
            ]
            finite = [value for value in per_method if math.isfinite(value)]
            if finite:
                matrix[i, j] = min(finite)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(delays)), [str(delay) for delay in delays])
    ax.set_yticks(range(len(rhos)), [f"{rho:g}" for rho in rhos])
    ax.set_xlabel("CSI delay")
    ax.set_ylabel("rho")
    ax.set_title(f"Best deployable oracle-gap heatmap\n{analysis_title(analysis_name, rows)}", fontsize=10)
    for i in range(len(rhos)):
        for j in range(len(delays)):
            if math.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Oracle gap")
    save_figure(fig, output_dir / "rho_delay_robustness_heatmap")


def plot_invitation_mismatch(rows, output_dir, analysis_name):
    """Plot failed/missed/gap components as a visible invitation-mismatch diagnostic."""
    methods = sort_methods_for_figures(rows)
    x = np.arange(len(methods))
    failed = [mean(finite_values([row for row in rows if row["method_name"] == method], "failed_invitations")) for method in methods]
    missed = [mean(finite_values([row for row in rows if row["method_name"] == method], "missed_opportunities")) for method in methods]
    gap = [mean(finite_values([row for row in rows if row["method_name"] == method], "oracle_gap")) for method in methods]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x, failed, label="Failed invitations")
    ax.bar(x, missed, bottom=failed, label="Missed opportunities")
    ax.plot(x, gap, color="black", marker="o", linewidth=1.2, label="Oracle gap")
    ax.set_xticks(x, [short_method(method) for method in methods], rotation=35, ha="right")
    ax.set_ylabel("Count / gap")
    ax.set_title(f"Invitation mismatch diagnostics\n{analysis_title(analysis_name, rows)}", fontsize=10)
    ax.legend(fontsize=8)
    setup_axes(ax)
    save_figure(fig, output_dir / "invitation_mismatch_decomposition")


def plot_failed_missed(rows, output_dir, analysis_name):
    """Plot failed invitation vs missed opportunity tradeoff."""
    by_method = grouped(rows, lambda row: row["method_name"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for method in sort_methods_for_figures(rows):
        method_rows = by_method[method]
        x = mean(finite_values(method_rows, "failed_invitations"))
        y = mean(finite_values(method_rows, "missed_opportunities"))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        ax.scatter(x, y, s=80)
        ax.annotate(short_method(method), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Failed invitations")
    ax.set_ylabel("Missed opportunities")
    ax.set_title(f"Failed vs missed tradeoff\n{analysis_title(analysis_name, rows)}", fontsize=10)
    setup_axes(ax)
    save_figure(fig, output_dir / "failed_vs_missed_tradeoff")


def write_latex_table(path, rows):
    """Write a small LaTeX snippet for the main table."""
    selected_metrics = ["Completion", "Slots", "OracleGap", "NMSE", "ProtocolCost"]
    wide = main_metric_wide(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as texfile:
        texfile.write("% Generated by experiments/analyze_paper_experiment_logs.py\n")
        texfile.write("\\begin{tabular}{lrrrrr}\n")
        texfile.write("\\toprule\n")
        texfile.write("Method & Completion & Slots & OracleGap & NMSE & ProtocolCost \\\\\n")
        texfile.write("\\midrule\n")
        for row in wide:
            values = [row.get(metric_key(metric), "") for metric in selected_metrics]
            texfile.write(
                f"{escape_latex(short_method(row['method_name']))} & "
                + " & ".join(value or "--" for value in values)
                + " \\\\\n"
            )
        texfile.write("\\bottomrule\n")
        texfile.write("\\end{tabular}\n")
    print(f"Saved: {path}")


def metric_key(label):
    """Map display labels back to metric field names."""
    for field, metric_label, _direction in METRICS:
        if metric_label == label:
            return field
    return label


def escape_latex(text):
    """Escape minimal LaTeX special characters."""
    return str(text).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def write_markdown(path, analysis_name, scenario_rows, main_rows, paired_rows, generated_files):
    """Write a short interpretation summary."""
    insufficient = sum(1 for row in paired_rows if row["claim_status"] == "insufficient_data")
    inconclusive = sum(1 for row in paired_rows if row["claim_status"] == "inconclusive")
    better = sum(1 for row in paired_rows if row["claim_status"] == "candidate_better")
    methods = method_names(scenario_rows)
    scenarios = len({scenario_key(row) for row in scenario_rows})
    seeds = len({row.get("episode_seed", "") for row in scenario_rows})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as md:
        md.write(f"# Paper Suite Raw-Log Analysis: {analysis_name}\n\n")
        md.write("This analysis consumes raw `structured_logs/scenario_summary.csv` and `slot_records.csv` files, not pre-averaged summary tables.\n\n")
        md.write(f"- Methods: {len(methods)}\n")
        md.write(f"- Scenarios: {scenarios}\n")
        md.write(f"- Episode seeds: {seeds}\n")
        md.write(f"- Paired comparisons marked candidate_better: {better}\n")
        md.write(f"- Paired comparisons inconclusive: {inconclusive}\n")
        md.write(f"- Paired comparisons with insufficient data: {insufficient}\n\n")
        if seeds < 2:
            md.write("Caution: fewer than two paired samples are available for at least this smoke-scale run, so significance claims are disabled.\n\n")
        md.write("## Generated Files\n\n")
        for file_path in generated_files:
            md.write(f"- `{file_path}`\n")
        md.write("\n## Method Roles\n\n")
        for row in deployable_oracle_table(main_rows):
            md.write(
                f"- `{row['method_name']}`: role={row['result_role']}, "
                f"hidden_current={row['inference_uses_hidden_current_csi']}\n"
            )
    print(f"Saved: {path}")


def figure_point_csv(rows, output_dir):
    """Write raw points used by main figures."""
    point_rows = []
    for method, method_rows in grouped(rows, lambda row: row["method_name"]).items():
        point_rows.append(
            {
                "method_name": method,
                "short_method": short_method(method),
                "result_role": result_metadata({"name": method})["result_role"],
                "protocol_cost": fmt(mean(finite_values(method_rows, "total_protocol_cost"))),
                "nmse": fmt(mean(finite_values(method_rows, "nmse"))),
                "oracle_gap": fmt(mean(finite_values(method_rows, "oracle_gap"))),
                "failed_invitations": fmt(mean(finite_values(method_rows, "failed_invitations"))),
                "missed_opportunities": fmt(mean(finite_values(method_rows, "missed_opportunities"))),
                "codebook_size_mean": fmt(mean(finite_values(method_rows, "num_codebook_states"))),
                "feedback_noise_std_mean": fmt(mean(finite_values(method_rows, "feedback_noise_std"))),
            }
        )
    path = output_dir / "figure_points.csv"
    write_csv(
        path,
        point_rows,
        [
            "method_name",
            "short_method",
            "result_role",
            "protocol_cost",
            "nmse",
            "oracle_gap",
            "failed_invitations",
            "missed_opportunities",
            "codebook_size_mean",
            "feedback_noise_std_mean",
        ],
    )
    return path


def main():
    """Analyze a suite run directory."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    analysis_name = args.analysis_name or input_dir.name
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.bootstrap_seed)
    scenario_rows, slot_rows = load_raw_logs(input_dir)
    baseline_patterns = csv_items(args.baseline_methods)

    main_rows = build_main_table(scenario_rows, rng, args.bootstrap_samples)
    paired_rows = paired_deltas(scenario_rows, baseline_patterns, rng, args.bootstrap_samples)
    scenario_delta_rows = per_scenario_deltas(scenario_rows, baseline_patterns)
    wtl_rows = win_tie_loss(scenario_rows, baseline_patterns)
    role_rows = deployable_oracle_table(main_rows)
    wide_rows = main_metric_wide(main_rows)

    generated = []
    main_path = output_dir / "main_metrics_bootstrap.csv"
    write_csv(
        main_path,
        main_rows,
        [
            "method_name",
            "metric",
            "metric_label",
            "direction",
            "mean",
            "ci95_low",
            "ci95_high",
            "sample_count",
            "scenario_count",
            "seed_count",
            "ci_status",
            "result_role",
            "uses_hidden_training_labels",
            "inference_uses_hidden_current_csi",
            "supervision_signal",
        ],
    )
    generated.append(main_path)

    wide_path = output_dir / "main_metrics_wide.csv"
    wide_fields = sorted({field for row in wide_rows for field in row})
    write_csv(wide_path, wide_rows, wide_fields)
    generated.append(wide_path)

    paired_path = output_dir / "paired_bootstrap_deltas.csv"
    write_csv(
        paired_path,
        paired_rows,
        [
            "baseline_method",
            "candidate_method",
            "metric",
            "metric_label",
            "direction",
            "paired_sample_count",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "ci_status",
            "claim_status",
        ],
    )
    generated.append(paired_path)

    scenario_delta_path = output_dir / "per_scenario_paired_deltas.csv"
    scenario_fields = sorted({field for row in scenario_delta_rows for field in row})
    write_csv(scenario_delta_path, scenario_delta_rows, scenario_fields)
    generated.append(scenario_delta_path)

    wtl_path = output_dir / "win_tie_loss_counts.csv"
    write_csv(
        wtl_path,
        wtl_rows,
        [
            "baseline_method",
            "candidate_method",
            "metric",
            "metric_label",
            "direction",
            "wins",
            "ties",
            "losses",
            "scenario_count",
        ],
    )
    generated.append(wtl_path)

    role_path = output_dir / "deployable_vs_oracle_diagnostics.csv"
    role_fields = sorted({field for row in role_rows for field in row})
    write_csv(role_path, role_rows, role_fields)
    generated.append(role_path)

    slot_path = output_dir / "slot_invitation_mismatch_summary.csv"
    slot_summary = build_slot_summary(slot_rows)
    slot_fields = sorted({field for row in slot_summary for field in row})
    write_csv(slot_path, slot_summary, slot_fields)
    generated.append(slot_path)

    figure_points_path = figure_point_csv(scenario_rows, output_dir)
    generated.append(figure_points_path)

    tex_path = output_dir / "main_metrics_table.tex"
    write_latex_table(tex_path, main_rows)
    generated.append(tex_path)

    figures_dir = output_dir / "figures"
    plot_frontier(scenario_rows, figures_dir, analysis_name, "nmse", "NMSE", "cost_vs_nmse_frontier")
    plot_frontier(scenario_rows, figures_dir, analysis_name, "oracle_gap", "Oracle gap", "cost_vs_oracle_gap_frontier")
    plot_codebook_scaling(scenario_rows, figures_dir, analysis_name)
    plot_feedback_noise(scenario_rows, figures_dir, analysis_name)
    plot_rho_delay_heatmap(scenario_rows, figures_dir, analysis_name)
    plot_invitation_mismatch(scenario_rows, figures_dir, analysis_name)
    plot_failed_missed(scenario_rows, figures_dir, analysis_name)
    generated.extend(sorted(figures_dir.glob("*.png")))
    generated.extend(sorted(figures_dir.glob("*.pdf")))

    md_path = output_dir / "interpretation.md"
    write_markdown(md_path, analysis_name, scenario_rows, main_rows, paired_rows, generated)
    print(f"Analysis complete: {output_dir}")
    return 0


def build_slot_summary(slot_rows):
    """Aggregate slot-level invitation mismatch diagnostics."""
    output = []
    for method, rows in grouped(slot_rows, lambda row: row["method_name"]).items():
        output.append(
            {
                "method_name": method,
                "slot_count": len(rows),
                "failed_invited_mean": fmt(mean(finite_values(rows, "failed_invited_count"))),
                "missed_feasible_mean": fmt(mean(finite_values(rows, "missed_feasible_count"))),
                "slot_oracle_gap_mean": fmt(mean(finite_values(rows, "oracle_gap"))),
                "slot_nmse_mean": fmt(mean(finite_values(rows, "nmse"))),
                "slot_protocol_cost_mean": fmt(mean(finite_values(rows, "total_protocol_cost"))),
                "result_role": result_metadata({"name": method})["result_role"],
            }
        )
    return output


if __name__ == "__main__":
    raise SystemExit(main())
