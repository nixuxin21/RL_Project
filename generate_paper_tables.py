"""从冻结结果 CSV 生成论文表 1、表 2、表 3 及不确定性 companion。"""

import argparse
import csv
import math
from pathlib import Path


DEFAULT_SOURCE = Path("results/execution_mismatch/final_execution_baseline_summary.csv")
DEFAULT_MAIN_FRONTIER = Path("results/execution_mismatch/main_frontier_analysis.csv")
DEFAULT_TABLE1_CSV = Path("results/paper/table1_main_results.csv")
DEFAULT_UNCERTAINTY_CSV = Path("results/paper/table1_scenario_uncertainty.csv")
DEFAULT_PAIRED_CSV = Path("results/paper/table1_paired_scenario_deltas.csv")
DEFAULT_TABLE1_MD = Path("docs/PAPER_TABLE1_MAIN_RESULTS.md")
DEFAULT_UNCERTAINTY_MD = Path("docs/PAPER_TABLE1_UNCERTAINTY.md")
DEFAULT_COVERAGE_SOURCE = Path("results/execution_mismatch/coverage_aware_ablation_analysis.csv")
DEFAULT_FAILURE_DIAGNOSIS_SOURCE = Path(
    "results/execution_mismatch/"
    "coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_"
    "b3_sm4p1_tf0p75_cw0p5_cpw0_summary.csv"
)
DEFAULT_TABLE2_CSV = Path("results/paper/table2_coverage_aware_ablation.csv")
DEFAULT_TABLE2_MD = Path("docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md")
DEFAULT_TABLE3_CSV = Path("results/paper/table3_failure_diagnosis.csv")
DEFAULT_TABLE3_MD = Path("docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md")

TABLE1_METHOD_ORDER = [
    "Rotating B=8",
    "Sparse-TopK B=4 sm=3",
    "Coverage-Aware B=4 cw=0.5 cpw=0",
    "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0",
    "Mask-Corrected Coverage-Aware B=3 mc=1",
    "Stale-TopK B=4",
    "Temporal Deviation Oracle B=4",
]

TABLE1_COLUMNS = ["Method", "Role", "Slots", "Failed", "Missed", "Preview", "Gap"]
UNCERTAINTY_COLUMNS = [
    "Method",
    "Metric",
    "Mean",
    "ScenarioStd",
    "ScenarioSE",
    "ScenarioCI95",
    "ScenarioCount",
    "SamplesPerScenario",
    "Seeds",
]
PAIRED_COLUMNS = [
    "Comparison",
    "Scope",
    "Metric",
    "Baseline",
    "Candidate",
    "BaselineMean",
    "CandidateMean",
    "MeanDelta",
    "DeltaScenarioStd",
    "DeltaScenarioSE",
    "DeltaScenarioCI95",
    "ImprovedScenarios",
    "ScenarioCount",
]
TABLE2_COLUMNS = [
    "Section",
    "Setting",
    "Scenarios",
    "Samples",
    "Seeds",
    "Slots",
    "Failed",
    "Missed",
    "Preview",
    "Gap",
    "DeltaSlotsVsSparse",
    "DeltaMissedVsSparse",
    "DeltaGapVsSparse",
    "Role",
]
TABLE3_COLUMNS = [
    "Scope",
    "TraceSlots",
    "GapSlots",
    "AvgGap",
    "PoolShare",
    "SelectionShare",
    "ConfirmationShare",
    "InvitationShare",
    "DominantComponent",
    "OracleNotInSeedRate",
    "Failed",
    "Missed",
]

METRIC_SPECS = [
    ("Slots", "slots_mean"),
    ("Failed", "failed_nodes_mean"),
    ("Missed", "missed_opportunities_mean"),
    ("Gap", "oracle_tx_gap_mean"),
]

PAIRED_COMPARISONS = [
    {
        "name": "Mask-Corrected vs Coverage-Aware B=3",
        "scope": "same-preview direct ablation",
        "baseline": "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0",
        "candidate": "Mask-Corrected Coverage-Aware B=3 mc=1",
    },
    {
        "name": "Mask-Corrected vs Coverage-Aware B=4",
        "scope": "same-preview coverage reference",
        "baseline": "Coverage-Aware B=4 cw=0.5 cpw=0",
        "candidate": "Mask-Corrected Coverage-Aware B=3 mc=1",
    },
    {
        "name": "Mask-Corrected vs Sparse-TopK B=4",
        "scope": "same-preview sparse baseline",
        "baseline": "Sparse-TopK B=4 sm=3",
        "candidate": "Mask-Corrected Coverage-Aware B=3 mc=1",
    },
]

T_CRIT_975_BY_DF = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def row_by_label(rows, label):
    matches = [row for row in rows if row.get("label") == label]
    if not matches:
        raise ValueError(f"missing required Table 1 method: {label}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous Table 1 method: {label}")
    return matches[0]


def as_float(row, field):
    return float(row[field])


def scenario_key(row):
    return row["scenario_key"]


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def ci95(values):
    if len(values) <= 1:
        return 0.0
    sem = sample_std(values) / math.sqrt(len(values))
    t_crit = T_CRIT_975_BY_DF.get(len(values) - 1, 1.96)
    return t_crit * sem


def format_number(value, digits=6):
    return f"{float(value):.{digits}f}"


def format_row(row):
    return {
        "Method": row["label"],
        "Role": row["role"],
        "Slots": f"{float(row['slots_mean']):.3f}",
        "Failed": f"{float(row['failed_nodes_mean']):.3f}",
        "Missed": f"{float(row['missed_opportunities_mean']):.3f}",
        "Preview": f"{float(row['decision_preview_calls_per_slot_mean']):.2f}",
        "Gap": f"{float(row['oracle_tx_gap_mean']):.3f}",
    }


def build_table1(source_rows):
    return [format_row(row_by_label(source_rows, label)) for label in TABLE1_METHOD_ORDER]


def average_rows(rows, fields):
    averaged = {}
    for field in fields:
        values = [as_float(row, field) for row in rows]
        averaged[field] = sum(values) / len(values)
    return averaged


def build_table2_rows(source_rows, coverage_rows):
    """构建table2、结果行所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    output_rows = []
    baseline = row_by_label(source_rows, "Sparse-TopK B=4 sm=3")
    coverage_b4 = row_by_label(source_rows, "Coverage-Aware B=4 cw=0.5 cpw=0")
    coverage_b3 = row_by_label(source_rows, "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0")
    anchor_rows = [
        (
            "Main calibration",
            baseline,
            "same-preview sparse baseline",
            {
                "slots_mean": 0.0,
                "missed_opportunities_mean": 0.0,
                "oracle_tx_gap_mean": 0.0,
            },
        ),
        (
            "Main calibration",
            coverage_b4,
            "selected B=4 coverage reference",
            {
                "slots_mean": as_float(coverage_b4, "slots_mean") - as_float(baseline, "slots_mean"),
                "missed_opportunities_mean": (
                    as_float(coverage_b4, "missed_opportunities_mean")
                    - as_float(baseline, "missed_opportunities_mean")
                ),
                "oracle_tx_gap_mean": (
                    as_float(coverage_b4, "oracle_tx_gap_mean")
                    - as_float(baseline, "oracle_tx_gap_mean")
                ),
            },
        ),
        (
            "Main calibration",
            coverage_b3,
            "selected same-preview budget split",
            {
                "slots_mean": as_float(coverage_b3, "slots_mean") - as_float(baseline, "slots_mean"),
                "missed_opportunities_mean": (
                    as_float(coverage_b3, "missed_opportunities_mean")
                    - as_float(baseline, "missed_opportunities_mean")
                ),
                "oracle_tx_gap_mean": (
                    as_float(coverage_b3, "oracle_tx_gap_mean")
                    - as_float(baseline, "oracle_tx_gap_mean")
                ),
            },
        ),
    ]
    for section, row, role, deltas in anchor_rows:
        output_rows.append(
            {
                "Section": section,
                "Setting": row["label"],
                "Scenarios": row["scenario_count"],
                "Samples": row["episodes_per_scenario"],
                "Seeds": row["num_seeds"],
                "Slots": f"{as_float(row, 'slots_mean'):.3f}",
                "Failed": f"{as_float(row, 'failed_nodes_mean'):.3f}",
                "Missed": f"{as_float(row, 'missed_opportunities_mean'):.3f}",
                "Preview": f"{as_float(row, 'decision_preview_calls_per_slot_mean'):.2f}",
                "Gap": f"{as_float(row, 'oracle_tx_gap_mean'):.3f}",
                "DeltaSlotsVsSparse": f"{deltas['slots_mean']:.3f}",
                "DeltaMissedVsSparse": f"{deltas['missed_opportunities_mean']:.3f}",
                "DeltaGapVsSparse": f"{deltas['oracle_tx_gap_mean']:.3f}",
                "Role": role,
            }
        )

    coverage_grouped = rows_by_label(coverage_rows)
    for label in sorted(coverage_grouped):
        rows = coverage_grouped[label]
        averaged = average_rows(
            rows,
            [
                "episodes",
                "num_seeds",
                "slots_mean",
                "failed_nodes_mean",
                "missed_opportunities_mean",
                "decision_preview_calls_per_slot_mean",
                "oracle_tx_gap_mean",
                "slots_delta_vs_sparse",
                "missed_delta_vs_sparse",
                "gap_delta_vs_sparse",
            ],
        )
        output_rows.append(
            {
                "Section": "B=4 coverage-weight ablation",
                "Setting": label,
                "Scenarios": str(len(rows)),
                "Samples": str(int(round(averaged["episodes"]))),
                "Seeds": str(int(round(averaged["num_seeds"]))),
                "Slots": f"{averaged['slots_mean']:.3f}",
                "Failed": f"{averaged['failed_nodes_mean']:.3f}",
                "Missed": f"{averaged['missed_opportunities_mean']:.3f}",
                "Preview": f"{averaged['decision_preview_calls_per_slot_mean']:.2f}",
                "Gap": f"{averaged['oracle_tx_gap_mean']:.3f}",
                "DeltaSlotsVsSparse": f"{averaged['slots_delta_vs_sparse']:.3f}",
                "DeltaMissedVsSparse": f"{averaged['missed_delta_vs_sparse']:.3f}",
                "DeltaGapVsSparse": f"{averaged['gap_delta_vs_sparse']:.3f}",
                "Role": "coverage-weight sensitivity row",
            }
        )
    return output_rows


def build_table3_rows(diagnosis_rows):
    """构建table3、结果行所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    output_rows = []
    for row in diagnosis_rows:
        shares = {
            "pool": as_float(row, "pool_gap_share"),
            "selection": as_float(row, "selection_gap_share"),
            "confirmation": as_float(row, "confirmation_gap_share"),
            "invitation": as_float(row, "invitation_gap_share"),
        }
        dominant = max(shares, key=shares.get)
        output_rows.append(
            {
                "Scope": row["label"],
                "TraceSlots": str(int(as_float(row, "trace_slots"))),
                "GapSlots": str(int(as_float(row, "slots_with_gap"))),
                "AvgGap": f"{as_float(row, 'avg_total_gap'):.3f}",
                "PoolShare": f"{shares['pool']:.3f}",
                "SelectionShare": f"{shares['selection']:.3f}",
                "ConfirmationShare": f"{shares['confirmation']:.3f}",
                "InvitationShare": f"{shares['invitation']:.3f}",
                "DominantComponent": dominant,
                "OracleNotInSeedRate": f"{as_float(row, 'oracle_not_in_seed_rate'):.3f}",
                "Failed": f"{as_float(row, 'avg_failed'):.3f}",
                "Missed": f"{as_float(row, 'avg_missed'):.3f}",
            }
        )
    return output_rows


def rows_by_label(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    for label, label_rows in grouped.items():
        grouped[label] = sorted(label_rows, key=scenario_key)
    return grouped


def build_uncertainty_rows(frontier_rows):
    grouped = rows_by_label(frontier_rows)
    output_rows = []
    for label in TABLE1_METHOD_ORDER:
        label_rows = grouped.get(label, [])
        if len(label_rows) != 9:
            raise ValueError(f"{label!r} should have 9 scenario rows, found {len(label_rows)}")
        first = label_rows[0]
        for metric_name, source_field in METRIC_SPECS:
            values = [as_float(row, source_field) for row in label_rows]
            output_rows.append(
                {
                    "Method": label,
                    "Metric": metric_name,
                    "Mean": format_number(sum(values) / len(values)),
                    "ScenarioStd": format_number(sample_std(values)),
                    "ScenarioSE": format_number(sample_std(values) / math.sqrt(len(values))),
                    "ScenarioCI95": format_number(ci95(values)),
                    "ScenarioCount": str(len(values)),
                    "SamplesPerScenario": str(int(as_float(first, "episodes"))),
                    "Seeds": str(int(as_float(first, "num_seeds"))),
                }
            )
    return output_rows


def rows_by_scenario(label_rows):
    return {scenario_key(row): row for row in label_rows}


def build_paired_delta_rows(frontier_rows):
    grouped = rows_by_label(frontier_rows)
    output_rows = []
    for comparison in PAIRED_COMPARISONS:
        baseline_rows = rows_by_scenario(grouped[comparison["baseline"]])
        candidate_rows = rows_by_scenario(grouped[comparison["candidate"]])
        scenario_keys = sorted(baseline_rows)
        if scenario_keys != sorted(candidate_rows):
            raise ValueError(f"paired comparison scenarios do not match: {comparison['name']}")
        for metric_name, source_field in METRIC_SPECS:
            baseline_values = [
                as_float(baseline_rows[key], source_field) for key in scenario_keys
            ]
            candidate_values = [
                as_float(candidate_rows[key], source_field) for key in scenario_keys
            ]
            deltas = [
                candidate_value - baseline_value
                for baseline_value, candidate_value in zip(
                    baseline_values,
                    candidate_values,
                    strict=True,
                )
            ]
            output_rows.append(
                {
                    "Comparison": comparison["name"],
                    "Scope": comparison["scope"],
                    "Metric": metric_name,
                    "Baseline": comparison["baseline"],
                    "Candidate": comparison["candidate"],
                    "BaselineMean": format_number(sum(baseline_values) / len(baseline_values)),
                    "CandidateMean": format_number(sum(candidate_values) / len(candidate_values)),
                    "MeanDelta": format_number(sum(deltas) / len(deltas)),
                    "DeltaScenarioStd": format_number(sample_std(deltas)),
                    "DeltaScenarioSE": format_number(sample_std(deltas) / math.sqrt(len(deltas))),
                    "DeltaScenarioCI95": format_number(ci95(deltas)),
                    "ImprovedScenarios": str(sum(delta < 0.0 for delta in deltas)),
                    "ScenarioCount": str(len(deltas)),
                }
            )
    return output_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=TABLE1_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_named_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    lines = [
        "| Method | Role | Slots | Failed | Missed | Preview | Gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['Method']}`",
                    row["Role"],
                    row["Slots"],
                    row["Failed"],
                    row["Missed"],
                    row["Preview"],
                    row["Gap"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown(path, rows, source_path, csv_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Paper Table 1 Main Results

This file is generated by `generate_paper_tables.py`. Do not edit table values by hand; update the source CSV or generator instead.

Source: `{source_path}`

Generated CSV: `{csv_path}`

## Table

{markdown_table(rows)}

## Caption

Table 1. Main execution-mismatch results averaged over 9 temporal AR(1) stale-CSI scenarios. All non-oracle rows are deployable under the stated stale-CSI and aggregate-feedback information model. The mask-corrected coverage-aware method keeps the same preview budget as the uncorrected B3 baseline and reduces slots, failed invitations, and missed opportunities, but it increases the no-noise oracle gap under the corrected temporal prehistory model. The temporal-deviation oracle is a hidden-information temporal diagnostic reference, not a deployable method or a global upper bound. This compact table reports means only; use the companion scenario-level uncertainty table for variability and paired-delta evidence.

## Notes

- Method order follows `docs/PAPER_FIGURE_TABLE_SPECS.md`.
- Information-role taxonomy follows `docs/PAPER_FIGURE_TABLE_SPECS.md`: deployable methods use no hidden current-channel device-level CSI at decision time; diagnostics are appendix/supplement-only; hidden-information temporal diagnostics are non-deployable headroom/reference rows only.
- All Table 1 rows except `Temporal Deviation Oracle B=4` are deployable comparisons or references under the stale-CSI / aggregate-feedback assumptions.
- `Temporal Deviation Oracle B=4` is a hidden-information temporal diagnostic reference because it selects the temporal deviation using current-channel outcomes unavailable to a deployable policy. It is not a global upper bound for methods that also alter invitation-mask correction.
- `Rotating B=4`, Adaptive V2 continuum points and learned variants are excluded from the paper-facing main table. Learned variants are diagnostics because their training labels may use hidden current-channel supervision even when closed-loop inference is deployable.
- `Perfect %` is omitted from the compact main table because it is near-saturated; failed, missed, preview and gap carry the main mechanism evidence.
- The compact main table is mean-only. Companion artifacts are generated at `docs/PAPER_TABLE1_UNCERTAINTY.md`, `results/paper/table1_scenario_uncertainty.csv`, and `results/paper/table1_paired_scenario_deltas.csv`.
- The companion uncertainty is scenario-level: each scenario row has already been averaged over the available run seeds. Do not describe it as a full seed-level significance test.
"""
    path.write_text(content, encoding="utf-8")


def compact_markdown_table(headers, rows, max_rows=None):
    selected_rows = rows if max_rows is None else rows[:max_rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in selected_rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def write_table2_markdown(path, rows, source_path, coverage_source_path, csv_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Section",
        "Setting",
        "Slots",
        "Failed",
        "Missed",
        "Preview",
        "Gap",
        "DeltaMissedVsSparse",
        "DeltaGapVsSparse",
        "Role",
    ]
    content = f"""# Paper Table 2 Coverage-Aware Ablation

This file is generated by `generate_paper_tables.py`. Do not edit table values by hand; update the source CSVs or generator instead.

Sources:

- `{source_path}`
- `{coverage_source_path}`

Generated CSV: `{csv_path}`

## Table

{compact_markdown_table(headers, rows)}

## Caption

Table 2. Compact coverage-aware ablation and budget-split evidence. The first block shows the paper-facing sparse baseline, selected B=4 coverage-aware reference, and selected B=3 same-preview budget split. The second block shows the B=4 coverage-weight sensitivity rows. Negative deltas improve over `Sparse-TopK B=4 sm=3` for lower-is-better metrics.

## Notes

- `Coverage-Aware B=4 cw=0.5 cpw=0` is retained as the interpretable B=4 coverage reference, even though `cw=0`, `cw=0.25`, and `cw=0.5` are effectively tied in the stored sweep.
- `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` is the selected no-noise same-preview gap reference at preview `16`.
- Full power-penalty and neighbor-coverage diagnostics remain in `docs/COVERAGE_AWARE_ANALYSIS.md`.
"""
    path.write_text(content, encoding="utf-8")


def write_table3_markdown(path, rows, source_path, csv_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    main_rows = [row for row in rows if row["Scope"] == "overall"]
    scenario_rows = [row for row in rows if row["Scope"] != "overall"]
    headers = [
        "Scope",
        "AvgGap",
        "PoolShare",
        "SelectionShare",
        "ConfirmationShare",
        "InvitationShare",
        "DominantComponent",
        "OracleNotInSeedRate",
    ]
    content = f"""# Paper Table 3 Failure Diagnosis

This file is generated by `generate_paper_tables.py`. Do not edit table values by hand; update the source CSV or generator instead.

Source: `{source_path}`

Generated CSV: `{csv_path}`

## Main Diagnostic Row

{compact_markdown_table(headers, main_rows)}

## Per-Scenario Companion

{compact_markdown_table(headers, scenario_rows)}

## Caption

Table 3. Compact gap decomposition for `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`. The dominant residual component is the invitation-mask mismatch after confirmation: overall invitation share is `0.687`, while pool, selection and confirmation shares are smaller. This supports treating invitation-mask correction as the next mechanism-level step rather than adding another ordinary candidate-pool heuristic.

## Notes

- The exact oracle-index rates are stricter than the decomposition because multiple IRS indices can have similar current tx counts.
- This is a diagnostic table, not a new deployable method.
- Full trace-level evidence remains in `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md` and the source trace CSV.
"""
    path.write_text(content, encoding="utf-8")


def write_uncertainty_markdown(
    path,
    uncertainty_rows,
    paired_rows,
    main_frontier_path,
    uncertainty_csv_path,
    paired_csv_path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    uncertainty_headers = [
        "Method",
        "Metric",
        "Mean",
        "ScenarioStd",
        "ScenarioCI95",
        "ScenarioCount",
    ]
    paired_headers = [
        "Comparison",
        "Metric",
        "MeanDelta",
        "DeltaScenarioCI95",
        "ImprovedScenarios",
        "ScenarioCount",
    ]
    content = f"""# Paper Table 1 Uncertainty Companion

This file is generated by `generate_paper_tables.py`.

Source: `{main_frontier_path}`

Generated CSVs:

- `{uncertainty_csv_path}`
- `{paired_csv_path}`

## Scope

This companion quantifies scenario-level variability for Table 1. Each scenario row is one `rho/delay` setting from the main frontier and has already been averaged over the available run seeds and episodes. The `ScenarioCI95` columns use a two-sided t critical value over the 9 scenario means. This is not a full seed-level significance test because the stored source CSVs do not retain per-seed values for every Table 1 metric.

For paired deltas, `MeanDelta = Candidate - Baseline`; negative values improve the lower-is-better metrics (`Slots`, `Failed`, `Missed`, and `Gap`). `ImprovedScenarios` counts the number of `rho/delay` scenarios where the candidate has a lower value than the baseline.

## Scenario-Level Variability

{compact_markdown_table(uncertainty_headers, uncertainty_rows)}

## Paired Same-Preview Deltas

{compact_markdown_table(paired_headers, paired_rows)}

## Interpretation Boundary

The companion supports statements such as "observed mean improvement with scenario-level robustness" and helps avoid overclaiming from a mean-only Table 1. Claims of formal statistical dominance should wait for raw per-seed paired outputs or a dedicated bootstrap/significance artifact.
"""
    path.write_text(content, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--main-frontier", type=Path, default=DEFAULT_MAIN_FRONTIER)
    parser.add_argument("--table1-csv", type=Path, default=DEFAULT_TABLE1_CSV)
    parser.add_argument(
        "--uncertainty-csv",
        type=Path,
        default=DEFAULT_UNCERTAINTY_CSV,
    )
    parser.add_argument("--paired-csv", type=Path, default=DEFAULT_PAIRED_CSV)
    parser.add_argument("--table1-md", type=Path, default=DEFAULT_TABLE1_MD)
    parser.add_argument(
        "--uncertainty-md",
        type=Path,
        default=DEFAULT_UNCERTAINTY_MD,
    )
    parser.add_argument("--coverage-source", type=Path, default=DEFAULT_COVERAGE_SOURCE)
    parser.add_argument(
        "--failure-diagnosis-source",
        type=Path,
        default=DEFAULT_FAILURE_DIAGNOSIS_SOURCE,
    )
    parser.add_argument("--table2-csv", type=Path, default=DEFAULT_TABLE2_CSV)
    parser.add_argument("--table2-md", type=Path, default=DEFAULT_TABLE2_MD)
    parser.add_argument("--table3-csv", type=Path, default=DEFAULT_TABLE3_CSV)
    parser.add_argument("--table3-md", type=Path, default=DEFAULT_TABLE3_MD)
    return parser.parse_args()


def main():
    args = parse_args()
    source_rows = read_rows(args.source)
    frontier_rows = read_rows(args.main_frontier)
    coverage_rows = read_rows(args.coverage_source)
    diagnosis_rows = read_rows(args.failure_diagnosis_source)
    table1_rows = build_table1(source_rows)
    uncertainty_rows = build_uncertainty_rows(frontier_rows)
    paired_rows = build_paired_delta_rows(frontier_rows)
    table2_rows = build_table2_rows(source_rows, coverage_rows)
    table3_rows = build_table3_rows(diagnosis_rows)
    write_csv(args.table1_csv, table1_rows)
    write_named_csv(args.uncertainty_csv, uncertainty_rows, UNCERTAINTY_COLUMNS)
    write_named_csv(args.paired_csv, paired_rows, PAIRED_COLUMNS)
    write_named_csv(args.table2_csv, table2_rows, TABLE2_COLUMNS)
    write_named_csv(args.table3_csv, table3_rows, TABLE3_COLUMNS)
    write_markdown(args.table1_md, table1_rows, args.source, args.table1_csv)
    write_uncertainty_markdown(
        args.uncertainty_md,
        uncertainty_rows,
        paired_rows,
        args.main_frontier,
        args.uncertainty_csv,
        args.paired_csv,
    )
    write_table2_markdown(
        args.table2_md,
        table2_rows,
        args.source,
        args.coverage_source,
        args.table2_csv,
    )
    write_table3_markdown(
        args.table3_md,
        table3_rows,
        args.failure_diagnosis_source,
        args.table3_csv,
    )
    print(f"wrote {args.table1_csv}")
    print(f"wrote {args.uncertainty_csv}")
    print(f"wrote {args.paired_csv}")
    print(f"wrote {args.table2_csv}")
    print(f"wrote {args.table3_csv}")
    print(f"wrote {args.table1_md}")
    print(f"wrote {args.uncertainty_md}")
    print(f"wrote {args.table2_md}")
    print(f"wrote {args.table3_md}")


if __name__ == "__main__":
    main()
