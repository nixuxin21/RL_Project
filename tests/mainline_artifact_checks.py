"""
Audit the existing artifact chain for the current execution-mismatch mainline.

This check does not re-run experiments. It verifies that the committed/generated
CSV, figure, and Markdown artifacts still support the current main claims.
"""

import csv
import re
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_RESULTS = Path("results/execution_mismatch")
PAPER_RESULTS = Path("results/paper")

FINAL_SUMMARY = EXECUTION_RESULTS / "final_execution_baseline_summary.csv"
MAIN_FRONTIER = EXECUTION_RESULTS / "main_frontier_analysis.csv"
FINAL_INVITATION = EXECUTION_RESULTS / "final_invitation_mask_analysis.csv"
PAPER_TABLE1 = PAPER_RESULTS / "table1_main_results.csv"
PAPER_TABLE1_MD = Path("docs/PAPER_TABLE1_MAIN_RESULTS.md")
PAPER_FIGURE_POINTS = PAPER_RESULTS / "figure2_figure3_points.csv"
PAPER_FIGURE2 = PAPER_RESULTS / "figure2_preview_gap_frontier.png"
PAPER_FIGURE3 = PAPER_RESULTS / "figure3_failed_missed_tradeoff.png"
PAPER_FIGURE4_POINTS = PAPER_RESULTS / "figure4_invitation_mask_noise_points.csv"
PAPER_FIGURE4_GAP = PAPER_RESULTS / "figure4_invitation_mask_gap_noise.png"
PAPER_FIGURE4_FAILED_MISSED = PAPER_RESULTS / "figure4_invitation_mask_failed_missed_noise.png"
PAPER_RESULT_PACKAGE = Path("docs/PAPER_RESULT_PACKAGE.md")
PAPER_STRUCTURE_MAP = Path("docs/PAPER_STRUCTURE_MAP.md")
PAPER_FIGURE_SPECS = Path("docs/PAPER_FIGURE_TABLE_SPECS.md")
PAPER_APPENDIX_BOUNDARY = Path("docs/PAPER_APPENDIX_BOUNDARY.md")
PAPER_TEXT_OUTLINE = Path("docs/PAPER_TEXT_OUTLINE.md")
PAPER_ASSET_GAP = Path("docs/PAPER_ASSET_GAP_CHECKLIST.md")
PAPER_FREEZE_MANIFEST = Path("docs/PAPER_FREEZE_MANIFEST.md")
FINAL_INVITATION_MD = Path("docs/FINAL_INVITATION_MASK_ANALYSIS.md")
INVITATION_FORMAL = (
    EXECUTION_RESULTS
    / "invitation_mask_correction_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1.csv"
)
INVITATION_NOISE = (
    EXECUTION_RESULTS
    / "invitation_mask_correction_noise_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_fbn0-0p02-0p05-0p1.csv"
)
INVITATION_NOISE_AWARE = (
    EXECUTION_RESULTS
    / "invitation_mask_correction_noise_aware_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_clipinf-2_fbn0-0p02-0p05-0p1.csv"
)
COVERAGE_DIAGNOSIS_SUMMARY = (
    EXECUTION_RESULTS
    / "coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_summary.csv"
)
COVERAGE_DIAGNOSIS_TRACE = (
    EXECUTION_RESULTS
    / "coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_trace.csv"
)

TABLE1_METHOD_ORDER = [
    "Rotating B=8",
    "Sparse-TopK B=4 sm=3",
    "Coverage-Aware B=4 cw=0.5 cpw=0",
    "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0",
    "Mask-Corrected Coverage-Aware B=3 mc=1",
    "Stale-TopK B=4",
    "Temporal Deviation Oracle B=4",
]

REQUIRED_ARTIFACTS = [
    FINAL_SUMMARY,
    MAIN_FRONTIER,
    EXECUTION_RESULTS / "main_frontier_preview_gap.png",
    EXECUTION_RESULTS / "main_frontier_failed_missed.png",
    EXECUTION_RESULTS / "coverage_aware_ablation_analysis.csv",
    EXECUTION_RESULTS / "coverage_aware_weight_ablation.png",
    EXECUTION_RESULTS / "coverage_aware_power_ablation.png",
    COVERAGE_DIAGNOSIS_SUMMARY,
    COVERAGE_DIAGNOSIS_TRACE,
    INVITATION_FORMAL,
    INVITATION_NOISE,
    INVITATION_NOISE_AWARE,
    FINAL_INVITATION,
    EXECUTION_RESULTS / "final_invitation_mask_gap_noise.png",
    EXECUTION_RESULTS / "final_invitation_mask_failed_missed_noise.png",
    Path("docs/EXECUTION_BASELINE_SUMMARY.md"),
    PAPER_RESULT_PACKAGE,
    PAPER_STRUCTURE_MAP,
    PAPER_FIGURE_SPECS,
    PAPER_APPENDIX_BOUNDARY,
    PAPER_TEXT_OUTLINE,
    PAPER_ASSET_GAP,
    PAPER_FREEZE_MANIFEST,
    Path("docs/figures/figure1_system_flow.mmd"),
    PAPER_TABLE1,
    PAPER_TABLE1_MD,
    PAPER_FIGURE_POINTS,
    PAPER_FIGURE2,
    PAPER_FIGURE3,
    PAPER_FIGURE4_POINTS,
    PAPER_FIGURE4_GAP,
    PAPER_FIGURE4_FAILED_MISSED,
    Path("docs/MAIN_RESULTS_ANALYSIS.md"),
    Path("docs/COVERAGE_AWARE_ANALYSIS.md"),
    Path("docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md"),
    Path("docs/INVITATION_MASK_CORRECTION.md"),
    Path("docs/INVITATION_MASK_CORRECTION_NOISE.md"),
    Path("docs/INVITATION_MASK_CORRECTION_NOISE_AWARE.md"),
    FINAL_INVITATION_MD,
    Path("docs/RESULTS_INDEX.md"),
]

REQUIRED_COLUMNS = {
    FINAL_SUMMARY: {
        "section",
        "label",
        "role",
        "source_file",
        "policy",
        "scenario_count",
        "episodes_per_scenario",
        "num_seeds",
        "slots_mean",
        "failed_nodes_mean",
        "missed_opportunities_mean",
        "decision_preview_calls_per_slot_mean",
        "oracle_tx_gap_mean",
    },
    MAIN_FRONTIER: {
        "label",
        "role",
        "source_file",
        "policy",
        "scenario_key",
        "channel_rho",
        "csi_delay_slots",
        "episodes",
        "num_seeds",
        "slots_mean",
        "failed_nodes_mean",
        "missed_opportunities_mean",
        "decision_preview_calls_per_slot_mean",
        "oracle_tx_gap_mean",
    },
    FINAL_INVITATION: {
        "label",
        "short_label",
        "role",
        "policy",
        "feedback_noise_std",
        "scenario_count",
        "episodes_per_scenario",
        "num_seeds",
        "slots_mean",
        "failed_nodes_mean",
        "missed_opportunities_mean",
        "decision_preview_calls_per_slot_mean",
        "oracle_tx_gap_mean",
        "mask_correction_applied_rate",
    },
    INVITATION_FORMAL: {
        "policy",
        "channel_rho",
        "csi_delay_slots",
        "episodes",
        "num_seeds",
        "mask_correction_strength",
        "decision_preview_calls_per_slot_mean",
        "oracle_tx_gap_mean",
    },
    INVITATION_NOISE_AWARE: {
        "policy",
        "channel_rho",
        "csi_delay_slots",
        "episodes",
        "num_seeds",
        "mask_correction_strength",
        "mask_correction_max_delta",
        "confirmation_feedback_noise_std",
        "decision_preview_calls_per_slot_mean",
        "oracle_tx_gap_mean",
    },
    COVERAGE_DIAGNOSIS_SUMMARY: {
        "label",
        "invitation_gap_share",
        "selection_gap_share",
        "confirmation_gap_share",
        "pool_gap_share",
        "oracle_not_in_seed_rate",
    },
    PAPER_TABLE1: {
        "Method",
        "Role",
        "Slots",
        "Failed",
        "Missed",
        "Preview",
        "Gap",
    },
    PAPER_FIGURE_POINTS: {
        "Method",
        "ShortLabel",
        "Role",
        "Preview",
        "Gap",
        "Failed",
        "Missed",
        "IsProposed",
        "IsOracle",
    },
    PAPER_FIGURE4_POINTS: {
        "Label",
        "ShortLabel",
        "Role",
        "FeedbackNoiseStd",
        "Gap",
        "Failed",
        "Missed",
        "IsReliableFeedbackMain",
        "IsHighNoiseVariant",
    },
}

SUMMARY_REQUIRED_LABELS = set(TABLE1_METHOD_ORDER)

FIGURE4_LABEL_ORDER = [
    "Coverage-Aware B=3",
    "Direct Mask Correction mc=1",
    "Clipped Mask Correction mc=1 clip=2",
]

FIGURE4_NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1]

PAPER_FACING_TEXT_REQUIREMENTS = {
    PAPER_RESULT_PACKAGE: [
        "docs/PAPER_APPENDIX_BOUNDARY.md",
        "docs/PAPER_TEXT_OUTLINE.md",
        "docs/PAPER_ASSET_GAP_CHECKLIST.md",
        "results/paper/table1_main_results.csv",
        str(PAPER_FIGURE_POINTS),
        str(PAPER_FIGURE2),
        str(PAPER_FIGURE3),
        str(PAPER_FIGURE4_POINTS),
        str(PAPER_FIGURE4_GAP),
        str(PAPER_FIGURE4_FAILED_MISSED),
        "When drafting, cite the `results/paper/` assets for main-text figures",
    ],
    PAPER_STRUCTURE_MAP: [
        "docs/PAPER_APPENDIX_BOUNDARY.md",
        "docs/PAPER_TEXT_OUTLINE.md",
        "docs/PAPER_ASSET_GAP_CHECKLIST.md",
        "默认只保留 Appendix A/B/C 三组证据",
        "Figure 4 中的 `Direct` 是 `Mask-Corrected Coverage-Aware B=3 mc=1`",
        "使用 `make paper-table1` 和 `make paper-figures`",
        str(PAPER_FIGURE4_POINTS),
        str(PAPER_FIGURE4_GAP),
        str(PAPER_FIGURE4_FAILED_MISSED),
    ],
    PAPER_FIGURE_SPECS: [
        "## Naming Crosswalk",
        "`Direct Mask Correction mc=1`",
        "`Clipped Mask Correction mc=1 clip=2`",
        str(PAPER_FIGURE4_POINTS),
        str(PAPER_FIGURE4_GAP),
        str(PAPER_FIGURE4_FAILED_MISSED),
        "Direct target-count correction is the reliable-feedback main method",
    ],
    PAPER_APPENDIX_BOUNDARY: [
        "## Default Appendix Minimum",
        "Appendix A: Historical baseline context",
        "Appendix B: Preview-cost and CSI motivation",
        "Appendix C: Diagnostic branches not used as proposed methods",
        "## Supplement Only On Demand",
        "## Main-Text Exclusion Rules",
        "## Reopen Criteria",
        "results/policy_comparison/policy_comparison_summary_ep1000_runs5_seed2026_featargmax_powertie_cbsac.csv",
        "results/runtime/runtime_benchmark_ep200_seed2026.csv",
        "results/bandit_feedback/bandit_feedback_stress_formal_ep1000_runs5_seed2026.csv",
        "`mc=1 clip=2` 是 high-noise robustness variant",
    ],
    PAPER_TEXT_OUTLINE: [
        "## Abstract Skeleton",
        "## Introduction",
        "## Related Work",
        "## System Model",
        "## Problem Formulation",
        "## Method",
        "## Experiments",
        "## Discussion",
        "## Appendix Minimum",
        "## Claim Traceability Checklist",
        "docs/PAPER_RESULT_PACKAGE.md",
        "docs/PAPER_STRUCTURE_MAP.md",
        "docs/PAPER_FIGURE_TABLE_SPECS.md",
        "docs/PAPER_APPENDIX_BOUNDARY.md",
        str(PAPER_FIGURE2),
        str(PAPER_FIGURE3),
        str(PAPER_FIGURE4_POINTS),
        "`Mask-Corrected Coverage-Aware B=3 mc=1` remains the reliable-feedback main method",
        "`mc=1 clip=2` remains high-noise robustness only",
        "No learned or RL branch is reintroduced as a main method",
        "make paper-table1",
        "make paper-figures",
        "make mainline-audit",
    ],
    PAPER_ASSET_GAP: [
        "## Current Asset Status",
        "## Missing Paper-Facing Artifacts",
        "## Do Not Generate By Default",
        "## Readiness Gates",
        "Figure 1 needs a publication export",
        "Table 2 and Table 3 need compact paper-facing artifacts",
        "docs/PAPER_RESULT_PACKAGE.md",
        "docs/PAPER_FIGURE_TABLE_SPECS.md",
        "docs/PAPER_TEXT_OUTLINE.md",
        "docs/PAPER_TABLE1_MAIN_RESULTS.md",
        str(PAPER_TABLE1),
        "docs/figures/figure1_system_flow.mmd",
        str(PAPER_FIGURE2),
        str(PAPER_FIGURE3),
        str(PAPER_FIGURE4_POINTS),
        str(PAPER_FIGURE4_GAP),
        str(PAPER_FIGURE4_FAILED_MISSED),
        "results/execution_mismatch/coverage_aware_ablation_analysis.csv",
        "docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md",
        "make paper-table1",
        "make paper-figures",
    ],
    FINAL_INVITATION_MD: [
        "Paper-facing Figure 4 artifacts are generated",
        str(PAPER_FIGURE4_POINTS),
        str(PAPER_FIGURE4_GAP),
        str(PAPER_FIGURE4_FAILED_MISSED),
    ],
    PAPER_FREEZE_MANIFEST: [
        "## Freeze Scope",
        "## Verification Commands",
        "## Frozen Generated Artifacts",
        "## Non-Freeze Boundary",
        "Mask-Corrected Coverage-Aware B=3 mc=1",
        "mc=1 clip=2",
        "make mainline-audit",
        str(FINAL_SUMMARY),
        str(PAPER_TABLE1),
        str(PAPER_FIGURE2),
        str(PAPER_FIGURE4_GAP),
    ],
}


def fail(message):
    raise AssertionError(message)


def project_path(path):
    return PROJECT_ROOT / path


def assert_exists(path):
    full_path = project_path(path)
    if not full_path.exists():
        fail(f"missing artifact: {path}")
    return full_path


def read_csv(path):
    full_path = assert_exists(path)
    with full_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    if not rows:
        fail(f"empty CSV artifact: {path}")
    missing = REQUIRED_COLUMNS.get(path, set()) - set(reader.fieldnames or [])
    if missing:
        fail(f"{path} missing columns: {sorted(missing)}")
    return rows


def as_float(row, key):
    return float(row[key])


def row_by(rows, key, value):
    matches = [row for row in rows if row.get(key) == value]
    if not matches:
        fail(f"missing row where {key}={value!r}")
    if len(matches) > 1:
        fail(f"ambiguous row where {key}={value!r}: {len(matches)} matches")
    return matches[0]


def rows_by(rows, **criteria):
    selected = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    if not selected:
        fail(f"missing rows matching {criteria}")
    return selected


def assert_close(row, key, expected, tol=1e-9):
    observed = as_float(row, key)
    if abs(observed - float(expected)) > tol:
        fail(f"{row.get('label', row.get('policy', '<row>'))}: {key}={observed:g}, expected {expected:g}")


def assert_all_exist():
    for path in REQUIRED_ARTIFACTS:
        assert_exists(path)


def assert_results_index_paths_exist():
    return assert_documented_paths_exist(Path("docs/RESULTS_INDEX.md"))


def assert_documented_paths_exist(document_path):
    full_document_path = assert_exists(document_path)
    text = full_document_path.read_text(encoding="utf-8")
    documented_paths = sorted(
        {
            Path(match)
            for match in re.findall(r"`((?:results|docs)/[^`]+)`", text)
            if not match.endswith("/")
        }
    )
    for path in documented_paths:
        assert_exists(path)
    return len(documented_paths)


def assert_paper_facing_text():
    checked = 0
    for path, snippets in PAPER_FACING_TEXT_REQUIREMENTS.items():
        text = assert_exists(path).read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in text:
                fail(f"{path} missing required paper-facing text: {snippet!r}")
            checked += 1
    for label in TABLE1_METHOD_ORDER:
        text = assert_exists(PAPER_FIGURE_SPECS).read_text(encoding="utf-8")
        if label not in text:
            fail(f"{PAPER_FIGURE_SPECS} missing Table 1 method label {label!r}")
        checked += 1
    return checked


def assert_source_files_exist(rows):
    for row in rows:
        source_file = row.get("source_file", "").strip()
        if not source_file:
            continue
        assert_exists(EXECUTION_RESULTS / source_file)


def assert_final_summary():
    rows = read_csv(FINAL_SUMMARY)
    labels = {row["label"] for row in rows}
    missing_labels = SUMMARY_REQUIRED_LABELS - labels
    if missing_labels:
        fail(f"{FINAL_SUMMARY} missing mainline labels: {sorted(missing_labels)}")
    assert_source_files_exist(rows)

    coverage_b3 = row_by(rows, "label", "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0")
    mask_corrected = row_by(rows, "label", "Mask-Corrected Coverage-Aware B=3 mc=1")
    temporal_oracle = row_by(rows, "label", "Temporal Deviation Oracle B=4")
    stale_topk = row_by(rows, "label", "Stale-TopK B=4")

    for row in (coverage_b3, mask_corrected, temporal_oracle, stale_topk):
        assert_close(row, "scenario_count", 9.0)

    assert_close(coverage_b3, "decision_preview_calls_per_slot_mean", 16.0)
    assert_close(mask_corrected, "decision_preview_calls_per_slot_mean", 16.0)
    assert_close(temporal_oracle, "decision_preview_calls_per_slot_mean", 4.0)
    assert_close(stale_topk, "decision_preview_calls_per_slot_mean", 20.0)

    if as_float(mask_corrected, "oracle_tx_gap_mean") >= as_float(coverage_b3, "oracle_tx_gap_mean"):
        fail("mask-corrected main method must improve oracle gap over uncorrected Coverage-Aware B=3")
    if as_float(mask_corrected, "failed_nodes_mean") >= as_float(coverage_b3, "failed_nodes_mean"):
        fail("mask-corrected main method must reduce failed nodes over uncorrected Coverage-Aware B=3")
    if as_float(mask_corrected, "missed_opportunities_mean") >= as_float(coverage_b3, "missed_opportunities_mean"):
        fail("mask-corrected main method must reduce missed opportunities over uncorrected Coverage-Aware B=3")
    return len(rows)


def assert_paper_table1():
    rows = read_csv(PAPER_TABLE1)
    observed_order = [row["Method"] for row in rows]
    if observed_order != TABLE1_METHOD_ORDER:
        fail(f"{PAPER_TABLE1} has unexpected method order: {observed_order}")

    summary_rows = read_csv(FINAL_SUMMARY)
    expected_metrics = {
        "Slots": ("slots_mean", 3),
        "Failed": ("failed_nodes_mean", 3),
        "Missed": ("missed_opportunities_mean", 3),
        "Preview": ("decision_preview_calls_per_slot_mean", 2),
        "Gap": ("oracle_tx_gap_mean", 3),
    }

    for row in rows:
        summary_row = row_by(summary_rows, "label", row["Method"])
        if row["Role"] != summary_row["role"]:
            fail(f"{PAPER_TABLE1}: role mismatch for {row['Method']!r}")
        for table_key, (summary_key, digits) in expected_metrics.items():
            expected = f"{as_float(summary_row, summary_key):.{digits}f}"
            if row[table_key] != expected:
                fail(
                    f"{PAPER_TABLE1}: {row['Method']} {table_key}={row[table_key]!r}, "
                    f"expected {expected!r}"
                )

    text = assert_exists(PAPER_TABLE1_MD).read_text(encoding="utf-8")
    for label in TABLE1_METHOD_ORDER:
        if f"`{label}`" not in text:
            fail(f"{PAPER_TABLE1_MD} missing method label {label!r}")
    return len(rows)


def assert_paper_figures():
    rows = read_csv(PAPER_FIGURE_POINTS)
    observed_order = [row["Method"] for row in rows]
    if observed_order != TABLE1_METHOD_ORDER:
        fail(f"{PAPER_FIGURE_POINTS} has unexpected method order: {observed_order}")

    summary_rows = read_csv(FINAL_SUMMARY)
    expected_metrics = {
        "Preview": ("decision_preview_calls_per_slot_mean", 2),
        "Gap": ("oracle_tx_gap_mean", 3),
        "Failed": ("failed_nodes_mean", 3),
        "Missed": ("missed_opportunities_mean", 3),
    }
    proposed_count = 0
    oracle_count = 0
    for row in rows:
        summary_row = row_by(summary_rows, "label", row["Method"])
        if row["Role"] != summary_row["role"]:
            fail(f"{PAPER_FIGURE_POINTS}: role mismatch for {row['Method']!r}")
        for table_key, (summary_key, digits) in expected_metrics.items():
            expected = f"{as_float(summary_row, summary_key):.{digits}f}"
            if row[table_key] != expected:
                fail(
                    f"{PAPER_FIGURE_POINTS}: {row['Method']} {table_key}={row[table_key]!r}, "
                    f"expected {expected!r}"
                )
        proposed_count += row["IsProposed"] == "True"
        oracle_count += row["IsOracle"] == "True"

    if proposed_count != 1:
        fail(f"{PAPER_FIGURE_POINTS} should mark exactly one proposed method")
    if oracle_count != 1:
        fail(f"{PAPER_FIGURE_POINTS} should mark exactly one oracle method")

    for path in (PAPER_FIGURE2, PAPER_FIGURE3):
        full_path = assert_exists(path)
        if full_path.stat().st_size <= 0:
            fail(f"empty paper figure artifact: {path}")
    return len(rows)


def invitation_row_by_label_noise(rows, label, noise):
    matches = [
        row
        for row in rows
        if row.get("label") == label
        and abs(as_float(row, "feedback_noise_std") - noise) < 1e-12
    ]
    if not matches:
        fail(f"missing final invitation row for {label!r} noise={noise:g}")
    if len(matches) > 1:
        fail(f"ambiguous final invitation row for {label!r} noise={noise:g}")
    return matches[0]


def assert_paper_figure4():
    rows = read_csv(PAPER_FIGURE4_POINTS)
    expected_order = [
        (f"{noise:g}", label)
        for noise in FIGURE4_NOISE_LEVELS
        for label in FIGURE4_LABEL_ORDER
    ]
    observed_order = [(row["FeedbackNoiseStd"], row["Label"]) for row in rows]
    if observed_order != expected_order:
        fail(f"{PAPER_FIGURE4_POINTS} has unexpected row order: {observed_order}")

    final_rows = read_csv(FINAL_INVITATION)
    expected_metrics = {
        "Gap": ("oracle_tx_gap_mean", 3),
        "Failed": ("failed_nodes_mean", 3),
        "Missed": ("missed_opportunities_mean", 3),
    }
    reliable_main_count = 0
    high_noise_variant_count = 0
    for row in rows:
        noise = float(row["FeedbackNoiseStd"])
        source_row = invitation_row_by_label_noise(final_rows, row["Label"], noise)
        if row["ShortLabel"] != source_row["short_label"]:
            fail(f"{PAPER_FIGURE4_POINTS}: short label mismatch for {row['Label']!r}")
        if row["Role"] != source_row["role"]:
            fail(f"{PAPER_FIGURE4_POINTS}: role mismatch for {row['Label']!r}")
        for figure_key, (source_key, digits) in expected_metrics.items():
            expected = f"{as_float(source_row, source_key):.{digits}f}"
            if row[figure_key] != expected:
                fail(
                    f"{PAPER_FIGURE4_POINTS}: {row['Label']} noise={noise:g} "
                    f"{figure_key}={row[figure_key]!r}, expected {expected!r}"
                )
        reliable_main_count += row["IsReliableFeedbackMain"] == "True"
        high_noise_variant_count += row["IsHighNoiseVariant"] == "True"

    if reliable_main_count != 1:
        fail(f"{PAPER_FIGURE4_POINTS} should mark exactly one reliable-feedback main point")
    if high_noise_variant_count != 1:
        fail(f"{PAPER_FIGURE4_POINTS} should mark exactly one high-noise variant point")

    for path in (PAPER_FIGURE4_GAP, PAPER_FIGURE4_FAILED_MISSED):
        full_path = assert_exists(path)
        if full_path.stat().st_size <= 0:
            fail(f"empty paper Figure 4 artifact: {path}")
    return len(rows)


def assert_main_frontier():
    rows = read_csv(MAIN_FRONTIER)
    assert_source_files_exist(rows)
    scenario_keys = {row["scenario_key"] for row in rows}
    if len(scenario_keys) != 9:
        fail(f"{MAIN_FRONTIER} should cover 9 rho/delay scenarios, found {len(scenario_keys)}")
    labels = {row["label"] for row in rows}
    for label in SUMMARY_REQUIRED_LABELS - {"Mask-Corrected Coverage-Aware B=3 mc=1"}:
        if label not in labels:
            fail(f"{MAIN_FRONTIER} missing label {label!r}")
    return len(rows)


def assert_invitation_artifacts():
    final_rows = read_csv(FINAL_INVITATION)
    formal_rows = read_csv(INVITATION_FORMAL)
    noise_rows = read_csv(INVITATION_NOISE)
    noise_aware_rows = read_csv(INVITATION_NOISE_AWARE)

    reliable_b3 = rows_by(final_rows, label="Coverage-Aware B=3", feedback_noise_std="0.0")[0]
    reliable_direct = rows_by(final_rows, label="Direct Mask Correction mc=1", feedback_noise_std="0.0")[0]
    noisy_direct = rows_by(final_rows, label="Direct Mask Correction mc=1", feedback_noise_std="0.1")[0]
    noisy_clipped = rows_by(final_rows, label="Clipped Mask Correction mc=1 clip=2", feedback_noise_std="0.1")[0]

    for row in (reliable_b3, reliable_direct, noisy_direct, noisy_clipped):
        assert_close(row, "scenario_count", 9.0)
        assert_close(row, "episodes_per_scenario", 900.0)
        assert_close(row, "num_seeds", 3.0)
        assert_close(row, "decision_preview_calls_per_slot_mean", 16.0)

    if as_float(reliable_direct, "oracle_tx_gap_mean") >= as_float(reliable_b3, "oracle_tx_gap_mean"):
        fail("reliable direct mask correction must improve gap over B3")
    if as_float(reliable_direct, "mask_correction_applied_rate") <= 0.0:
        fail("reliable direct mask correction should apply on at least some slots")
    if as_float(noisy_clipped, "oracle_tx_gap_mean") >= as_float(noisy_direct, "oracle_tx_gap_mean"):
        fail("high-noise clipped variant must improve gap over direct mc=1")
    if as_float(noisy_clipped, "failed_nodes_mean") >= as_float(noisy_direct, "failed_nodes_mean"):
        fail("high-noise clipped variant must reduce failed nodes over direct mc=1")

    formal_strengths = {float(row["mask_correction_strength"]) for row in formal_rows}
    if formal_strengths != {0.0, 0.75, 1.0}:
        fail(f"{INVITATION_FORMAL} has unexpected correction strengths: {sorted(formal_strengths)}")
    if len(formal_rows) != 27:
        fail(f"{INVITATION_FORMAL} should have 27 rows, found {len(formal_rows)}")

    noise_values = {float(row["confirmation_feedback_noise_std"]) for row in noise_rows}
    if noise_values != {0.0, 0.02, 0.05, 0.1}:
        fail(f"{INVITATION_NOISE} has unexpected feedback-noise values: {sorted(noise_values)}")

    max_deltas = {float(row["mask_correction_max_delta"]) for row in noise_aware_rows}
    if max_deltas != {-1.0, 2.0}:
        fail(f"{INVITATION_NOISE_AWARE} has unexpected max-delta values: {sorted(max_deltas)}")
    return len(final_rows) + len(formal_rows) + len(noise_rows) + len(noise_aware_rows)


def assert_coverage_diagnosis():
    rows = read_csv(COVERAGE_DIAGNOSIS_SUMMARY)
    overall = row_by(rows, "label", "overall")
    invitation_share = as_float(overall, "invitation_gap_share")
    competing_shares = [
        as_float(overall, "selection_gap_share"),
        as_float(overall, "confirmation_gap_share"),
        as_float(overall, "pool_gap_share"),
    ]
    if invitation_share <= max(competing_shares):
        fail("Coverage B3 diagnosis should identify invitation mismatch as the dominant residual gap")
    if as_float(overall, "oracle_not_in_seed_rate") <= 0.0:
        fail("Coverage B3 diagnosis should retain nonzero seed-pool miss evidence")
    assert_exists(COVERAGE_DIAGNOSIS_TRACE)
    return len(rows)


def main():
    checked_rows = 0
    assert_all_exist()
    documented_count = assert_results_index_paths_exist()
    appendix_documented_count = assert_documented_paths_exist(PAPER_APPENDIX_BOUNDARY)
    outline_documented_count = assert_documented_paths_exist(PAPER_TEXT_OUTLINE)
    asset_gap_documented_count = assert_documented_paths_exist(PAPER_ASSET_GAP)
    freeze_documented_count = assert_documented_paths_exist(PAPER_FREEZE_MANIFEST)
    paper_text_count = assert_paper_facing_text()
    checked_rows += assert_final_summary()
    checked_rows += assert_paper_table1()
    checked_rows += assert_paper_figures()
    checked_rows += assert_paper_figure4()
    checked_rows += assert_main_frontier()
    checked_rows += assert_invitation_artifacts()
    checked_rows += assert_coverage_diagnosis()

    print("mainline artifact checks passed")
    print(f"  checked artifacts: {len(REQUIRED_ARTIFACTS)}")
    print(f"  documented paths verified: {documented_count}")
    print(f"  appendix paths verified: {appendix_documented_count}")
    print(f"  outline paths verified: {outline_documented_count}")
    print(f"  asset gap paths verified: {asset_gap_documented_count}")
    print(f"  freeze manifest paths verified: {freeze_documented_count}")
    print(f"  paper-facing text snippets verified: {paper_text_count}")
    print(f"  checked CSV rows: {checked_rows}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"mainline artifact checks failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
