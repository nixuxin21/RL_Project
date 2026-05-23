"""审计主线 artifact、source_file 链接、论文表图和结果文档的一致性。"""

import csv
import re
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_RESULTS = Path("results/execution_mismatch")
PAPER_RESULTS = Path("results/paper")

FINAL_SUMMARY = EXECUTION_RESULTS / "final_execution_baseline_summary.csv"
MAIN_FRONTIER = EXECUTION_RESULTS / "main_frontier_analysis.csv"
FINAL_INVITATION = EXECUTION_RESULTS / "final_invitation_mask_analysis.csv"
PAPER_TABLE1 = PAPER_RESULTS / "table1_main_results.csv"
PAPER_TABLE1_MD = Path("docs/PAPER_TABLE1_MAIN_RESULTS.md")
PAPER_TABLE1_UNCERTAINTY = Path("docs/PAPER_TABLE1_UNCERTAINTY.md")
PAPER_TABLE1_SCENARIO_UNCERTAINTY = PAPER_RESULTS / "table1_scenario_uncertainty.csv"
PAPER_TABLE1_PAIRED_DELTAS = PAPER_RESULTS / "table1_paired_scenario_deltas.csv"
PAPER_TABLE2 = PAPER_RESULTS / "table2_coverage_aware_ablation.csv"
PAPER_TABLE2_MD = Path("docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md")
PAPER_TABLE3 = PAPER_RESULTS / "table3_failure_diagnosis.csv"
PAPER_TABLE3_MD = Path("docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md")
PAPER_FIGURE1_SOURCE = Path("docs/figures/figure1_system_flow.mmd")
PAPER_FIGURE1_SVG = PAPER_RESULTS / "figure1_system_flow.svg"
PAPER_FIGURE1_PDF = PAPER_RESULTS / "figure1_system_flow.pdf"
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
PROJECT_README = Path("README.md")
CANDIDATE_EVIDENCE_README = Path("docs/candidate_evidence/cost_frontier_main_v1/README.md")
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
    PROJECT_README,
    PAPER_FIGURE1_SOURCE,
    PAPER_FIGURE1_SVG,
    PAPER_FIGURE1_PDF,
    PAPER_TABLE1,
    PAPER_TABLE1_MD,
    PAPER_TABLE1_UNCERTAINTY,
    PAPER_TABLE1_SCENARIO_UNCERTAINTY,
    PAPER_TABLE1_PAIRED_DELTAS,
    PAPER_TABLE2,
    PAPER_TABLE2_MD,
    PAPER_TABLE3,
    PAPER_TABLE3_MD,
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
    PAPER_TABLE1_SCENARIO_UNCERTAINTY: {
        "Method",
        "Metric",
        "Mean",
        "ScenarioStd",
        "ScenarioSE",
        "ScenarioCI95",
        "ScenarioCount",
        "SamplesPerScenario",
        "Seeds",
    },
    PAPER_TABLE1_PAIRED_DELTAS: {
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
    },
    PAPER_TABLE2: {
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
    },
    PAPER_TABLE3: {
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
        "IsNoNoiseCorrectionTradeoff",
        "IsHighNoiseGapBest",
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
        str(PAPER_FIGURE1_SOURCE),
        str(PAPER_FIGURE1_SVG),
        str(PAPER_FIGURE1_PDF),
        str(PAPER_TABLE2),
        str(PAPER_TABLE3),
        "docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md",
        "docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md",
        str(PAPER_FIGURE_POINTS),
        str(PAPER_FIGURE2),
        str(PAPER_FIGURE3),
        str(PAPER_FIGURE4_POINTS),
        str(PAPER_FIGURE4_GAP),
        str(PAPER_FIGURE4_FAILED_MISSED),
        "When drafting, cite the `results/paper/` assets for main-text figures",
        "Uncertainty status: the compact paper Table 1 is mean-only",
        "docs/PAPER_TABLE1_UNCERTAINTY.md",
        str(PAPER_TABLE1_PAIRED_DELTAS),
        "stale-gain reranking",
    ],
    PAPER_STRUCTURE_MAP: [
        "docs/PAPER_APPENDIX_BOUNDARY.md",
        "docs/PAPER_TEXT_OUTLINE.md",
        "docs/PAPER_ASSET_GAP_CHECKLIST.md",
        "默认只保留 Appendix A/B/C 三组证据",
        "Figure 4 中的 `Direct` 是 `Mask-Corrected Coverage-Aware B=3 mc=1`",
        "使用 `make paper-tables` 和 `make paper-figures`",
        "docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md",
        "docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md",
        str(PAPER_FIGURE1_SVG),
        str(PAPER_FIGURE1_PDF),
        str(PAPER_FIGURE4_POINTS),
        str(PAPER_FIGURE4_GAP),
        str(PAPER_FIGURE4_FAILED_MISSED),
    ],
    PAPER_FIGURE_SPECS: [
        "## Naming Crosswalk",
        "## Information-Role Taxonomy",
        "`deployable`",
        "`diagnostic`",
        "`hidden-information temporal diagnostic`",
        "`Direct Mask Correction mc=1`",
        "`Clipped Mask Correction mc=1 clip=2`",
        str(PAPER_FIGURE1_SOURCE),
        str(PAPER_FIGURE1_SVG),
        str(PAPER_FIGURE1_PDF),
        str(PAPER_FIGURE4_POINTS),
        str(PAPER_FIGURE4_GAP),
        str(PAPER_FIGURE4_FAILED_MISSED),
        "Direct target-count correction is a no-noise trade-off",
        str(PAPER_TABLE2),
        str(PAPER_TABLE3),
        "Table 1 is a compact mean-only table",
        "docs/PAPER_TABLE1_UNCERTAINTY.md",
        "stale-gain reranking",
        "all non-oracle Table 1 methods in these figures as deployable",
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
        "`mc=1 clip=2` 是 failed-invitation control diagnostic",
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
        str(PAPER_TABLE2),
        str(PAPER_TABLE3),
        "`Mask-Corrected Coverage-Aware B=3 mc=1` is a no-noise trade-off",
        "`mc=1 clip=2` is a failed-invitation control diagnostic",
        "No learned or RL branch is reintroduced as a main method",
        "make paper-tables",
        "make paper-figures",
        "make mainline-audit",
    ],
    PAPER_ASSET_GAP: [
        "## Current Asset Status",
        "## Missing Paper-Facing Artifacts",
        "## Do Not Generate By Default",
        "## Readiness Gates",
        "Figure 1 SVG/PDF export is generated",
        "Table 2 and Table 3 now have compact paper-facing artifacts",
        "docs/PAPER_RESULT_PACKAGE.md",
        "docs/PAPER_FIGURE_TABLE_SPECS.md",
        "docs/PAPER_TEXT_OUTLINE.md",
        "docs/PAPER_TABLE1_MAIN_RESULTS.md",
        "docs/PAPER_TABLE1_UNCERTAINTY.md",
        str(PAPER_TABLE1),
        str(PAPER_TABLE1_SCENARIO_UNCERTAINTY),
        str(PAPER_TABLE1_PAIRED_DELTAS),
        "docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md",
        "docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md",
        str(PAPER_TABLE2),
        str(PAPER_TABLE3),
        str(PAPER_FIGURE1_SOURCE),
        str(PAPER_FIGURE1_SVG),
        str(PAPER_FIGURE1_PDF),
        str(PAPER_FIGURE2),
        str(PAPER_FIGURE3),
        str(PAPER_FIGURE4_POINTS),
        str(PAPER_FIGURE4_GAP),
        str(PAPER_FIGURE4_FAILED_MISSED),
        "results/execution_mismatch/coverage_aware_ablation_analysis.csv",
        "docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md",
        "make paper-tables",
        "make paper-figures",
    ],
    PAPER_TABLE1_MD: [
        "All non-oracle rows are deployable under the stated stale-CSI and aggregate-feedback information model",
        "Information-role taxonomy follows `docs/PAPER_FIGURE_TABLE_SPECS.md`",
        "All Table 1 rows except `Temporal Deviation Oracle B=4` are deployable comparisons or references",
        "hidden-information temporal diagnostic reference",
        "Learned variants are diagnostics",
        "The compact main table is mean-only",
        "Companion artifacts are generated",
    ],
    PAPER_TABLE1_UNCERTAINTY: [
        "scenario-level variability",
        "not a full seed-level significance test",
        "Paired Same-Preview Deltas",
        "ImprovedScenarios",
        "Mask-Corrected vs Coverage-Aware B=3",
    ],
    PAPER_TABLE2_MD: [
        "Paper Table 2 Coverage-Aware Ablation",
        str(PAPER_TABLE2),
        "Compact coverage-aware ablation and budget-split evidence",
        "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0",
        "Full power-penalty and neighbor-coverage diagnostics remain",
    ],
    PAPER_TABLE3_MD: [
        "Paper Table 3 Failure Diagnosis",
        str(PAPER_TABLE3),
        "overall invitation share is `0.687`",
        "This is a diagnostic table",
        "Full trace-level evidence remains",
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
        str(PAPER_FIGURE1_SOURCE),
        str(PAPER_FIGURE1_SVG),
        str(PAPER_FIGURE1_PDF),
        str(PAPER_TABLE1),
        str(PAPER_TABLE1_SCENARIO_UNCERTAINTY),
        str(PAPER_TABLE1_PAIRED_DELTAS),
        str(PAPER_TABLE2),
        str(PAPER_TABLE3),
        str(PAPER_FIGURE2),
        str(PAPER_FIGURE4_GAP),
    ],
    PROJECT_README: [
        "## Quick Reproduction Audit",
        "make quick-audit",
        "formal experiments",
        "summary `source_file` CSV",
        "`Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` 是当前 no-noise same-preview gap reference",
        "`Mask-Corrected Coverage-Aware B=3 mc=1` 是同 preview `16` 下的 trade-off result",
        "failed-invitation control diagnostic",
        "make paper-tables",
        str(PAPER_FIGURE1_SVG),
    ],
}
README_MAX_LINES = 260
README_FORBIDDEN_DEFAULT_TARGETS = [
    "make learned-sparse-shortlist-pilot",
    "make learned-set-shortlist-pilot",
    "make learned-pairwise-shortlist-pilot",
    "make policy-comparison-learning",
    "make bandit-feedback-stress",
]


def fail(message):
    raise AssertionError(message)


def project_path(path):
    return PROJECT_ROOT / path


def assert_exists(path):
    full_path = project_path(path)
    if not full_path.exists():
        fail(f"missing artifact: {path}")
    return full_path


def assert_git_tracked(path):
    """处理assert、git、tracked相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(path)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(path)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ignored.returncode == 0:
        fail(f"freeze artifact is ignored by .gitignore and will be absent in a clean clone: {path}")
    fail(f"freeze artifact is not tracked/staged and will be absent in a clean clone: {path}")


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
        assert_git_tracked(path)


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


def assert_readme_cleanup_boundary():
    text = assert_exists(PROJECT_README).read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) > README_MAX_LINES:
        fail(f"{PROJECT_README} should stay a compact entry point; found {len(lines)} lines")
    for target in README_FORBIDDEN_DEFAULT_TARGETS:
        if target in text:
            fail(f"{PROJECT_README} should not list diagnostic/archive target {target!r}")
    return len(lines)


def assert_candidate_evidence_boundary():
    text = assert_exists(CANDIDATE_EVIDENCE_README).read_text(encoding="utf-8")
    snippets = [
        "candidate evidence package",
        "not part of the current paper-freeze boundary",
        "results/main/cost_frontier_main_v1/",
        "results/main_analysis/cost_frontier_main_v1/",
        "docs/PAPER_FREEZE_MANIFEST.md",
    ]
    for snippet in snippets:
        if snippet not in text:
            fail(f"{CANDIDATE_EVIDENCE_README} missing candidate-boundary text: {snippet!r}")

    if subprocess.run(
        ["git", "check-ignore", "-q", "--", "docs/paper_evidence/example.md"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode != 0:
        fail("docs/paper_evidence/ should stay ignored for local paper-evidence drafts")

    if subprocess.run(
        ["git", "check-ignore", "-q", "--", str(CANDIDATE_EVIDENCE_README)],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0:
        fail(f"{CANDIDATE_EVIDENCE_README} should not be ignored")

    return len(snippets)


def assert_source_files_exist(rows):
    for row in rows:
        source_file = row.get("source_file", "").strip()
        if not source_file:
            continue
        source_path = EXECUTION_RESULTS / source_file
        assert_exists(source_path)
        assert_git_tracked(source_path)


def assert_source_files_documented(rows):
    manifest_text = assert_exists(PAPER_FREEZE_MANIFEST).read_text(encoding="utf-8")
    for row in rows:
        source_file = row.get("source_file", "").strip()
        if not source_file:
            continue
        source_path = str(EXECUTION_RESULTS / source_file)
        if source_file not in manifest_text and source_path not in manifest_text:
            fail(f"{PAPER_FREEZE_MANIFEST} missing source_file reference: {source_file}")


def assert_final_summary():
    rows = read_csv(FINAL_SUMMARY)
    labels = {row["label"] for row in rows}
    missing_labels = SUMMARY_REQUIRED_LABELS - labels
    if missing_labels:
        fail(f"{FINAL_SUMMARY} missing mainline labels: {sorted(missing_labels)}")
    assert_source_files_exist(rows)
    assert_source_files_documented(rows)

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

    if as_float(mask_corrected, "slots_mean") >= as_float(coverage_b3, "slots_mean"):
        fail("mask-corrected trade-off should reduce slots over uncorrected Coverage-Aware B=3")
    if as_float(mask_corrected, "failed_nodes_mean") >= as_float(coverage_b3, "failed_nodes_mean"):
        fail("mask-corrected trade-off should reduce failed nodes over uncorrected Coverage-Aware B=3")
    if as_float(mask_corrected, "missed_opportunities_mean") >= as_float(coverage_b3, "missed_opportunities_mean"):
        fail("mask-corrected trade-off should reduce missed opportunities over uncorrected Coverage-Aware B=3")
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


def assert_paper_table1_uncertainty():
    uncertainty_rows = read_csv(PAPER_TABLE1_SCENARIO_UNCERTAINTY)
    paired_rows = read_csv(PAPER_TABLE1_PAIRED_DELTAS)
    text = assert_exists(PAPER_TABLE1_UNCERTAINTY).read_text(encoding="utf-8")

    expected_metrics = {"Slots", "Failed", "Missed", "Gap"}
    expected_uncertainty_count = len(TABLE1_METHOD_ORDER) * len(expected_metrics)
    if len(uncertainty_rows) != expected_uncertainty_count:
        fail(
            f"{PAPER_TABLE1_SCENARIO_UNCERTAINTY} should have "
            f"{expected_uncertainty_count} rows, found {len(uncertainty_rows)}"
        )

    methods_by_metric = {
        (row["Method"], row["Metric"]): row for row in uncertainty_rows
    }
    for label in TABLE1_METHOD_ORDER:
        for metric in expected_metrics:
            row = methods_by_metric.get((label, metric))
            if row is None:
                fail(f"{PAPER_TABLE1_SCENARIO_UNCERTAINTY} missing {label} {metric}")
            assert_close(row, "ScenarioCount", 9.0)
            assert_close(row, "SamplesPerScenario", 900.0)
            assert_close(row, "Seeds", 3.0)
            if as_float(row, "ScenarioCI95") < 0.0:
                fail(f"{PAPER_TABLE1_SCENARIO_UNCERTAINTY} has negative CI for {label} {metric}")

    expected_comparisons = {
        "Mask-Corrected vs Coverage-Aware B=3",
        "Mask-Corrected vs Coverage-Aware B=4",
        "Mask-Corrected vs Sparse-TopK B=4",
    }
    if len(paired_rows) != len(expected_comparisons) * len(expected_metrics):
        fail(
            f"{PAPER_TABLE1_PAIRED_DELTAS} should have "
            f"{len(expected_comparisons) * len(expected_metrics)} rows, found {len(paired_rows)}"
        )
    for comparison in expected_comparisons:
        for metric in expected_metrics:
            row = row_by(
                rows_by(paired_rows, Comparison=comparison),
                "Metric",
                metric,
            )
            assert_close(row, "ScenarioCount", 9.0)
            if comparison == "Mask-Corrected vs Coverage-Aware B=3" and metric in {
                "Slots",
                "Failed",
                "Missed",
            }:
                if as_float(row, "MeanDelta") >= 0.0:
                    fail(f"{comparison} should improve lower-is-better metric {metric}")

    for snippet in (
        str(PAPER_TABLE1_SCENARIO_UNCERTAINTY),
        str(PAPER_TABLE1_PAIRED_DELTAS),
        "Claims of formal statistical dominance should wait",
    ):
        if snippet not in text:
            fail(f"{PAPER_TABLE1_UNCERTAINTY} missing snippet: {snippet!r}")
    return len(uncertainty_rows) + len(paired_rows)


def assert_paper_table2():
    rows = read_csv(PAPER_TABLE2)
    if len(rows) != 8:
        fail(f"{PAPER_TABLE2} should have 8 compact rows, found {len(rows)}")
    sparse = row_by(rows, "Setting", "Sparse-TopK B=4 sm=3")
    coverage_b4 = row_by(rows, "Setting", "Coverage-Aware B=4 cw=0.5 cpw=0")
    coverage_b3 = row_by(rows, "Setting", "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0")
    assert_close(sparse, "Preview", 16.0)
    assert_close(coverage_b4, "Preview", 16.0)
    assert_close(coverage_b3, "Preview", 16.0)
    if as_float(coverage_b4, "Missed") >= as_float(sparse, "Missed"):
        fail("Table 2 should show B4 coverage-aware missed reduction over Sparse-TopK")
    if as_float(coverage_b3, "Gap") >= as_float(coverage_b4, "Gap"):
        fail("Table 2 should show B3 budget split as the lower-gap same-preview row")
    text = assert_exists(PAPER_TABLE2_MD).read_text(encoding="utf-8")
    for snippet in (
        str(PAPER_TABLE2),
        "Compact coverage-aware ablation and budget-split evidence",
        "Full power-penalty and neighbor-coverage diagnostics remain",
    ):
        if snippet not in text:
            fail(f"{PAPER_TABLE2_MD} missing snippet: {snippet!r}")
    return len(rows)


def assert_paper_table3():
    rows = read_csv(PAPER_TABLE3)
    if len(rows) != 10:
        fail(f"{PAPER_TABLE3} should have overall plus 9 scenario rows, found {len(rows)}")
    overall = row_by(rows, "Scope", "overall")
    invitation_share = as_float(overall, "InvitationShare")
    competing_shares = [
        as_float(overall, "PoolShare"),
        as_float(overall, "SelectionShare"),
        as_float(overall, "ConfirmationShare"),
    ]
    if overall["DominantComponent"] != "invitation" or invitation_share <= max(competing_shares):
        fail("Table 3 should identify invitation as the dominant residual component")
    assert_close(overall, "InvitationShare", 0.687, tol=0.001)
    text = assert_exists(PAPER_TABLE3_MD).read_text(encoding="utf-8")
    for snippet in (
        str(PAPER_TABLE3),
        "overall invitation share is `0.687`",
        "This is a diagnostic table",
    ):
        if snippet not in text:
            fail(f"{PAPER_TABLE3_MD} missing snippet: {snippet!r}")
    return len(rows)


def assert_paper_figure1():
    source_text = assert_exists(PAPER_FIGURE1_SOURCE).read_text(encoding="utf-8")
    for snippet in (
        "flowchart TB",
        "Aggregate feedback count",
        "Mask correction",
        "Hidden current-channel oracle",
    ):
        if snippet not in source_text:
            fail(f"{PAPER_FIGURE1_SOURCE} missing Figure 1 source snippet: {snippet!r}")

    svg_text = assert_exists(PAPER_FIGURE1_SVG).read_text(encoding="utf-8", errors="replace")
    if "<svg" not in svg_text or "Aggregate feedback count" not in svg_text:
        fail(f"{PAPER_FIGURE1_SVG} does not look like the exported Figure 1 SVG")

    pdf_header = assert_exists(PAPER_FIGURE1_PDF).read_bytes()[:4]
    if pdf_header != b"%PDF":
        fail(f"{PAPER_FIGURE1_PDF} does not look like a PDF export")

    for path in (PAPER_FIGURE1_SVG, PAPER_FIGURE1_PDF):
        if project_path(path).stat().st_size <= 0:
            fail(f"empty paper Figure 1 artifact: {path}")


def assert_paper_figures():
    assert_paper_figure1()
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
    no_noise_tradeoff_count = 0
    high_noise_gap_best_count = 0
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
        no_noise_tradeoff_count += row["IsNoNoiseCorrectionTradeoff"] == "True"
        high_noise_gap_best_count += row["IsHighNoiseGapBest"] == "True"

    if no_noise_tradeoff_count != 1:
        fail(f"{PAPER_FIGURE4_POINTS} should mark exactly one no-noise correction trade-off point")
    if high_noise_gap_best_count != 1:
        fail(f"{PAPER_FIGURE4_POINTS} should mark exactly one high-noise gap-best point")

    for path in (PAPER_FIGURE4_GAP, PAPER_FIGURE4_FAILED_MISSED):
        full_path = assert_exists(path)
        if full_path.stat().st_size <= 0:
            fail(f"empty paper Figure 4 artifact: {path}")
    return len(rows)


def assert_main_frontier():
    rows = read_csv(MAIN_FRONTIER)
    assert_source_files_exist(rows)
    assert_source_files_documented(rows)
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
    noisy_b3 = rows_by(final_rows, label="Coverage-Aware B=3", feedback_noise_std="0.1")[0]

    for row in (reliable_b3, reliable_direct, noisy_direct, noisy_clipped):
        assert_close(row, "scenario_count", 9.0)
        assert_close(row, "episodes_per_scenario", 900.0)
        assert_close(row, "num_seeds", 3.0)
        assert_close(row, "decision_preview_calls_per_slot_mean", 16.0)

    if as_float(reliable_direct, "slots_mean") >= as_float(reliable_b3, "slots_mean"):
        fail("no-noise direct mask correction should reduce slots over B3")
    if as_float(reliable_direct, "failed_nodes_mean") >= as_float(reliable_b3, "failed_nodes_mean"):
        fail("no-noise direct mask correction should reduce failed nodes over B3")
    if as_float(reliable_direct, "missed_opportunities_mean") >= as_float(reliable_b3, "missed_opportunities_mean"):
        fail("no-noise direct mask correction should reduce missed opportunities over B3")
    if as_float(reliable_direct, "mask_correction_applied_rate") <= 0.0:
        fail("reliable direct mask correction should apply on at least some slots")
    if as_float(noisy_direct, "oracle_tx_gap_mean") >= as_float(noisy_b3, "oracle_tx_gap_mean"):
        fail("high-noise direct mask correction should improve gap over noisy B3")
    if as_float(noisy_direct, "missed_opportunities_mean") >= as_float(noisy_b3, "missed_opportunities_mean"):
        fail("high-noise direct mask correction should reduce missed opportunities over noisy B3")
    if as_float(noisy_clipped, "failed_nodes_mean") >= as_float(noisy_direct, "failed_nodes_mean"):
        fail("high-noise clipped diagnostic should reduce failed nodes over direct mc=1")

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
    readme_line_count = assert_readme_cleanup_boundary()
    candidate_boundary_count = assert_candidate_evidence_boundary()
    checked_rows += assert_final_summary()
    checked_rows += assert_paper_table1()
    checked_rows += assert_paper_table1_uncertainty()
    checked_rows += assert_paper_table2()
    checked_rows += assert_paper_table3()
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
    print(f"  candidate evidence boundary snippets verified: {candidate_boundary_count}")
    print(f"  README entry-point lines: {readme_line_count}")
    print(f"  checked CSV rows: {checked_rows}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"mainline artifact checks failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
