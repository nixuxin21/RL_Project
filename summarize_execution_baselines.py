"""从执行信道错配 CSV 中抽取主线基线和诊断方法，生成紧凑汇总表。"""

import argparse
import csv
import os
from collections import defaultdict

from evaluate_policy_comparison import ensure_parent_dir


DEFAULT_RESULTS_DIR = os.path.join("results", "execution_mismatch")
DEFAULT_CSV_OUTPUT = os.path.join(
    DEFAULT_RESULTS_DIR,
    "final_execution_baseline_summary.csv",
)
DEFAULT_MD_OUTPUT = os.path.join("docs", "EXECUTION_BASELINE_SUMMARY.md")

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
LEARNED_ABS_FILE = (
    "learned_sparse_shortlist_pilot_ep100_runs2_"
    "rho0p7-0p9-0p98_delay1-2-3_b4_ex1-2.csv"
)
LEARNED_SET_FILE = (
    "learned_set_shortlist_pilot_ep100_runs2_"
    "rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2.csv"
)
LEARNED_EXECUTION_FILE = (
    "learned_execution_value_shortlist_pilot_ep100_runs2_"
    "rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5.csv"
)
LEARNED_PAIRWISE_PC0_FILE = (
    "learned_pairwise_shortlist_pilot_ep100_runs2_"
    "rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5_pc0.csv"
)
LEARNED_PAIRWISE_PC0005_FILE = (
    "learned_pairwise_shortlist_pilot_ep100_runs2_"
    "rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5_pc0p005.csv"
)

FIELDNAMES = [
    "section",
    "label",
    "role",
    "source_file",
    "policy",
    "scenario_count",
    "episodes_per_scenario",
    "num_seeds",
    "success_mean",
    "perfect_rate",
    "slots_mean",
    "failed_nodes_mean",
    "missed_opportunities_mean",
    "decision_preview_calls_per_slot_mean",
    "oracle_tx_gap_mean",
    "adaptive_sparse_expansion_rate",
    "learned_shortlist_selected_extra_preview_mean",
]


def parse_args():
    """解析命令行参数，集中声明实验规模、策略配置、输入输出路径和绘图开关。"""
    parser = argparse.ArgumentParser(
        description="Summarize final execution-mismatch baselines."
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--md-output", default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def read_rows(path):
    """从给定路径读取 CSV 文件并返回字典行，供后续筛选、聚合和绘图使用。"""
    with open(path, newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def select_policy(rows, policy):
    """按照策略规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    selected = [row for row in rows if row.get("policy") == policy]
    if not selected:
        raise ValueError(f"No rows found for policy: {policy}")
    return selected


def mean(rows, key, default=0.0):
    """处理均值相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            values.append(float(default))
        else:
            values.append(float(value))
    return sum(values) / len(values) if values else float(default)


def first_int(rows, key):
    """处理first、整数相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    for row in rows:
        value = row.get(key, "")
        if value != "":
            return int(float(value))
    return 0


def summary_row(section, label, role, source_file, policy, rows):
    """处理摘要、结果行相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    return {
        "section": section,
        "label": label,
        "role": role,
        "source_file": source_file,
        "policy": policy,
        "scenario_count": len(rows),
        "episodes_per_scenario": first_int(rows, "episodes"),
        "num_seeds": first_int(rows, "num_seeds"),
        "success_mean": mean(rows, "success_mean"),
        "perfect_rate": mean(rows, "perfect_rate"),
        "slots_mean": mean(rows, "slots_mean"),
        "failed_nodes_mean": mean(rows, "failed_nodes_mean"),
        "missed_opportunities_mean": mean(rows, "missed_opportunities_mean"),
        "decision_preview_calls_per_slot_mean": mean(
            rows,
            "decision_preview_calls_per_slot_mean",
        ),
        "oracle_tx_gap_mean": mean(rows, "oracle_tx_gap_mean"),
        "adaptive_sparse_expansion_rate": mean(
            rows,
            "adaptive_sparse_expansion_rate",
        ),
        "learned_shortlist_selected_extra_preview_mean": mean(
            rows,
            "learned_shortlist_selected_extra_preview_mean",
        ),
    }


def add_policy(summary, section, label, role, results_dir, file_name, policy):
    """更新策略相关状态、历史记录或结果行，保证后续时隙和聚合阶段能继续累积信息。"""
    path = os.path.join(results_dir, file_name)
    rows = read_rows(path)
    summary.append(
        summary_row(
            section,
            label,
            role,
            file_name,
            policy,
            select_policy(rows, policy),
        )
    )


def build_summary(results_dir):
    """构建摘要所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    rows = []
    add_policy(
        rows,
        "main_frontier",
        "Rotating B=4",
        "low-budget reference",
        results_dir,
        FRONTIER_FILE,
        "Estimated Rotating Grid B=4",
    )
    add_policy(
        rows,
        "main_frontier",
        "Rotating B=8",
        "low-cost deployment baseline",
        results_dir,
        FRONTIER_FILE,
        "Estimated Rotating Grid B=8",
    )
    add_policy(
        rows,
        "main_frontier",
        "Sparse-TopK B=4 sm=2",
        "negative boundary",
        results_dir,
        FRONTIER_FILE,
        "Sparse-TopK Feedback Grid B=4 sm=2 tf=0.75",
    )
    add_policy(
        rows,
        "main_frontier",
        "Adaptive V2 mt=0.05 pc=0.005",
        "adaptive continuum point",
        results_dir,
        ADAPTIVE_V2_FILE,
        "Adaptive Sparse-TopK V2 Feedback Grid B=4 bm=2 em=3 mt=0.05 pc=0.005 tf=0.75",
    )
    add_policy(
        rows,
        "main_frontier",
        "Adaptive V2 mt=0.05 pc=0.002",
        "adaptive high-quality point",
        results_dir,
        ADAPTIVE_V2_FILE,
        "Adaptive Sparse-TopK V2 Feedback Grid B=4 bm=2 em=3 mt=0.05 pc=0.002 tf=0.75",
    )
    add_policy(
        rows,
        "main_frontier",
        "Sparse-TopK B=4 sm=3",
        "reportable medium-cost baseline",
        results_dir,
        FRONTIER_FILE,
        "Sparse-TopK Feedback Grid B=4 sm=3 tf=0.75",
    )
    add_policy(
        rows,
        "main_frontier",
        "Coverage-Aware B=4 cw=0.5 cpw=0",
        "same-cost B=4 coverage reference",
        results_dir,
        COVERAGE_SPARSE_FILE,
        "Coverage-Aware Sparse-TopK Feedback Grid B=4 sm=3 tf=0.75 cw=0.5 cpw=0",
    )
    add_policy(
        rows,
        "main_frontier",
        "Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0",
        "current budget-split refinement",
        results_dir,
        COVERAGE_BUDGET_SPLIT_FILE,
        "Coverage-Aware Sparse-TopK Feedback Grid B=3 sm=4.1 tf=0.75 cw=0.5 cpw=0",
    )
    add_policy(
        rows,
        "main_frontier",
        "Mask-Corrected Coverage-Aware B=3 mc=1",
        "slot/failed trade-off; no-noise gap regression",
        results_dir,
        INVITATION_MASK_CORRECTION_FILE,
        "Mask-Corrected Coverage-Aware B=3 mc=1",
    )
    add_policy(
        rows,
        "main_frontier",
        "Stale-TopK B=4",
        "high-cost positive reference",
        results_dir,
        FRONTIER_FILE,
        "Stale-TopK Feedback Grid B=4",
    )
    add_policy(
        rows,
        "main_frontier",
        "Temporal Deviation Oracle B=4",
        "hidden-info temporal diagnostic",
        results_dir,
        FRONTIER_FILE,
        "Temporal Deviation Oracle B=4",
    )

    learned_specs = [
        (
            LEARNED_ABS_FILE,
            "Learned absolute ex=2",
            "best learned diagnostic",
            "Learned Sparse Shortlist Feedback Grid B=4 bm=2 ex=2 tf=0.75",
        ),
        (
            LEARNED_SET_FILE,
            "Learned hidden set maxex=2",
            "negative set-value diagnostic",
            "Learned Set Shortlist Feedback Grid B=4 bm=2 maxex=2 tf=0.75",
        ),
        (
            LEARNED_EXECUTION_FILE,
            "Learned execution-value maxex=2",
            "closed-loop label diagnostic",
            "Learned Set Shortlist Feedback Grid B=4 bm=2 maxex=2 tf=0.75",
        ),
        (
            LEARNED_PAIRWISE_PC0_FILE,
            "Learned pairwise pc=0 maxex=2",
            "pairwise diagnostic",
            "Learned Set Shortlist Feedback Grid B=4 bm=2 maxex=2 tf=0.75",
        ),
        (
            LEARNED_PAIRWISE_PC0005_FILE,
            "Learned pairwise pc=0.005 maxex=2",
            "cost-aware pairwise diagnostic",
            "Learned Set Shortlist Feedback Grid B=4 bm=2 maxex=2 tf=0.75",
        ),
    ]
    for file_name, label, role, policy in learned_specs:
        add_policy(
            rows,
            "learned_diagnostics",
            label,
            role,
            results_dir,
            file_name,
            policy,
        )
    return rows


def write_csv(path, rows):
    """写出CSV结果，并统一字段顺序、目录创建和后续文档读取口径。"""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def format_num(value, digits=3):
    """格式化num显示文本，保证控制台、CSV 和 Markdown 中的数值表达一致。"""
    return f"{float(value):.{digits}f}"


def markdown_table(rows):
    """处理markdown、table相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    header = (
        "| Label | Role | Slots | Perfect % | Failed | Missed | Preview | "
        "Gap | Expansion | Extra |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {label} | {role} | {slots} | {perfect} | {failed} | {missed} | "
            "{preview} | {gap} | {expansion} | {extra} |".format(
                label=row["label"],
                role=row["role"],
                slots=format_num(row["slots_mean"]),
                perfect=format_num(row["perfect_rate"], 2),
                failed=format_num(row["failed_nodes_mean"]),
                missed=format_num(row["missed_opportunities_mean"]),
                preview=format_num(row["decision_preview_calls_per_slot_mean"], 2),
                gap=format_num(row["oracle_tx_gap_mean"]),
                expansion=format_num(row["adaptive_sparse_expansion_rate"]),
                extra=format_num(row["learned_shortlist_selected_extra_preview_mean"]),
            )
        )
    return "\n".join(lines)


def write_markdown(path, rows):
    """写出markdown结果，并统一字段顺序、目录创建和后续文档读取口径。"""
    ensure_parent_dir(path)
    sections = defaultdict(list)
    for row in rows:
        sections[row["section"]].append(row)

    content = [
        "# Execution Baseline Summary",
        "",
        "Generated by `summarize_execution_baselines.py` from existing CSV results.",
        "All values are equal-weight averages across the 9 temporal AR(1) rho/delay scenarios.",
        "",
        "## Main Frontier",
        "",
        markdown_table(sections["main_frontier"]),
        "",
        "## Learned Diagnostics",
        "",
        markdown_table(sections["learned_diagnostics"]),
        "",
        "## Interpretation",
        "",
        "- Main reportable baseline stack: `Rotating B=8`, adaptive v2 continuum, `Sparse-TopK B=4 sm=3`, `Coverage-Aware B=4`, `Coverage-Aware B=3 sm=4.1`, `Mask-Corrected Coverage-Aware B=3 mc=1`, `Stale-TopK B=4`, and `Temporal Deviation Oracle B=4`.",
        "- `Temporal Deviation Oracle B=4` is a hidden-information temporal diagnostic reference, not a global upper bound: the proposed mask-corrected method can have lower observed gap because it changes invitation-mask correction rather than only the temporal probe offset.",
        "- `Sparse-TopK B=4 sm=2` is a negative boundary: it no longer beats `Rotating B=8` after the formal frontier sweep.",
        "- `Mask-Corrected Coverage-Aware B=3 mc=1` is a same-preview invitation-mask correction trade-off: under the corrected temporal prehistory model it lowers slots, failed invitations, and missed opportunities versus B3, but it does not lower the no-noise oracle gap.",
        "- The feedback-noise sweep should be reported as a boundary result: direct `mc=1` improves gap under higher feedback noise, while `mc=1 clip=2` is a failed-invitation control diagnostic rather than the high-noise gap-best method.",
        "- Learned shortlist variants remain useful diagnostics, but the best learned point is still weaker than adaptive v2 and `Sparse-TopK sm=3`.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as mdfile:
        mdfile.write("\n".join(content))
    print(f"Saved: {path}")


def main():
    """脚本入口，负责串联参数解析、数据处理和结果写出。"""
    args = parse_args()
    rows = build_summary(args.results_dir)
    write_csv(args.csv_output, rows)
    write_markdown(args.md_output, rows)


if __name__ == "__main__":
    main()
