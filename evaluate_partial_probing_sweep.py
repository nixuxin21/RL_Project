"""评估 partial probing 策略，比较只预览少量码本索引时的成本和性能。"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_policy_comparison import (
    codebook_index_to_action,
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
    physical_to_action,
    update_energy,
)
from test_env import MSAirCompEnv


NUMERIC_RESULT_KEYS = (
    "success_nodes",
    "avg_power",
    "episode_reward",
    "slots_used",
    "total_energy",
    "decision_preview_calls_per_slot",
    "candidate_count_mean",
    "oracle_match_rate",
    "oracle_tx_gap_mean",
)


POLICY_RANDOM = "Random Probe"
POLICY_FIXED_GRID = "Fixed Grid Probe"
POLICY_ROTATING_GRID = "Rotating Grid Probe"
POLICY_LOCAL = "Local Probe"
POLICY_HYBRID = "Hybrid Local+Grid Probe"
POLICY_GREEDY = "Greedy Full Preview"


POLICY_OFFSETS = {
    POLICY_RANDOM: 0x243F6A88,
    POLICY_FIXED_GRID: 0x85A308D3,
    POLICY_ROTATING_GRID: 0x13198A2E,
    POLICY_LOCAL: 0x03707344,
    POLICY_HYBRID: 0xA4093822,
    POLICY_GREEDY: 0x299F31D0,
}


def parse_int_list(value):
    """解析整数、列表参数，通常把逗号分隔的命令行字符串转换成类型明确的 Python 列表。"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    """解析命令行参数，集中声明实验规模、策略配置、输入输出路径和开关选项。"""
    parser = argparse.ArgumentParser(
        description="Sweep per-slot IRS codebook preview budgets for partial probing policies."
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--probe-budgets", default="1,2,4,8")
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """校验解析后的命令行参数，尽早拒绝非法规模、预算或概率配置。"""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    for name in ("num_nodes", "num_slots", "num_irs_elements", "num_codebook_states"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")

    budgets = parse_int_list(args.probe_budgets)
    if not budgets:
        raise ValueError("--probe-budgets must contain at least one value")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("--probe-budgets must contain positive integers")

    clipped = sorted({min(int(budget), args.num_codebook_states) for budget in budgets})
    args.probe_budgets = clipped


def resolve_output_prefix(args):
    """处理resolve、输出、前缀相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    budget_label = "-".join(format_float_for_suffix(budget) for budget in args.probe_budgets)
    suffix = f"ep{args.episodes}_runs{args.num_seeds}_{seed_label}_b{budget_label}"
    output_prefix = os.path.join("results", "partial_probing", f"partial_probing_sweep_{suffix}")
    ensure_parent_dir(output_prefix)
    return output_prefix


def make_env(args):
    """构建env所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode="codebook",
    )


def make_base_action(args):
    """构建base、action所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    g_action = physical_to_action(args.g_th, low=0.001, scale=0.05)
    alpha_action = physical_to_action(args.alpha_th, low=0.05, scale=0.05)
    return np.array([g_action, alpha_action, 0.0], dtype=np.float32)


def candidate_key(candidate):
    """处理候选、排序键相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    tx_count = int(candidate["tx_this_slot"])
    power_avg = float(candidate["power_avg"])
    mean_gain = float(candidate["mean_gain_remaining"])
    power_tiebreak = -power_avg if tx_count > 0 else 0.0
    return tx_count, power_tiebreak, mean_gain


def best_candidate(candidates):
    """处理best、候选相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    return max(candidates, key=candidate_key)


def preview_indices(env, args, indices):
    """处理预览、索引集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    clean_indices = unique_fill(indices, len(indices), args.num_codebook_states)
    return [
        env.preview_codebook_index(index, args.g_th, args.alpha_th)
        for index in clean_indices
    ]


def full_greedy_candidate(env, args):
    """处理full、贪心、候选相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    return best_candidate(
        [
            env.preview_codebook_index(index, args.g_th, args.alpha_th)
            for index in range(args.num_codebook_states)
        ]
    )


def stable_probe_rng(episode_seed, policy_name, budget):
    """处理稳定、probe、随机数流相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if episode_seed is None:
        return np.random.default_rng()
    offset = POLICY_OFFSETS[policy_name]
    seed = (int(episode_seed) + offset + int(budget) * 0x9E3779B1) % (2**32)
    return np.random.default_rng(seed)


def unique_fill(indices, budget, num_codebook_states):
    """按输入优先级去重选择码本索引；候选不足时用未出现过的索引确定性补齐预算。"""
    selected = []
    seen = set()
    for index in indices:
        clean_index = int(index) % num_codebook_states
        if clean_index not in seen:
            selected.append(clean_index)
            seen.add(clean_index)
        if len(selected) >= budget:
            return selected

    for index in range(num_codebook_states):
        if index not in seen:
            selected.append(index)
            seen.add(index)
        if len(selected) >= budget:
            break
    return selected


def grid_indices(num_codebook_states, budget, offset=0):
    """在离散码本环上近似均匀抽取索引，并用 offset 实现随时隙轮换。"""
    budget = min(int(budget), num_codebook_states)
    if budget >= num_codebook_states:
        return list(range(num_codebook_states))
    raw = np.floor(np.arange(budget) * num_codebook_states / budget).astype(int)
    return unique_fill(raw + int(offset), budget, num_codebook_states)


def local_indices(center, budget, num_codebook_states):
    """处理local、索引集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    offsets = [0]
    radius = 1
    while len(offsets) < budget:
        offsets.append(-radius)
        if len(offsets) >= budget:
            break
        offsets.append(radius)
        radius += 1
    return unique_fill([int(center) + offset for offset in offsets], budget, num_codebook_states)


def select_probe_indices(policy_name, args, budget, slot_idx, state, rng):
    """按照probe、索引集合规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    c_count = args.num_codebook_states
    budget = min(int(budget), c_count)
    if budget >= c_count:
        return list(range(c_count))

    if policy_name == POLICY_RANDOM:
        return [int(index) for index in rng.choice(c_count, size=budget, replace=False)]

    if policy_name == POLICY_FIXED_GRID:
        return grid_indices(c_count, budget, offset=0)

    if policy_name == POLICY_ROTATING_GRID:
        return grid_indices(c_count, budget, offset=slot_idx)

    previous = state.get("previous_index")
    if policy_name == POLICY_LOCAL:
        if previous is None:
            return grid_indices(c_count, budget, offset=slot_idx)
        return local_indices(previous, budget, c_count)

    if policy_name == POLICY_HYBRID:
        if previous is None or budget == 1:
            return grid_indices(c_count, budget, offset=slot_idx)
        local_budget = max(1, budget // 2)
        explore_budget = budget - local_budget
        local_part = local_indices(previous, local_budget, c_count)
        explore_part = grid_indices(c_count, explore_budget, offset=slot_idx + local_budget)
        return unique_fill(local_part + explore_part, budget, c_count)

    raise ValueError(f"Unknown probing policy: {policy_name}")


def evaluate_probe_policy(episode_seeds, args, budget, policy_name, base_action):
    """评估单个 probing 策略配置，返回后续聚合和报告生成所需的指标。"""
    env = make_env(args)
    success_nodes = []
    avg_power = []
    rewards = []
    tx_per_slot = []
    slots_used = []
    total_energy = []
    preview_calls_per_slot = []
    candidate_count_mean = []
    oracle_match_rate = []
    oracle_tx_gap_mean = []

    print(f"Running {policy_name} (B={budget})...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        rng = stable_probe_rng(episode_seed, policy_name, budget)
        state = {"previous_index": None}
        episode_power = []
        episode_reward = 0.0
        episode_tx = []
        episode_slots = args.num_slots
        episode_energy = 0.0
        episode_candidate_counts = []
        episode_oracle_matches = []
        episode_oracle_gaps = []
        total_tx = 0

        for slot_idx in range(args.num_slots):
            oracle = full_greedy_candidate(env, args)
            if budget >= args.num_codebook_states:
                candidates = [oracle]
                selected = oracle
                candidate_count = args.num_codebook_states
            else:
                indices = select_probe_indices(policy_name, args, budget, slot_idx, state, rng)
                candidates = preview_indices(env, args, indices[:budget])
                selected = best_candidate(candidates)
                candidate_count = len(candidates)
            selected_index = int(selected["irs_index"])

            episode_candidate_counts.append(candidate_count)
            episode_oracle_matches.append(float(selected_index == int(oracle["irs_index"])))
            episode_oracle_gaps.append(
                max(0.0, float(oracle["tx_this_slot"]) - float(selected["tx_this_slot"]))
            )

            action = base_action.copy()
            action[2] = codebook_index_to_action(selected_index, args.num_codebook_states)
            _obs, reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            episode_tx.append(int(info["tx_this_slot"]))
            episode_reward += float(reward)
            episode_slots = int(info.get("slots_used", len(episode_tx)))
            episode_energy = update_energy(episode_energy, info)
            state["previous_index"] = selected_index

            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            if terminated or truncated:
                break

        success_nodes.append(total_tx)
        avg_power.append(float(np.mean(episode_power)) if episode_power else 0.0)
        rewards.append(float(episode_reward))
        tx_per_slot.append(episode_tx)
        slots_used.append(int(episode_slots))
        total_energy.append(float(episode_energy))
        active_slots = max(len(episode_candidate_counts), 1)
        preview_calls_per_slot.append(float(sum(episode_candidate_counts)) / active_slots)
        candidate_count_mean.append(float(np.mean(episode_candidate_counts)) if episode_candidate_counts else 0.0)
        oracle_match_rate.append(float(np.mean(episode_oracle_matches)) if episode_oracle_matches else 0.0)
        oracle_tx_gap_mean.append(float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0)

        print_progress(policy_name, budget, ep, args.episodes, success_nodes, args.num_nodes)

    return {
        "name": policy_name,
        "probe_budget": int(budget),
        "success_nodes": np.asarray(success_nodes, dtype=float),
        "avg_power": np.asarray(avg_power, dtype=float),
        "episode_reward": np.asarray(rewards, dtype=float),
        "tx_per_slot": tx_per_slot,
        "slots_used": np.asarray(slots_used, dtype=float),
        "total_energy": np.asarray(total_energy, dtype=float),
        "decision_preview_calls_per_slot": np.asarray(preview_calls_per_slot, dtype=float),
        "candidate_count_mean": np.asarray(candidate_count_mean, dtype=float),
        "oracle_match_rate": np.asarray(oracle_match_rate, dtype=float),
        "oracle_tx_gap_mean": np.asarray(oracle_tx_gap_mean, dtype=float),
    }


def print_progress(name, budget, ep, episodes, success_nodes, num_nodes):
    """按 10% 进度间隔打印实验状态，避免长实验运行时没有可见反馈。"""
    interval = max(episodes // 10, 1)
    if ep % interval == 0 or ep == episodes:
        recent = np.mean(success_nodes[-interval:])
        print(f"  {name} B={budget}: [{ep:04d}/{episodes:04d}] recent success {recent:.2f}/{num_nodes}")


def run_policy_suite_for_budget(args, episode_seeds, budget, base_action):
    """运行策略、suite、for、预算流程，串联参数解析、实验执行、结果聚合和文件输出。"""
    policies = [
        POLICY_RANDOM,
        POLICY_FIXED_GRID,
        POLICY_ROTATING_GRID,
        POLICY_LOCAL,
        POLICY_HYBRID,
    ]
    return [
        evaluate_probe_policy(episode_seeds, args, budget, policy_name, base_action)
        for policy_name in policies
    ]


def evaluate_greedy_upper_bound(args, episode_seeds, base_action):
    """评估贪心、upper、bound对应的策略或实验配置，返回后续聚合和报告生成所需的指标。"""
    return evaluate_probe_policy(
        episode_seeds,
        args,
        args.num_codebook_states,
        POLICY_GREEDY,
        base_action,
    )


def seed_summary(result):
    """把单个 run seed 的逐 episode 结果压缩为 seed-level 均值，供多 seed 聚合使用。"""
    return {key: float(np.mean(result[key])) for key in NUMERIC_RESULT_KEYS}


def aggregate_seed_results(seed_result_sets):
    """跨 run seed 聚合同一策略的结果，得到可写入 CSV 的稳定统计量。"""
    if not seed_result_sets:
        return []

    aggregated_results = []
    result_count = len(seed_result_sets[0])
    for result_idx in range(result_count):
        parts = [seed_results[result_idx] for seed_results in seed_result_sets]
        aggregated = {
            "name": parts[0]["name"],
            "probe_budget": parts[0]["probe_budget"],
            "tx_per_slot": [],
        }
        for key in NUMERIC_RESULT_KEYS:
            aggregated[key] = np.concatenate([part[key] for part in parts])
        for part in parts:
            aggregated["tx_per_slot"].extend(part["tx_per_slot"])
        aggregated["seed_summaries"] = [seed_summary(part) for part in parts]
        aggregated_results.append(aggregated)
    return aggregated_results


def metric_mean_ci(result, key):
    """计算跨 run seed 的总体均值和 95% 置信区间，用于结果表中的不确定性展示。"""
    seed_values = np.asarray(
        [summary[key] for summary in result.get("seed_summaries", [seed_summary(result)])],
        dtype=float,
    )
    mean_value = float(np.mean(result[key]))
    if len(seed_values) <= 1:
        return mean_value, 0.0
    ci95 = 1.96 * float(np.std(seed_values, ddof=1)) / np.sqrt(len(seed_values))
    return mean_value, ci95


def summarize_results(args, results):
    """把聚合后的结果转换成 CSV 行，统一字段名和后续分析脚本的读取口径。"""
    rows = []
    for result in results:
        success_mean, success_ci95 = metric_mean_ci(result, "success_nodes")
        slots_mean, slots_ci95 = metric_mean_ci(result, "slots_used")
        energy_mean, energy_ci95 = metric_mean_ci(result, "total_energy")
        rows.append(
            {
                "probe_budget": int(result["probe_budget"]),
                "budget_fraction": float(result["probe_budget"]) / args.num_codebook_states,
                "policy": result["name"],
                "episodes": len(result["success_nodes"]),
                "num_seeds": args.num_seeds,
                "success_mean": success_mean,
                "success_ci95": success_ci95,
                "success_rate_mean": success_mean / args.num_nodes,
                "perfect_rate": float(np.mean(result["success_nodes"] == args.num_nodes) * 100.0),
                "slots_mean": slots_mean,
                "slots_ci95": slots_ci95,
                "avg_power": float(np.mean(result["avg_power"])),
                "total_energy_mean": energy_mean,
                "total_energy_ci95": energy_ci95,
                "decision_preview_calls_per_slot_mean": float(
                    np.mean(result["decision_preview_calls_per_slot"])
                ),
                "candidate_count_mean": float(np.mean(result["candidate_count_mean"])),
                "oracle_match_rate": float(np.mean(result["oracle_match_rate"]) * 100.0),
                "oracle_tx_gap_mean": float(np.mean(result["oracle_tx_gap_mean"])),
                "avg_reward": float(np.mean(result["episode_reward"])),
            }
        )
    return rows


def write_csv(path, rows):
    """写出CSV结果，并统一字段顺序、目录创建和后续文档读取口径。"""
    ensure_parent_dir(path)
    fieldnames = [
        "probe_budget",
        "budget_fraction",
        "policy",
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
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def print_summary(rows):
    """处理摘要相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    print("=" * 138)
    print("Partial Codebook Probing Sweep Summary")
    print("=" * 138)
    print(
        f"{'B':>3} {'Policy':<28} {'Success':>9} {'Rate':>7} {'Perfect%':>9} "
        f"{'Slots':>8} {'Energy':>10} {'Preview':>9} {'Oracle%':>9} {'Gap':>7}"
    )
    for row in sorted(rows, key=lambda item: (item["probe_budget"], item["policy"])):
        print(
            f"{row['probe_budget']:>3} {row['policy']:<28} "
            f"{row['success_mean']:>9.3f} {row['success_rate_mean']:>7.3f} "
            f"{row['perfect_rate']:>8.2f}% {row['slots_mean']:>8.3f} "
            f"{row['total_energy_mean']:>10.3f} "
            f"{row['decision_preview_calls_per_slot_mean']:>9.2f} "
            f"{row['oracle_match_rate']:>8.2f}% {row['oracle_tx_gap_mean']:>7.3f}"
        )


def plot_results(rows, args, output_prefix):
    """绘制results图像，把聚合指标转换成论文或诊断文档可直接查看的图。"""
    policies = []
    for row in rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cmap = plt.get_cmap("tab10")
    colors = {policy: cmap(idx % 10) for idx, policy in enumerate(policies)}

    for policy in policies:
        policy_rows = sorted(
            [row for row in rows if row["policy"] == policy],
            key=lambda row: row["probe_budget"],
        )
        x = [row["probe_budget"] for row in policy_rows]
        axes[0, 0].plot(
            x,
            [row["success_rate_mean"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[0, 1].plot(
            x,
            [row["perfect_rate"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[1, 0].plot(
            x,
            [row["slots_mean"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[1, 1].plot(
            x,
            [row["oracle_tx_gap_mean"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )

    axes[0, 0].set_title("Success Rate vs Probe Budget")
    axes[0, 0].set_ylabel("Success Rate")
    axes[0, 0].set_ylim(0.0, 1.03)
    axes[0, 1].set_title("Perfect Coverage vs Probe Budget")
    axes[0, 1].set_ylabel("Perfect Episodes (%)")
    axes[0, 1].set_ylim(0.0, 103.0)
    axes[1, 0].set_title("Latency vs Probe Budget")
    axes[1, 0].set_ylabel("Slots Used")
    axes[1, 0].set_ylim(0.0, args.num_slots + 1)
    axes[1, 1].set_title("Per-Slot Oracle Tx Gap")
    axes[1, 1].set_ylabel("Missed Tx Count")

    for ax in axes.ravel():
        ax.set_xlabel("Preview Budget B")
        ax.set_xticks(sorted({row["probe_budget"] for row in rows}))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = f"{output_prefix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    """脚本入口：串联参数解析、实验执行、结果聚合和文件输出。"""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)
    base_action = make_base_action(args)
    run_seeds = make_run_seeds(args)
    episode_seed_sets = [make_episode_seeds(args, run_seed) for run_seed in run_seeds]

    print("=" * 96)
    print(
        f"Partial probing sweep: episodes={args.episodes}, num_seeds={args.num_seeds}, "
        f"base_seed={args.seed}, budgets={args.probe_budgets}"
    )
    print(f"Fixed transmission parameters: g_th={args.g_th}, alpha_th={args.alpha_th}")
    print(f"Output prefix: {output_prefix}")
    print("=" * 96)

    all_rows = []
    for budget in args.probe_budgets:
        print("=" * 96)
        print(f"Probe budget B={budget}/{args.num_codebook_states}")
        print("=" * 96)
        seed_result_sets = []
        for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
            print(f"Run seed [{run_idx}/{len(run_seeds)}]: {run_seeds[run_idx - 1]}")
            seed_result_sets.append(run_policy_suite_for_budget(args, episode_seeds, budget, base_action))
        all_rows.extend(summarize_results(args, aggregate_seed_results(seed_result_sets)))

    print("=" * 96)
    print("Full Greedy upper bound")
    print("=" * 96)
    greedy_seed_result_sets = []
    for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
        print(f"Run seed [{run_idx}/{len(run_seeds)}]: {run_seeds[run_idx - 1]}")
        greedy_seed_result_sets.append([evaluate_greedy_upper_bound(args, episode_seeds, base_action)])
    all_rows.extend(summarize_results(args, aggregate_seed_results(greedy_seed_result_sets)))

    print_summary(all_rows)
    write_csv(f"{output_prefix}.csv", all_rows)
    if not args.no_plots:
        plot_results(all_rows, args, output_prefix)


if __name__ == "__main__":
    main()
