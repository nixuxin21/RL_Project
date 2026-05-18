"""评估信道估计误差对 IRS 选择和 AirComp 调度结果的影响。"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_partial_probing_sweep import (
    best_candidate,
    candidate_key,
    grid_indices,
)
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


POLICY_EXACT_GREEDY = "Exact Greedy Full Preview"
POLICY_EST_COUNT_ARGMAX = "Estimated Count Argmax"
POLICY_EST_GREEDY = "Estimated Greedy Full Preview"
POLICY_EST_ROTATING_B4 = "Estimated Rotating Grid B=4"
POLICY_EST_ROTATING_B8 = "Estimated Rotating Grid B=8"
POLICY_EST_RANDOM_B4 = "Estimated Random Probe B=4"


NUMERIC_RESULT_KEYS = (
    "success_nodes",
    "avg_power",
    "episode_reward",
    "slots_used",
    "total_energy",
    "decision_preview_calls_per_slot",
    "oracle_match_rate",
    "oracle_tx_gap_mean",
)


def parse_float_list(value):
    """解析浮点数、列表参数，通常把逗号分隔的命令行字符串转换成类型明确的 Python 列表。"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    """解析命令行参数，集中声明实验规模、策略配置、输入输出路径和开关选项。"""
    parser = argparse.ArgumentParser(
        description="Sweep equivalent-channel estimation error std for probing policies."
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--error-std-values", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--random-probe-budget", type=int, default=4)
    parser.add_argument("--rotating-probe-budgets", default="4,8")
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

    args.error_std_values = parse_float_list(args.error_std_values)
    if not args.error_std_values:
        raise ValueError("--error-std-values must contain at least one value")
    if any(value < 0.0 for value in args.error_std_values):
        raise ValueError("--error-std-values must be non-negative")

    args.rotating_probe_budgets = [
        min(int(value), args.num_codebook_states)
        for value in parse_float_list(args.rotating_probe_budgets)
    ]
    if not args.rotating_probe_budgets:
        raise ValueError("--rotating-probe-budgets must contain at least one value")
    if any(value <= 0 for value in args.rotating_probe_budgets):
        raise ValueError("--rotating-probe-budgets must be positive")
    args.random_probe_budget = min(args.random_probe_budget, args.num_codebook_states)
    if args.random_probe_budget <= 0:
        raise ValueError("--random-probe-budget must be positive")


def resolve_output_prefix(args):
    """处理resolve、输出、前缀相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    error_label = "-".join(format_float_for_suffix(value) for value in args.error_std_values)
    suffix = f"ep{args.episodes}_runs{args.num_seeds}_{seed_label}_err{error_label}"
    output_prefix = os.path.join(
        "results",
        "channel_estimation",
        f"channel_estimation_error_sweep_{suffix}",
    )
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


def make_error_rng(episode_seed, error_std):
    """构建error、随机数流所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    if episode_seed is None:
        return np.random.default_rng()
    error_tag = int(round(float(error_std) * 1_000_000))
    seed = (int(episode_seed) + 0x6A09E667 + error_tag * 0x9E3779B1) % (2**32)
    return np.random.default_rng(seed)


def make_random_probe_rng(episode_seed, error_std):
    """构建随机、probe、随机数流所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    if episode_seed is None:
        return np.random.default_rng()
    error_tag = int(round(float(error_std) * 1_000_000))
    seed = (int(episode_seed) + 0xBB67AE85 + error_tag * 0x85EBCA6B) % (2**32)
    return np.random.default_rng(seed)


def exact_preview_candidates(env, args, indices):
    """处理exact、预览、候选集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    return [
        env.preview_codebook_index(index, args.g_th, args.alpha_th)
        for index in indices
    ]


def effective_channels(env, indices):
    """计算给定 IRS 索引集合下的等效信道；无 IRS 时只返回直达链路。"""
    clean_indices = np.asarray(indices, dtype=int)
    weighted_reflection = env.h_r * env.h_bs_r
    cascade = weighted_reflection @ env.codebook[clean_indices].T
    cascade = (cascade.T / np.sqrt(env.M)) * 0.05
    return env.h_d[np.newaxis, :] + cascade


def estimated_preview_candidates(env, args, indices, error_std, rng):
    """处理估计、预览、候选集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    clean_indices = [int(np.clip(index, 0, args.num_codebook_states - 1)) for index in indices]
    h_total = effective_channels(env, clean_indices)
    if error_std > 0.0:
        rms = np.sqrt(np.mean(np.abs(h_total) ** 2, axis=1, keepdims=True))
        noise = (
            rng.normal(size=h_total.shape)
            + 1j * rng.normal(size=h_total.shape)
        ) / np.sqrt(2.0)
        h_total = h_total + float(error_std) * np.maximum(rms, 1e-12) * noise

    h_gain = np.abs(h_total) ** 2
    remaining = ~env.transmitted_flags
    required_power = (args.alpha_th**2) / (h_gain + 1e-12)
    tx_mask = (h_gain >= args.g_th) & remaining[np.newaxis, :]
    final_tx_mask = tx_mask & (required_power <= env.P_max)

    candidates = []
    for row_idx, index in enumerate(clean_indices):
        mask = final_tx_mask[row_idx]
        tx_count = int(np.sum(mask))
        power_avg = float(np.mean(required_power[row_idx, mask])) if tx_count > 0 else 0.0
        mean_gain_remaining = float(np.mean(h_gain[row_idx, remaining])) if np.any(remaining) else 0.0
        candidates.append(
            {
                "irs_index": int(index),
                "tx_this_slot": tx_count,
                "power_avg": power_avg,
                "mean_gain_remaining": mean_gain_remaining,
            }
        )
    return candidates


def choose_count_argmax(candidates):
    """按照count、argmax规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    return max(candidates, key=lambda candidate: int(candidate["tx_this_slot"]))


def policy_indices(policy_name, args, slot_idx, random_rng):
    """处理策略、索引集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if policy_name in {
        POLICY_EXACT_GREEDY,
        POLICY_EST_COUNT_ARGMAX,
        POLICY_EST_GREEDY,
    }:
        return list(range(args.num_codebook_states)), args.num_codebook_states
    if policy_name == POLICY_EST_ROTATING_B4:
        budget = min(4, args.num_codebook_states)
        return grid_indices(args.num_codebook_states, budget, offset=slot_idx), budget
    if policy_name == POLICY_EST_ROTATING_B8:
        budget = min(8, args.num_codebook_states)
        return grid_indices(args.num_codebook_states, budget, offset=slot_idx), budget
    if policy_name == POLICY_EST_RANDOM_B4:
        budget = min(args.random_probe_budget, args.num_codebook_states)
        return [int(index) for index in random_rng.choice(args.num_codebook_states, size=budget, replace=False)], budget
    raise ValueError(f"Unknown policy: {policy_name}")


def choose_policy_candidate(env, args, policy_name, error_std, slot_idx, error_rng, random_rng):
    """按照策略、候选规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    indices, preview_budget = policy_indices(policy_name, args, slot_idx, random_rng)
    if policy_name == POLICY_EXACT_GREEDY:
        candidates = exact_preview_candidates(env, args, indices)
        return best_candidate(candidates), preview_budget

    candidates = estimated_preview_candidates(env, args, indices, error_std, error_rng)
    if policy_name == POLICY_EST_COUNT_ARGMAX:
        return choose_count_argmax(candidates), preview_budget
    return best_candidate(candidates), preview_budget


def evaluate_policy(args, episode_seeds, error_std, policy_name, base_action):
    """评估单个策略配置在当前场景下的表现，返回后续聚合和报告生成所需的指标。"""
    env = make_env(args)
    success_nodes = []
    avg_power = []
    rewards = []
    tx_per_slot = []
    slots_used = []
    total_energy = []
    preview_calls_per_slot = []
    oracle_match_rate = []
    oracle_tx_gap_mean = []

    print(f"Running {policy_name} (error_std={error_std:g})...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        error_rng = make_error_rng(episode_seed, error_std)
        random_rng = make_random_probe_rng(episode_seed, error_std)
        episode_power = []
        episode_reward = 0.0
        episode_tx = []
        episode_slots = args.num_slots
        episode_energy = 0.0
        episode_preview_calls = []
        episode_oracle_matches = []
        episode_oracle_gaps = []
        total_tx = 0

        for slot_idx in range(args.num_slots):
            oracle = best_candidate(exact_preview_candidates(env, args, range(args.num_codebook_states)))
            selected, preview_budget = choose_policy_candidate(
                env,
                args,
                policy_name,
                error_std,
                slot_idx,
                error_rng,
                random_rng,
            )
            selected_index = int(selected["irs_index"])
            selected_true = env.preview_codebook_index(selected_index, args.g_th, args.alpha_th)

            episode_preview_calls.append(int(preview_budget))
            episode_oracle_matches.append(float(selected_index == int(oracle["irs_index"])))
            episode_oracle_gaps.append(
                max(0.0, float(oracle["tx_this_slot"]) - float(selected_true["tx_this_slot"]))
            )

            action = base_action.copy()
            action[2] = codebook_index_to_action(selected_index, args.num_codebook_states)
            _obs, reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            episode_tx.append(int(info["tx_this_slot"]))
            episode_reward += float(reward)
            episode_slots = int(info.get("slots_used", len(episode_tx)))
            episode_energy = update_energy(episode_energy, info)

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
        active_slots = max(len(episode_preview_calls), 1)
        preview_calls_per_slot.append(float(sum(episode_preview_calls)) / active_slots)
        oracle_match_rate.append(float(np.mean(episode_oracle_matches)) if episode_oracle_matches else 0.0)
        oracle_tx_gap_mean.append(float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0)

        print_progress(policy_name, error_std, ep, args.episodes, success_nodes, args.num_nodes)

    return {
        "name": policy_name,
        "error_std": float(error_std),
        "success_nodes": np.asarray(success_nodes, dtype=float),
        "avg_power": np.asarray(avg_power, dtype=float),
        "episode_reward": np.asarray(rewards, dtype=float),
        "tx_per_slot": tx_per_slot,
        "slots_used": np.asarray(slots_used, dtype=float),
        "total_energy": np.asarray(total_energy, dtype=float),
        "decision_preview_calls_per_slot": np.asarray(preview_calls_per_slot, dtype=float),
        "oracle_match_rate": np.asarray(oracle_match_rate, dtype=float),
        "oracle_tx_gap_mean": np.asarray(oracle_tx_gap_mean, dtype=float),
    }


def print_progress(policy_name, error_std, ep, episodes, success_nodes, num_nodes):
    """按 10% 进度间隔打印实验状态，避免长实验运行时没有可见反馈。"""
    interval = max(episodes // 10, 1)
    if ep % interval == 0 or ep == episodes:
        recent = np.mean(success_nodes[-interval:])
        print(
            f"  {policy_name} err={error_std:g}: "
            f"[{ep:04d}/{episodes:04d}] recent success {recent:.2f}/{num_nodes}"
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
            "error_std": parts[0]["error_std"],
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
                "error_std": float(result["error_std"]),
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
        "error_std",
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
    print("=" * 140)
    print("Channel Estimation Error Sweep Summary")
    print("=" * 140)
    print(
        f"{'ErrStd':>7} {'Policy':<32} {'Success':>9} {'Rate':>7} "
        f"{'Perfect%':>9} {'Slots':>8} {'Energy':>10} {'Preview':>9} {'Oracle%':>9} {'Gap':>7}"
    )
    for row in rows:
        print(
            f"{row['error_std']:>7.3f} {row['policy']:<32} "
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.get_cmap("tab10")
    colors = {policy: cmap(idx % 10) for idx, policy in enumerate(policies)}

    for policy in policies:
        policy_rows = sorted(
            [row for row in rows if row["policy"] == policy],
            key=lambda row: row["error_std"],
        )
        x = [row["error_std"] for row in policy_rows]
        axes[0].plot(
            x,
            [row["perfect_rate"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[1].plot(
            x,
            [row["slots_mean"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[2].plot(
            x,
            [row["oracle_tx_gap_mean"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )

    axes[0].set_title("Perfect Coverage vs Estimation Error")
    axes[0].set_ylabel("Perfect Episodes (%)")
    axes[0].set_ylim(0.0, 103.0)
    axes[1].set_title("Latency vs Estimation Error")
    axes[1].set_ylabel("Slots Used")
    axes[1].set_ylim(0.0, args.num_slots + 1)
    axes[2].set_title("Per-Slot Oracle Tx Gap")
    axes[2].set_ylabel("Missed Tx Count")

    for ax in axes:
        ax.set_xlabel("Equivalent Channel Error Std")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = f"{output_prefix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def policy_suite(args):
    """处理策略、suite相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    policies = [
        POLICY_EXACT_GREEDY,
        POLICY_EST_COUNT_ARGMAX,
        POLICY_EST_GREEDY,
    ]
    if 4 in args.rotating_probe_budgets:
        policies.append(POLICY_EST_ROTATING_B4)
    if 8 in args.rotating_probe_budgets:
        policies.append(POLICY_EST_ROTATING_B8)
    policies.append(POLICY_EST_RANDOM_B4)
    return policies


def main():
    """脚本入口：串联参数解析、实验执行、结果聚合和文件输出。"""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)
    base_action = make_base_action(args)
    run_seeds = make_run_seeds(args)
    episode_seed_sets = [make_episode_seeds(args, run_seed) for run_seed in run_seeds]
    policies = policy_suite(args)

    print("=" * 96)
    print(
        f"Channel estimation error sweep: episodes={args.episodes}, "
        f"num_seeds={args.num_seeds}, errors={args.error_std_values}"
    )
    print(f"Output prefix: {output_prefix}")
    print("=" * 96)

    all_rows = []
    for error_std in args.error_std_values:
        print("=" * 96)
        print(f"Equivalent channel estimation error std={error_std:g}")
        print("=" * 96)
        seed_result_sets = []
        for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
            print(f"Run seed [{run_idx}/{len(run_seeds)}]: {run_seeds[run_idx - 1]}")
            seed_results = [
                evaluate_policy(args, episode_seeds, error_std, policy_name, base_action)
                for policy_name in policies
            ]
            seed_result_sets.append(seed_results)
        all_rows.extend(summarize_results(args, aggregate_seed_results(seed_result_sets)))

    print_summary(all_rows)
    write_csv(f"{output_prefix}.csv", all_rows)
    if not args.no_plots:
        plot_results(all_rows, args, output_prefix)


if __name__ == "__main__":
    main()
