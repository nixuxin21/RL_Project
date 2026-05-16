"""
K/N/M/C 参数扫描脚本。

这个脚本用于验证 rule-based IRS 策略在不同系统规模下是否稳定：

- K: 节点数量。
- N: episode 内可用时隙数量。
- M: IRS 反射单元数量。
- C: IRS 离散码本大小。

脚本采用“一因子变化”设计：每次只改变一个参数，其余参数保持 base 配置。
这样可以更清楚地判断性能变化来自哪个维度。当前比较的策略包括：

1. Feature Argmax IRS：直接选择 codebook quality feature 最大的码本。
2. Feature Argmax PowerTie IRS：在最大 feature 并列候选中选择平均功率更低的码本。
3. Greedy IRS：每个时隙 preview 全部 C 个码本，按调度数量、功率、剩余增益排序。
4. Random IRS：每个时隙随机 IRS 相位。
5. No IRS：关闭 IRS 级联链路。

输出包括 summary CSV、episode 级 CSV、slot 级 CSV，以及 success/latency/energy 三张图。
"""

import argparse
import csv
import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from test_env import MSAirCompEnv


# 策略名集中定义，避免 CSV、图例和控制台输出之间出现拼写不一致。
POLICY_FEATURE_ARGMAX = "Feature Argmax IRS"
POLICY_FEATURE_ARGMAX_POWER_TIE = "Feature Argmax PowerTie IRS"
POLICY_GREEDY = "Greedy IRS"
POLICY_RANDOM = "Random IRS"
POLICY_NO_IRS = "No IRS"
POLICIES = (
    POLICY_FEATURE_ARGMAX,
    POLICY_FEATURE_ARGMAX_POWER_TIE,
    POLICY_GREEDY,
    POLICY_RANDOM,
    POLICY_NO_IRS,
)
FEATURE_POLICIES = (POLICY_FEATURE_ARGMAX, POLICY_FEATURE_ARGMAX_POWER_TIE)


def parse_int_list(value):
    """把形如 `"30,50,80"` 的命令行字符串解析为整数列表。"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    """
    解析参数扫描配置。

    base 参数定义默认环境；`node/slot/irs-element/codebook-values`
    定义一因子扫描时每个维度要尝试的取值集合。
    """
    parser = argparse.ArgumentParser(
        description="Run K/N/M/C parameter sweeps for rule-based IRS baselines."
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--base-num-nodes", type=int, default=50)
    parser.add_argument("--base-num-slots", type=int, default=10)
    parser.add_argument("--base-num-irs-elements", type=int, default=64)
    parser.add_argument("--base-num-codebook-states", type=int, default=16)
    parser.add_argument("--node-values", default="30,50,80")
    parser.add_argument("--slot-values", default="5,10,15")
    parser.add_argument("--irs-element-values", default="32,64,128")
    parser.add_argument("--codebook-values", default="8,16,32")
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """对关键正整数参数做早期校验，避免进入长时间仿真后才失败。"""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    for name in (
        "base_num_nodes",
        "base_num_slots",
        "base_num_irs_elements",
        "base_num_codebook_states",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.base_num_codebook_states <= 1:
        raise ValueError("--base-num-codebook-states must be greater than 1")


def ensure_parent_dir(path):
    """确保输出文件所在目录存在；支持用户把输出写到 `/tmp/...` 或子目录。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_output_prefix(args):
    """生成所有 CSV/图像共享的输出前缀。"""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix
    output_prefix = os.path.join(
        "results",
        "parameter_sweep",
        f"parameter_sweep_ep{args.episodes}_seed{args.seed}",
    )
    ensure_parent_dir(output_prefix)
    return output_prefix


def physical_to_action(value, low, scale):
    """将真实物理值映射到环境动作空间 [-1, 1]。"""
    action_value = (value - low) / scale - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def codebook_index_to_action(index, num_codebook_states):
    """将离散码本索引映射到环境第三维连续动作。"""
    if num_codebook_states <= 1:
        return 0.0
    action_value = 2.0 * index / (num_codebook_states - 1) - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def make_base_action(args, config, irs_index=0):
    """
    构造三维环境动作 `[g_th, alpha_th, irs]`。

    本脚本所有 rule-based baseline 都固定 `g_th/alpha_th`，
    只改变第三维 IRS 码本索引或 IRS 模式。
    """
    return np.array(
        [
            physical_to_action(args.g_th, low=0.001, scale=0.05),
            physical_to_action(args.alpha_th, low=0.05, scale=0.05),
            codebook_index_to_action(irs_index, config["num_codebook_states"]),
        ],
        dtype=np.float32,
    )


def make_env(args, config, irs_phase_mode="codebook", include_codebook_features=False):
    """
    按当前 sweep config 创建环境。

    `include_codebook_features=True` 只给 Feature Argmax 系列策略使用，
    因为它们需要从观测中读取 C 维码本质量特征。
    """
    return MSAirCompEnv(
        num_nodes=config["num_nodes"],
        num_slots=config["num_slots"],
        num_irs_elements=config["num_irs_elements"],
        num_codebook_states=config["num_codebook_states"],
        irs_phase_mode=irs_phase_mode,
        include_codebook_features=include_codebook_features,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
    )


def make_episode_seeds(seed, episodes):
    """
    生成 episode 级随机种子列表。

    同一 config 内不同策略复用完全相同的 episode seeds，
    从而保证策略差异来自动作选择，而不是信道随机性。
    """
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**31 - 1, size=episodes)]


def make_sweep_configs(args):
    """
    构造一因子扫描配置列表。

    每个返回的 config 都包含实际环境参数以及 `config_id`，
    例如 `K50`、`N10`、`M64`、`C16`，便于 CSV 和图表分组。
    """
    base = {
        "num_nodes": args.base_num_nodes,
        "num_slots": args.base_num_slots,
        "num_irs_elements": args.base_num_irs_elements,
        "num_codebook_states": args.base_num_codebook_states,
    }
    dimensions = [
        ("K", "num_nodes", parse_int_list(args.node_values)),
        ("N", "num_slots", parse_int_list(args.slot_values)),
        ("M", "num_irs_elements", parse_int_list(args.irs_element_values)),
        ("C", "num_codebook_states", parse_int_list(args.codebook_values)),
    ]

    configs = []
    seen = set()
    for sweep_name, key, values in dimensions:
        for value in values:
            # 从 base 复制一份，只覆盖当前 sweep 维度，确保是一因子实验。
            config = dict(base)
            config[key] = value
            config["sweep"] = sweep_name
            config["sweep_value"] = value
            config["config_id"] = f"{sweep_name}{value}"
            dedupe_key = (config["config_id"], config["num_nodes"], config["num_slots"], config["num_irs_elements"], config["num_codebook_states"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            configs.append(config)
    return configs


def greedy_candidate(env, args, config):
    """
    返回当前时隙下 Greedy IRS 认为最好的码本候选。

    排序原则：
    1. 优先最大化本时隙可调度节点数 `tx_this_slot`。
    2. 若调度数量相同，优先选择平均发射功率更低的候选。
    3. 若仍相同，再用剩余节点平均信道增益作为稳定 tie-break。
    """
    candidates = [
        env.preview_codebook_index(codebook_index, args.g_th, args.alpha_th)
        for codebook_index in range(config["num_codebook_states"])
    ]

    def candidate_key(candidate):
        """Greedy 排序键：调度数量优先，其次低功率，最后剩余平均增益。"""
        tx_count = int(candidate["tx_this_slot"])
        power_avg = float(candidate["power_avg"])
        mean_gain = float(candidate["mean_gain_remaining"])
        power_tiebreak = -power_avg if tx_count > 0 else 0.0
        return tx_count, power_tiebreak, mean_gain

    return max(candidates, key=candidate_key)


def feature_argmax_index(obs, config):
    """从观测尾部 C 维 codebook features 中取最大值索引。"""
    features = obs[7 : 7 + config["num_codebook_states"]]
    return int(np.argmax(features))


def feature_argmax_candidates(obs, config):
    """
    找到所有与最大 codebook feature 并列的码本索引。

    因为 feature 是预计可调度节点比例，很多码本会拥有相同的最大调度数量；
    PowerTie 策略只在这些并列候选中额外比较功率。
    """
    features = obs[7 : 7 + config["num_codebook_states"]]
    max_feature = float(np.max(features))
    return np.flatnonzero(np.isclose(features, max_feature, rtol=0.0, atol=1e-7))


def feature_argmax_power_tie_index(env, args, config, obs):
    """
    Feature Argmax 的低复杂度能耗改进版本。

    先使用已经存在的 codebook features 找到 max-count 候选集合；
    只有当存在多个并列候选时，才对这些候选调用 `preview_codebook_index`
    读取平均功率，并选择 Greedy tie-break 下最优的候选。

    Returns:
        `(irs_index, preview_calls, tie_count)`：
        - `irs_index`: 最终选择的 IRS 码本索引；
        - `preview_calls`: 决策阶段额外 preview 的候选数量；
        - `tie_count`: max-count 并列候选数量。
    """
    candidate_indices = feature_argmax_candidates(obs, config)
    tie_count = int(len(candidate_indices))
    if tie_count <= 1:
        return int(candidate_indices[0]), 0, tie_count

    candidates = [
        env.preview_codebook_index(codebook_index, args.g_th, args.alpha_th)
        for codebook_index in candidate_indices
    ]

    def candidate_key(candidate):
        """PowerTie 排序键，与 Greedy 的 tie-break 规则保持一致。"""
        tx_count = int(candidate["tx_this_slot"])
        power_avg = float(candidate["power_avg"])
        mean_gain = float(candidate["mean_gain_remaining"])
        power_tiebreak = -power_avg if tx_count > 0 else 0.0
        return tx_count, power_tiebreak, mean_gain

    best_candidate = max(candidates, key=candidate_key)
    return int(best_candidate["irs_index"]), tie_count, tie_count


def update_energy(episode_energy, info):
    """累计能耗代理指标：本槽平均功率乘以本槽发送节点数。"""
    return episode_energy + float(info["power_avg"]) * int(info["tx_this_slot"])


def evaluate_policy(args, config, policy, episode_seeds):
    """
    在一个固定 sweep config 下评估单个策略。

    返回三类数据：
    - episode_rows: 每个 episode 的总体结果；
    - slot_tx / slot_total: 每个时隙的平均增量和累计调度情况；
    - slot_preview_calls / slot_tie_candidates: 决策复杂度相关统计。
    """
    include_features = policy in FEATURE_POLICIES
    irs_mode = "random" if policy == POLICY_RANDOM else "none" if policy == POLICY_NO_IRS else "codebook"
    env = make_env(args, config, irs_phase_mode=irs_mode, include_codebook_features=include_features)

    episode_rows = []
    slot_tx = [[] for _idx in range(config["num_slots"])]
    slot_total = [[] for _idx in range(config["num_slots"])]
    slot_preview_calls = [[] for _idx in range(config["num_slots"])]
    slot_tie_candidates = [[] for _idx in range(config["num_slots"])]

    for episode_idx, episode_seed in enumerate(episode_seeds, start=1):
        obs, _info = env.reset(seed=episode_seed)
        episode_reward = 0.0
        episode_power = []
        episode_energy = 0.0
        episode_preview_calls = 0
        episode_tie_candidates = []
        slots_used = config["num_slots"]
        total_tx = 0

        for slot in range(1, config["num_slots"] + 1):
            # 根据策略类型选择 IRS 码本索引，并同步记录该决策额外用了多少 preview。
            if policy == POLICY_FEATURE_ARGMAX:
                irs_index = feature_argmax_index(obs, config)
                preview_calls = 0
                tie_candidates = int(len(feature_argmax_candidates(obs, config)))
            elif policy == POLICY_FEATURE_ARGMAX_POWER_TIE:
                irs_index, preview_calls, tie_candidates = feature_argmax_power_tie_index(
                    env, args, config, obs
                )
            elif policy == POLICY_GREEDY:
                irs_index = int(greedy_candidate(env, args, config)["irs_index"])
                preview_calls = config["num_codebook_states"]
                tie_candidates = config["num_codebook_states"]
            else:
                # Random IRS 和 No IRS 不使用动作中的码本索引；这里填 0 只是占位。
                irs_index = 0
                preview_calls = 0
                tie_candidates = 0

            # 所有策略都通过同一个环境 step 执行，保证奖励、功率和调度统计口径一致。
            action = make_base_action(args, config, irs_index=irs_index)
            obs, reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            episode_reward += float(reward)
            slots_used = int(info.get("slots_used", slot))
            episode_energy = update_energy(episode_energy, info)
            episode_preview_calls += int(preview_calls)
            episode_tie_candidates.append(int(tie_candidates))
            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            slot_tx[slot - 1].append(int(info["tx_this_slot"]))
            slot_total[slot - 1].append(total_tx)
            slot_preview_calls[slot - 1].append(int(preview_calls))
            slot_tie_candidates[slot - 1].append(int(tie_candidates))

            if terminated or truncated:
                break

        for remaining_slot in range(slots_used + 1, config["num_slots"] + 1):
            # 如果 episode 提前完成，后续时隙补零，便于 slot 级平均曲线长度一致。
            slot_tx[remaining_slot - 1].append(0)
            slot_total[remaining_slot - 1].append(total_tx)
            slot_preview_calls[remaining_slot - 1].append(0)
            slot_tie_candidates[remaining_slot - 1].append(0)

        active_slots = max(len(episode_tie_candidates), 1)
        episode_rows.append(
            {
                "policy": policy,
                "episode_idx": episode_idx,
                "episode_seed": episode_seed,
                "success_nodes": total_tx,
                "success_rate": total_tx / config["num_nodes"],
                "perfect": int(total_tx == config["num_nodes"]),
                "missed_nodes": config["num_nodes"] - total_tx,
                "slots_used": slots_used,
                "slot_fraction": slots_used / config["num_slots"],
                "episode_reward": episode_reward,
                "avg_power": float(np.mean(episode_power)) if episode_power else 0.0,
                "total_energy": episode_energy,
                "decision_preview_calls": episode_preview_calls,
                "decision_preview_calls_per_slot": episode_preview_calls / active_slots,
                "tie_candidates_mean": mean(episode_tie_candidates),
            }
        )

    return episode_rows, slot_tx, slot_total, slot_preview_calls, slot_tie_candidates


def mean(values):
    """空列表安全均值，避免某些极端配置下没有可统计样本。"""
    return float(np.mean(values)) if values else 0.0


def std(values):
    """空列表安全标准差。"""
    return float(np.std(values)) if values else 0.0


def summarize_policy(config, policy, episode_rows, slot_tx, slot_total):
    """
    将 episode 级结果聚合为 summary CSV 的一行。

    这里同时保留覆盖率、时延、能耗、奖励和 preview 次数，
    便于从“性能”和“复杂度”两个角度评价 rule-based 策略。
    """
    success_nodes = [row["success_nodes"] for row in episode_rows]
    success_rates = [row["success_rate"] for row in episode_rows]
    perfect = [row["perfect"] for row in episode_rows]
    slots = [row["slots_used"] for row in episode_rows]
    slot_fraction = [row["slot_fraction"] for row in episode_rows]
    rewards = [row["episode_reward"] for row in episode_rows]
    powers = [row["avg_power"] for row in episode_rows]
    energies = [row["total_energy"] for row in episode_rows]
    preview_calls = [row["decision_preview_calls"] for row in episode_rows]
    preview_calls_per_slot = [row["decision_preview_calls_per_slot"] for row in episode_rows]
    tie_candidates = [row["tie_candidates_mean"] for row in episode_rows]

    return {
        "config_id": config["config_id"],
        "sweep": config["sweep"],
        "sweep_value": config["sweep_value"],
        "num_nodes": config["num_nodes"],
        "num_slots": config["num_slots"],
        "num_irs_elements": config["num_irs_elements"],
        "num_codebook_states": config["num_codebook_states"],
        "policy": policy,
        "episodes": len(episode_rows),
        "success_mean": mean(success_nodes),
        "success_rate_mean": mean(success_rates),
        "success_std": std(success_nodes),
        "perfect_rate": mean(perfect) * 100.0,
        "slots_mean": mean(slots),
        "slot_fraction_mean": mean(slot_fraction),
        "avg_power_mean": mean(powers),
        "total_energy_mean": mean(energies),
        "decision_preview_calls_mean": mean(preview_calls),
        "decision_preview_calls_per_slot_mean": mean(preview_calls_per_slot),
        "tie_candidates_mean": mean(tie_candidates),
        "episode_reward_mean": mean(rewards),
        "slot1_tx_mean": mean(slot_tx[0]) if slot_tx else 0.0,
        "slot2_total_mean": mean(slot_total[1]) if len(slot_total) > 1 else mean(slot_total[0]),
    }


def write_csv(path, rows, fieldnames):
    """按固定列顺序写 CSV，保证后续报告和画图脚本读取稳定。"""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def run_sweep(args):
    """
    执行完整参数扫描。

    外层循环遍历 K/N/M/C 的一因子配置；内层循环遍历策略。
    每个 config 使用独立但可复现的 seed 区间，避免不同 config 之间复用同一批信道。
    """
    configs = make_sweep_configs(args)
    summary_rows = []
    episode_rows_all = []
    slot_rows = []

    print("=" * 96)
    print(f"Parameter sweep: configs={len(configs)}, episodes={args.episodes}, seed={args.seed}")
    print("=" * 96)

    for config_idx, config in enumerate(configs, start=1):
        config_seed = args.seed + config_idx * 1000
        episode_seeds = make_episode_seeds(config_seed, args.episodes)
        print(
            f"[{config_idx:02d}/{len(configs):02d}] {config['config_id']} | "
            f"K={config['num_nodes']} N={config['num_slots']} "
            f"M={config['num_irs_elements']} C={config['num_codebook_states']}"
        )

        for policy in POLICIES:
            episode_rows, slot_tx, slot_total, slot_preview_calls, slot_tie_candidates = evaluate_policy(
                args, config, policy, episode_seeds
            )
            summary = summarize_policy(config, policy, episode_rows, slot_tx, slot_total)
            summary_rows.append(summary)
            for row in episode_rows:
                full_row = {
                    "config_id": config["config_id"],
                    "sweep": config["sweep"],
                    "sweep_value": config["sweep_value"],
                    "num_nodes": config["num_nodes"],
                    "num_slots": config["num_slots"],
                    "num_irs_elements": config["num_irs_elements"],
                    "num_codebook_states": config["num_codebook_states"],
                }
                full_row.update(row)
                episode_rows_all.append(full_row)

            for slot_idx in range(config["num_slots"]):
                slot_rows.append(
                    {
                        "config_id": config["config_id"],
                        "sweep": config["sweep"],
                        "sweep_value": config["sweep_value"],
                        "num_nodes": config["num_nodes"],
                        "num_slots": config["num_slots"],
                        "num_irs_elements": config["num_irs_elements"],
                        "num_codebook_states": config["num_codebook_states"],
                        "policy": policy,
                        "slot": slot_idx + 1,
                        "tx_mean_padded": mean(slot_tx[slot_idx]),
                        "total_tx_mean_padded": mean(slot_total[slot_idx]),
                        "decision_preview_calls_mean_padded": mean(slot_preview_calls[slot_idx]),
                        "tie_candidates_mean_padded": mean(slot_tie_candidates[slot_idx]),
                    }
                )

            print(
                f"  {policy:<20} success={summary['success_mean']:.2f}/{config['num_nodes']} "
                f"perfect={summary['perfect_rate']:.1f}% slots={summary['slots_mean']:.2f} "
                f"energy={summary['total_energy_mean']:.3f}"
            )

    return summary_rows, episode_rows_all, slot_rows


def plot_success(summary_rows, output_prefix):
    """绘制四个 sweep 维度下的成功率曲线。"""
    by_sweep = defaultdict(list)
    for row in summary_rows:
        by_sweep[row["sweep"]].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, sweep in zip(axes, ["K", "N", "M", "C"]):
        rows = by_sweep[sweep]
        values = sorted({row["sweep_value"] for row in rows})
        for policy in POLICIES:
            policy_rows = {row["sweep_value"]: row for row in rows if row["policy"] == policy}
            y = [policy_rows[value]["success_rate_mean"] for value in values]
            ax.plot(values, y, marker="o", linewidth=1.8, label=policy)
        ax.set_title(f"Sweep {sweep}: Success Rate")
        ax.set_xlabel(sweep)
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0.0, 1.03)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = f"{output_prefix}_success.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_latency(summary_rows, output_prefix):
    """绘制四个 sweep 维度下的时延比例曲线，即 `slots_used / N`。"""
    by_sweep = defaultdict(list)
    for row in summary_rows:
        by_sweep[row["sweep"]].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, sweep in zip(axes, ["K", "N", "M", "C"]):
        rows = by_sweep[sweep]
        values = sorted({row["sweep_value"] for row in rows})
        for policy in POLICIES:
            policy_rows = {row["sweep_value"]: row for row in rows if row["policy"] == policy}
            y = [policy_rows[value]["slot_fraction_mean"] for value in values]
            ax.plot(values, y, marker="o", linewidth=1.8, label=policy)
        ax.set_title(f"Sweep {sweep}: Slots Used / N")
        ax.set_xlabel(sweep)
        ax.set_ylabel("Slot Fraction")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = f"{output_prefix}_latency.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_energy(summary_rows, output_prefix):
    """绘制四个 sweep 维度下的总能耗代理指标曲线。"""
    by_sweep = defaultdict(list)
    for row in summary_rows:
        by_sweep[row["sweep"]].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, sweep in zip(axes, ["K", "N", "M", "C"]):
        rows = by_sweep[sweep]
        values = sorted({row["sweep_value"] for row in rows})
        for policy in POLICIES:
            policy_rows = {row["sweep_value"]: row for row in rows if row["policy"] == policy}
            y = [policy_rows[value]["total_energy_mean"] for value in values]
            ax.plot(values, y, marker="o", linewidth=1.8, label=policy)
        ax.set_title(f"Sweep {sweep}: Total Energy Proxy")
        ax.set_xlabel(sweep)
        ax.set_ylabel("Sum Power Across Slots")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = f"{output_prefix}_energy.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def print_compact_summary(summary_rows):
    """在控制台打印比 CSV 更紧凑的摘要，方便长实验结束后快速确认结果。"""
    print("=" * 118)
    print("Parameter Sweep Summary")
    print("=" * 118)
    print(
        f"{'Config':<8} {'Policy':<20} {'Success':>10} {'Rate':>8} "
        f"{'Perfect%':>9} {'Slots':>8} {'SlotFrac':>9} {'Energy':>10} {'Preview':>9}"
    )
    for row in summary_rows:
        print(
            f"{row['config_id']:<8} {row['policy']:<20} "
            f"{row['success_mean']:>10.2f} {row['success_rate_mean']:>8.3f} "
            f"{row['perfect_rate']:>8.1f}% {row['slots_mean']:>8.2f} "
            f"{row['slot_fraction_mean']:>9.3f} {row['total_energy_mean']:>10.3f} "
            f"{row['decision_preview_calls_per_slot_mean']:>9.2f}"
        )


def main():
    """脚本入口：解析参数、运行扫描、写 CSV、按需生成图像。"""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)
    summary_rows, episode_rows, slot_rows = run_sweep(args)
    print_compact_summary(summary_rows)

    write_csv(
        f"{output_prefix}_summary.csv",
        summary_rows,
        [
            "config_id",
            "sweep",
            "sweep_value",
            "num_nodes",
            "num_slots",
            "num_irs_elements",
            "num_codebook_states",
            "policy",
            "episodes",
            "success_mean",
            "success_rate_mean",
            "success_std",
            "perfect_rate",
            "slots_mean",
            "slot_fraction_mean",
            "avg_power_mean",
            "total_energy_mean",
            "decision_preview_calls_mean",
            "decision_preview_calls_per_slot_mean",
            "tie_candidates_mean",
            "episode_reward_mean",
            "slot1_tx_mean",
            "slot2_total_mean",
        ],
    )
    write_csv(
        f"{output_prefix}_episodes.csv",
        episode_rows,
        [
            "config_id",
            "sweep",
            "sweep_value",
            "num_nodes",
            "num_slots",
            "num_irs_elements",
            "num_codebook_states",
            "policy",
            "episode_idx",
            "episode_seed",
            "success_nodes",
            "success_rate",
            "perfect",
            "missed_nodes",
            "slots_used",
            "slot_fraction",
            "episode_reward",
            "avg_power",
            "total_energy",
            "decision_preview_calls",
            "decision_preview_calls_per_slot",
            "tie_candidates_mean",
        ],
    )
    write_csv(
        f"{output_prefix}_slot_stats.csv",
        slot_rows,
        [
            "config_id",
            "sweep",
            "sweep_value",
            "num_nodes",
            "num_slots",
            "num_irs_elements",
            "num_codebook_states",
            "policy",
            "slot",
            "tx_mean_padded",
            "total_tx_mean_padded",
            "decision_preview_calls_mean_padded",
            "tie_candidates_mean_padded",
        ],
    )

    if not args.no_plots:
        plot_success(summary_rows, output_prefix)
        plot_latency(summary_rows, output_prefix)
        plot_energy(summary_rows, output_prefix)


if __name__ == "__main__":
    main()
