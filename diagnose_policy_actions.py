"""
策略动作诊断脚本。

主策略对比只能告诉我们“哪个策略表现好”，但不能解释“为什么表现差”。
本脚本专门把 SAC、Codebook-Aware SAC 和 Greedy IRS 的逐时隙动作拆开记录：

- 完整 SAC 学到的 `g_th/alpha_th` 是否偏离稳定固定参数。
- IRS index 是否集中在少数码本。
- 策略选择的 IRS 与局部 Greedy oracle 是否匹配。
- 每个时隙相对 Greedy 少调度了多少节点。

输出包括 episode 级结果、step 级动作明细、summary 统计、slot 曲线和 IRS 索引直方图。
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from test_env import MSAirCompEnv
from train_codebook_aware_agent import FixedTransmissionActionWrapper


# 策略名称集中定义，保证 CSV、图像和控制台表格使用同一套名字。
POLICY_SAC = "SAC"
POLICY_CODEBOOK = "Codebook-Aware SAC"
POLICY_GREEDY = "Greedy IRS"


def parse_args():
    """解析动作诊断实验的环境参数、模型路径和输出前缀。"""
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose SAC, codebook-aware SAC, and greedy IRS actions on shared "
            "MS-AirComp channel seeds."
        )
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--model-dir", default="./rl_models")
    parser.add_argument("--sac-model-name", default="sac_final_model_v3.zip")
    parser.add_argument("--sac-stats-name", default="vec_normalize.pkl")
    parser.add_argument("--codebook-aware-model-name", default="sac_codebook_aware_irs_selector.zip")
    parser.add_argument(
        "--codebook-aware-stats-name",
        default="vec_normalize_codebook_aware_irs_selector.pkl",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Defaults to action_diagnostics_<parameter suffix>.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """校验 episode 数、seed 数和环境规模参数必须为正。"""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if args.num_nodes <= 0:
        raise ValueError("--num-nodes must be positive")
    if args.num_slots <= 0:
        raise ValueError("--num-slots must be positive")
    if args.num_irs_elements <= 0:
        raise ValueError("--num-irs-elements must be positive")
    if args.num_codebook_states <= 0:
        raise ValueError("--num-codebook-states must be positive")


def ensure_parent_dir(path):
    """确保输出文件所在目录存在。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_output_prefix(args):
    """生成动作诊断输出文件的统一前缀。"""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    output_prefix = os.path.join(
        "results",
        "action_diagnostics",
        f"action_diagnostics_ep{args.episodes}_runs{args.num_seeds}_{seed_label}",
    )
    ensure_parent_dir(output_prefix)
    return output_prefix


def physical_to_action(value, low, scale):
    """将真实物理值反向映射到环境动作范围 [-1, 1]。"""
    action_value = (value - low) / scale - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def codebook_index_to_action(index, num_codebook_states):
    """将离散 IRS 码本索引映射到环境第三维动作。"""
    if num_codebook_states <= 1:
        return 0.0
    action_value = 2.0 * index / (num_codebook_states - 1) - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def decode_full_action(action, args):
    """
    解码完整 SAC 的三维动作。

    Returns:
        `(g_th, alpha_th, irs_index)`，用于记录 SAC 在物理层真正执行了什么。
    """
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    g_th = 0.001 + (float(action[0]) + 1.0) * 0.05
    alpha_th = 0.05 + (float(action[1]) + 1.0) * 0.05
    irs_index = int(
        np.clip(
            np.round((float(action[2]) + 1.0) * 0.5 * (args.num_codebook_states - 1)),
            0,
            args.num_codebook_states - 1,
        )
    )
    return g_th, alpha_th, irs_index


def decode_selector_action(action, args):
    """解码 IRS-only selector 的一维动作，得到离散 IRS 码本索引。"""
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    return int(
        np.clip(
            np.round((float(action[0]) + 1.0) * 0.5 * (args.num_codebook_states - 1)),
            0,
            args.num_codebook_states - 1,
        )
    )


def make_base_env(args, irs_phase_mode="codebook", include_codebook_features=False):
    """按诊断参数创建基础环境。"""
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode=irs_phase_mode,
        include_codebook_features=include_codebook_features,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
    )


def make_run_seeds(args):
    """生成外层 run seeds，用于多 seed 诊断。"""
    if args.num_seeds <= 1:
        return [None if args.seed < 0 else args.seed]
    if args.seed < 0:
        rng = np.random.default_rng()
        return [int(seed) for seed in rng.integers(0, 2**31 - 1, size=args.num_seeds)]
    return [args.seed + idx * args.seed_stride for idx in range(args.num_seeds)]


def make_episode_specs(args):
    """
    生成 episode 规格列表。

    每个 spec 记录 run_idx、run_seed、episode_idx 和 episode_seed，
    后续 step/episode CSV 都依赖这些字段做关联。
    """
    specs = []
    for run_idx, run_seed in enumerate(make_run_seeds(args), start=1):
        rng = np.random.default_rng() if run_seed is None else np.random.default_rng(run_seed)
        episode_seeds = [int(seed) for seed in rng.integers(0, 2**31 - 1, size=args.episodes)]
        for episode_idx, episode_seed in enumerate(episode_seeds, start=1):
            specs.append(
                {
                    "run_idx": run_idx,
                    "run_seed": "" if run_seed is None else int(run_seed),
                    "episode_idx": episode_idx,
                    "episode_seed": episode_seed,
                }
            )
    return specs


def local_greedy_candidate(env, g_th, alpha_th, args):
    """
    在当前环境状态下计算局部 Greedy oracle。

    该 oracle 使用和当前被诊断策略相同的 `g_th/alpha_th`，
    只比较 IRS 码本选择。因此它能回答：
    “如果传输参数不变，仅替换成局部最佳 IRS，本时隙最多能多调度多少节点？”
    """
    candidates = [
        env.preview_codebook_index(codebook_index, g_th, alpha_th)
        for codebook_index in range(args.num_codebook_states)
    ]

    def candidate_key(candidate):
        """Greedy 排序键：调度数量优先，其次低功率，最后剩余平均增益。"""
        tx_count = int(candidate["tx_this_slot"])
        power_avg = float(candidate["power_avg"])
        mean_gain = float(candidate["mean_gain_remaining"])
        power_tiebreak = -power_avg if tx_count > 0 else 0.0
        return tx_count, power_tiebreak, mean_gain

    return max(candidates, key=candidate_key)


def print_progress(policy_name, done_count, total_count, recent_success):
    """按 10% 进度输出最近 episode 的平均成功节点数。"""
    interval = max(total_count // 10, 1)
    if done_count % interval == 0 or done_count == total_count:
        print(
            f"  {policy_name}: [{done_count:04d}/{total_count:04d}] "
            f"recent success {np.mean(recent_success[-interval:]):.2f}"
        )


def update_energy(episode_energy, info):
    """累计能耗代理指标：本槽平均功率乘以本槽成功发送节点数。"""
    return episode_energy + float(info["power_avg"]) * int(info["tx_this_slot"])


def append_step_row(
    step_rows,
    spec,
    policy,
    slot,
    g_th,
    alpha_th,
    irs_index,
    info,
    reward,
    oracle,
):
    """
    追加一行 step 级动作诊断数据。

    `oracle` 非空时，会额外记录：
    - 策略 IRS index 是否与 oracle index 相同；
    - 策略本槽调度数与 oracle 本槽调度数之间的 gap。
    """
    oracle_index = int(oracle["irs_index"]) if oracle is not None else ""
    oracle_tx = int(oracle["tx_this_slot"]) if oracle is not None else ""
    match = int(irs_index == oracle_index) if oracle is not None else ""
    gap = int(oracle_tx) - int(info["tx_this_slot"]) if oracle is not None else ""

    step_rows.append(
        {
            "policy": policy,
            "run_idx": spec["run_idx"],
            "run_seed": spec["run_seed"],
            "episode_idx": spec["episode_idx"],
            "episode_seed": spec["episode_seed"],
            "slot": slot,
            "g_th": float(g_th),
            "alpha_th": float(alpha_th),
            "irs_index": int(irs_index),
            "tx_this_slot": int(info["tx_this_slot"]),
            "total_tx": int(info["total_tx"]),
            "power_avg": float(info["power_avg"]),
            "reward": float(reward),
            "oracle_irs_index": oracle_index,
            "oracle_tx_this_slot": oracle_tx,
            "irs_matches_oracle": match,
            "oracle_tx_gap": gap,
            "termination_reason": info.get("termination_reason", ""),
        }
    )


def append_episode_row(
    episode_rows,
    spec,
    policy,
    total_tx,
    slots_used,
    episode_reward,
    episode_power,
    episode_energy,
    args,
):
    """追加一行 episode 级汇总数据。"""
    episode_rows.append(
        {
            "policy": policy,
            "run_idx": spec["run_idx"],
            "run_seed": spec["run_seed"],
            "episode_idx": spec["episode_idx"],
            "episode_seed": spec["episode_seed"],
            "success_nodes": int(total_tx),
            "perfect": int(total_tx == args.num_nodes),
            "missed_nodes": int(args.num_nodes - total_tx),
            "slots_used": int(slots_used),
            "episode_reward": float(episode_reward),
            "avg_power": float(np.mean(episode_power)) if episode_power else 0.0,
            "total_energy": float(episode_energy),
        }
    )


def load_sac(args):
    """
    加载完整 SAC 及其 VecNormalize 环境。

    返回 `raw_venv` 是为了在诊断时访问未包装的底层环境状态，
    从而在同一时隙调用 `preview_codebook_index` 计算局部 Greedy oracle。
    """
    model_path = os.path.join(args.model_dir, args.sac_model_name)
    stats_path = os.path.join(args.model_dir, args.sac_stats_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAC model not found: {model_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"SAC VecNormalize stats not found: {stats_path}")

    raw_venv = DummyVecEnv([lambda: make_base_env(args, "codebook")])
    stacked_venv = VecFrameStack(raw_venv, n_stack=4)
    venv = VecNormalize.load(stats_path, stacked_venv)
    venv.training = False
    venv.norm_reward = False
    model = SAC.load(model_path, env=venv)
    return model, venv, raw_venv


def load_codebook_aware_sac(args):
    """
    加载 Codebook-Aware IRS-only SAC 及其 wrapped 环境。

    这里的底层环境带 codebook features，外层 wrapper 固定 `g_th/alpha_th`。
    """
    model_path = os.path.join(args.model_dir, args.codebook_aware_model_name)
    stats_path = os.path.join(args.model_dir, args.codebook_aware_stats_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Codebook-aware SAC model not found: {model_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Codebook-aware SAC VecNormalize stats not found: {stats_path}")

    def make_wrapped_env():
        """创建与 Codebook-Aware SAC 训练时一致的环境包装。"""
        env = make_base_env(args, "codebook", include_codebook_features=True)
        return FixedTransmissionActionWrapper(env, g_th=args.g_th, alpha_th=args.alpha_th)

    raw_venv = DummyVecEnv([make_wrapped_env])
    stacked_venv = VecFrameStack(raw_venv, n_stack=4)
    venv = VecNormalize.load(stats_path, stacked_venv)
    venv.training = False
    venv.norm_reward = False
    model = SAC.load(model_path, env=venv)
    return model, venv, raw_venv


def diagnose_sac(specs, args, step_rows, episode_rows):
    """
    诊断完整 SAC 的逐时隙动作。

    对每个 step，先解码 SAC 输出的 `g_th/alpha_th/irs_index`，
    再用同样的 `g_th/alpha_th` 计算局部 Greedy oracle，最后执行 SAC 动作并记录 gap。
    """
    model, venv, raw_venv = load_sac(args)
    recent_success = []
    print(f"Running {POLICY_SAC} diagnostics...")
    try:
        for done_count, spec in enumerate(specs, start=1):
            venv.seed(int(spec["episode_seed"]))
            obs = venv.reset()
            base_env = raw_venv.envs[0]
            episode_reward = 0.0
            episode_power = []
            episode_energy = 0.0
            total_tx = 0
            slots_used = args.num_slots

            for slot in range(1, args.num_slots + 1):
                action, _states = model.predict(obs, deterministic=True)
                action_vec = action[0]
                g_th, alpha_th, irs_index = decode_full_action(action_vec, args)
                oracle = local_greedy_candidate(base_env, g_th, alpha_th, args)

                obs, reward, done, info_list = venv.step(action)
                info = info_list[0]
                reward_value = float(reward[0])
                episode_reward += reward_value
                total_tx = int(info["total_tx"])
                slots_used = int(info.get("slots_used", slot))
                episode_energy = update_energy(episode_energy, info)
                if info["tx_this_slot"] > 0:
                    episode_power.append(float(info["power_avg"]))

                append_step_row(
                    step_rows,
                    spec,
                    POLICY_SAC,
                    slot,
                    g_th,
                    alpha_th,
                    irs_index,
                    info,
                    reward_value,
                    oracle,
                )

                if done[0]:
                    break

            append_episode_row(
                episode_rows,
                spec,
                POLICY_SAC,
                total_tx,
                slots_used,
                episode_reward,
                episode_power,
                episode_energy,
                args,
            )
            recent_success.append(total_tx)
            print_progress(POLICY_SAC, done_count, len(specs), recent_success)
    finally:
        venv.close()


def diagnose_codebook_aware_sac(specs, args, step_rows, episode_rows):
    """
    诊断 Codebook-Aware SAC 的 IRS 选择行为。

    因为传输参数固定，本策略与 Greedy 的差距主要来自 IRS index 选择；
    这使 oracle gap 更容易解释。
    """
    model, venv, raw_venv = load_codebook_aware_sac(args)
    recent_success = []
    print(f"Running {POLICY_CODEBOOK} diagnostics...")
    try:
        for done_count, spec in enumerate(specs, start=1):
            venv.seed(int(spec["episode_seed"]))
            obs = venv.reset()
            wrapped_env = raw_venv.envs[0]
            base_env = wrapped_env.env
            episode_reward = 0.0
            episode_power = []
            episode_energy = 0.0
            total_tx = 0
            slots_used = args.num_slots

            for slot in range(1, args.num_slots + 1):
                action, _states = model.predict(obs, deterministic=True)
                irs_index = decode_selector_action(action[0], args)
                oracle = local_greedy_candidate(base_env, args.g_th, args.alpha_th, args)

                obs, reward, done, info_list = venv.step(action)
                info = info_list[0]
                reward_value = float(reward[0])
                episode_reward += reward_value
                total_tx = int(info["total_tx"])
                slots_used = int(info.get("slots_used", slot))
                episode_energy = update_energy(episode_energy, info)
                if info["tx_this_slot"] > 0:
                    episode_power.append(float(info["power_avg"]))

                append_step_row(
                    step_rows,
                    spec,
                    POLICY_CODEBOOK,
                    slot,
                    args.g_th,
                    args.alpha_th,
                    int(info["irs_index"]),
                    info,
                    reward_value,
                    oracle,
                )

                if done[0]:
                    break

            append_episode_row(
                episode_rows,
                spec,
                POLICY_CODEBOOK,
                total_tx,
                slots_used,
                episode_reward,
                episode_power,
                episode_energy,
                args,
            )
            recent_success.append(total_tx)
            print_progress(POLICY_CODEBOOK, done_count, len(specs), recent_success)
    finally:
        venv.close()


def diagnose_greedy_irs(specs, args, step_rows, episode_rows):
    """
    运行 Greedy IRS 并记录与 oracle 的自匹配结果。

    该策略本身就是局部 oracle，因此 match rate 应为 100%，gap 应为 0；
    它作为诊断表中的上界参照。
    """
    env = make_base_env(args, "codebook")
    g_action = physical_to_action(args.g_th, low=0.001, scale=0.05)
    alpha_action = physical_to_action(args.alpha_th, low=0.05, scale=0.05)
    recent_success = []
    print(f"Running {POLICY_GREEDY} diagnostics...")

    for done_count, spec in enumerate(specs, start=1):
        env.reset(seed=int(spec["episode_seed"]))
        episode_reward = 0.0
        episode_power = []
        episode_energy = 0.0
        total_tx = 0
        slots_used = args.num_slots

        for slot in range(1, args.num_slots + 1):
            oracle = local_greedy_candidate(env, args.g_th, args.alpha_th, args)
            irs_index = int(oracle["irs_index"])
            action = np.array(
                [
                    g_action,
                    alpha_action,
                    codebook_index_to_action(irs_index, args.num_codebook_states),
                ],
                dtype=np.float32,
            )
            _obs, reward, terminated, truncated, info = env.step(action)
            reward_value = float(reward)
            episode_reward += reward_value
            total_tx = int(info["total_tx"])
            slots_used = int(info.get("slots_used", slot))
            episode_energy = update_energy(episode_energy, info)
            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            append_step_row(
                step_rows,
                spec,
                POLICY_GREEDY,
                slot,
                args.g_th,
                args.alpha_th,
                irs_index,
                info,
                reward_value,
                oracle,
            )

            if terminated or truncated:
                break

        append_episode_row(
            episode_rows,
            spec,
            POLICY_GREEDY,
            total_tx,
            slots_used,
            episode_reward,
            episode_power,
            episode_energy,
            args,
        )
        recent_success.append(total_tx)
        print_progress(POLICY_GREEDY, done_count, len(specs), recent_success)


def write_csv(path, rows, fieldnames):
    """按固定字段顺序写 CSV 文件。"""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def mean(values):
    """空列表安全均值。"""
    return float(np.mean(values)) if values else 0.0


def std(values):
    """空列表安全标准差。"""
    return float(np.std(values)) if values else 0.0


def quantile(values, q):
    """空列表安全分位数。"""
    return float(np.quantile(values, q)) if values else 0.0


def compute_summary_rows(episode_rows, step_rows, args):
    """
    计算策略级 summary 表。

    输出字段覆盖三类诊断：
    - episode 表现：success、perfect rate、slots、reward、energy；
    - 动作分布：`g_th/alpha_th` 均值、分位数、IRS index 多样性；
    - oracle 对齐：IRS 匹配率和平均调度 gap。
    """
    episodes_by_policy = defaultdict(list)
    steps_by_policy = defaultdict(list)
    for row in episode_rows:
        episodes_by_policy[row["policy"]].append(row)
    for row in step_rows:
        steps_by_policy[row["policy"]].append(row)

    summary_rows = []
    for policy in [POLICY_SAC, POLICY_CODEBOOK, POLICY_GREEDY]:
        episodes = episodes_by_policy[policy]
        steps = steps_by_policy[policy]
        success = [row["success_nodes"] for row in episodes]
        slots = [row["slots_used"] for row in episodes]
        rewards = [row["episode_reward"] for row in episodes]
        powers = [row["avg_power"] for row in episodes]
        energies = [row["total_energy"] for row in episodes]
        g_values = [row["g_th"] for row in steps]
        alpha_values = [row["alpha_th"] for row in steps]
        irs_values = [row["irs_index"] for row in steps]
        match_values = [
            row["irs_matches_oracle"]
            for row in steps
            if row["irs_matches_oracle"] != ""
        ]
        gap_values = [row["oracle_tx_gap"] for row in steps if row["oracle_tx_gap"] != ""]
        dominant_index = ""
        dominant_rate = 0.0
        if irs_values:
            dominant_index, dominant_count = Counter(irs_values).most_common(1)[0]
            dominant_rate = dominant_count / len(irs_values)

        summary_rows.append(
            {
                "policy": policy,
                "episodes": len(episodes),
                "step_count": len(steps),
                "success_mean": mean(success),
                "success_std": std(success),
                "perfect_rate": mean([row["perfect"] for row in episodes]) * 100.0,
                "slots_mean": mean(slots),
                "slots_std": std(slots),
                "episode_reward_mean": mean(rewards),
                "avg_power_mean": mean(powers),
                "total_energy_mean": mean(energies),
                "g_th_mean": mean(g_values),
                "g_th_std": std(g_values),
                "g_th_p05": quantile(g_values, 0.05),
                "g_th_p50": quantile(g_values, 0.50),
                "g_th_p95": quantile(g_values, 0.95),
                "alpha_th_mean": mean(alpha_values),
                "alpha_th_std": std(alpha_values),
                "alpha_th_p05": quantile(alpha_values, 0.05),
                "alpha_th_p50": quantile(alpha_values, 0.50),
                "alpha_th_p95": quantile(alpha_values, 0.95),
                "unique_irs_count": len(set(irs_values)),
                "dominant_irs_index": dominant_index,
                "dominant_irs_rate": dominant_rate,
                "irs_oracle_match_rate": mean(match_values) * 100.0,
                "oracle_tx_gap_mean": mean(gap_values),
            }
        )
    return summary_rows


def compute_slot_stats_rows(episode_rows, step_rows, args):
    """
    计算逐时隙统计表。

    已提前完成的 episode 会在后续时隙补 0 个新增调度节点，
    但累计成功节点保持最终值。这样不同策略的时隙曲线长度一致。
    """
    episodes_by_policy = defaultdict(list)
    steps_by_key = defaultdict(dict)
    for row in episode_rows:
        episodes_by_policy[row["policy"]].append(row)
    for row in step_rows:
        key = (row["policy"], row["run_idx"], row["episode_idx"])
        steps_by_key[key][row["slot"]] = row

    slot_rows = []
    for policy in [POLICY_SAC, POLICY_CODEBOOK, POLICY_GREEDY]:
        episodes = episodes_by_policy[policy]
        for slot in range(1, args.num_slots + 1):
            tx_values = []
            total_values = []
            active_count = 0
            power_values = []
            match_values = []
            gap_values = []

            for episode in episodes:
                key = (policy, episode["run_idx"], episode["episode_idx"])
                step = steps_by_key[key].get(slot)
                if step is None:
                    tx_values.append(0)
                    total_values.append(episode["success_nodes"])
                    continue

                active_count += 1
                tx_values.append(step["tx_this_slot"])
                total_values.append(step["total_tx"])
                power_values.append(step["power_avg"])
                if step["irs_matches_oracle"] != "":
                    match_values.append(step["irs_matches_oracle"])
                if step["oracle_tx_gap"] != "":
                    gap_values.append(step["oracle_tx_gap"])

            slot_rows.append(
                {
                    "policy": policy,
                    "slot": slot,
                    "episode_count": len(episodes),
                    "active_episode_count": active_count,
                    "tx_mean_padded": mean(tx_values),
                    "tx_std_padded": std(tx_values),
                    "total_tx_mean_padded": mean(total_values),
                    "power_mean_active": mean(power_values),
                    "irs_oracle_match_rate_active": mean(match_values) * 100.0,
                    "oracle_tx_gap_mean_active": mean(gap_values),
                }
            )
    return slot_rows


def plot_irs_hist(step_rows, args, output_prefix):
    """绘制不同策略选择 IRS 码本索引的分布直方图。"""
    counts_by_policy = defaultdict(Counter)
    for row in step_rows:
        counts_by_policy[row["policy"]][row["irs_index"]] += 1

    indices = np.arange(args.num_codebook_states)
    policies = [POLICY_SAC, POLICY_CODEBOOK, POLICY_GREEDY]
    width = 0.8 / len(policies)

    fig, ax = plt.subplots(figsize=(12, 5))
    for policy_idx, policy in enumerate(policies):
        counts = np.array([counts_by_policy[policy][idx] for idx in indices], dtype=float)
        total = np.sum(counts)
        rates = counts / total if total > 0 else counts
        offset = (policy_idx - (len(policies) - 1) / 2.0) * width
        ax.bar(indices + offset, rates, width=width, label=policy)

    ax.set_title("IRS Codebook Index Distribution")
    ax.set_xlabel("IRS Codebook Index")
    ax.set_ylabel("Selection Rate")
    ax.set_xticks(indices)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path = f"{output_prefix}_irs_hist.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_latency(episode_rows, args, output_prefix):
    """绘制完成时延分布，即每个策略用多少个时隙结束 episode。"""
    slots_by_policy = defaultdict(list)
    for row in episode_rows:
        slots_by_policy[row["policy"]].append(row["slots_used"])

    policies = [POLICY_SAC, POLICY_CODEBOOK, POLICY_GREEDY]
    x = np.arange(1, args.num_slots + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for policy in policies:
        values = slots_by_policy[policy]
        counts = Counter(values)
        rates = np.array([counts[slot] / len(values) if values else 0.0 for slot in x])
        ax.plot(x, rates, marker="o", linewidth=1.8, label=policy)

    ax.set_title("Completion Latency Distribution")
    ax.set_xlabel("Slots Used")
    ax.set_ylabel("Episode Rate")
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path = f"{output_prefix}_latency.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_slot_curves(slot_rows, args, output_prefix):
    """绘制逐时隙新增调度节点数和累计成功节点数曲线。"""
    rows_by_policy = defaultdict(list)
    for row in slot_rows:
        rows_by_policy[row["policy"]].append(row)

    policies = [POLICY_SAC, POLICY_CODEBOOK, POLICY_GREEDY]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for policy in policies:
        rows = sorted(rows_by_policy[policy], key=lambda row: row["slot"])
        slots = [row["slot"] for row in rows]
        tx_means = [row["tx_mean_padded"] for row in rows]
        total_means = [row["total_tx_mean_padded"] for row in rows]
        axes[0].plot(slots, tx_means, marker="o", linewidth=1.8, label=policy)
        axes[1].plot(slots, total_means, marker="o", linewidth=1.8, label=policy)

    axes[0].set_title("Mean Scheduled Nodes per Slot")
    axes[0].set_xlabel("Slot")
    axes[0].set_ylabel("Scheduled Nodes")
    axes[1].set_title("Mean Cumulative Success Nodes")
    axes[1].set_xlabel("Slot")
    axes[1].set_ylabel("Cumulative Success Nodes")
    axes[1].axhline(args.num_nodes, color="#444444", linestyle="--", linewidth=1.0)

    for ax in axes:
        ax.set_xticks(range(1, args.num_slots + 1))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.tight_layout()
    path = f"{output_prefix}_slot_curves.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(summary_rows):
    """在控制台打印压缩版动作诊断 summary。"""
    print("=" * 118)
    print("Action Diagnostics Summary")
    print("=" * 118)
    print(
        f"{'Policy':<22} {'Success':>9} {'Perfect%':>9} {'Slots':>8} "
        f"{'g_mean':>9} {'a_mean':>9} {'Match%':>9} {'Gap':>7} {'Dominant IRS':>13}"
    )
    for row in summary_rows:
        print(
            f"{row['policy']:<22} {row['success_mean']:>9.3f} "
            f"{row['perfect_rate']:>8.2f}% {row['slots_mean']:>8.3f} "
            f"{row['g_th_mean']:>9.5f} {row['alpha_th_mean']:>9.5f} "
            f"{row['irs_oracle_match_rate']:>8.2f}% {row['oracle_tx_gap_mean']:>7.3f} "
            f"{row['dominant_irs_index']:>13}"
        )


def main():
    """脚本入口：运行三类策略诊断、写出 CSV、按需生成图像。"""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)
    specs = make_episode_specs(args)
    step_rows = []
    episode_rows = []

    print("=" * 96)
    print(
        f"Action diagnostics: episodes={args.episodes}, num_seeds={args.num_seeds}, "
        f"base_seed={args.seed}"
    )
    print(f"Output prefix: {output_prefix}")
    print("=" * 96)

    diagnose_sac(specs, args, step_rows, episode_rows)
    diagnose_codebook_aware_sac(specs, args, step_rows, episode_rows)
    diagnose_greedy_irs(specs, args, step_rows, episode_rows)

    summary_rows = compute_summary_rows(episode_rows, step_rows, args)
    slot_rows = compute_slot_stats_rows(episode_rows, step_rows, args)
    print_summary(summary_rows)

    write_csv(
        f"{output_prefix}_summary.csv",
        summary_rows,
        [
            "policy",
            "episodes",
            "step_count",
            "success_mean",
            "success_std",
            "perfect_rate",
            "slots_mean",
            "slots_std",
            "episode_reward_mean",
            "avg_power_mean",
            "total_energy_mean",
            "g_th_mean",
            "g_th_std",
            "g_th_p05",
            "g_th_p50",
            "g_th_p95",
            "alpha_th_mean",
            "alpha_th_std",
            "alpha_th_p05",
            "alpha_th_p50",
            "alpha_th_p95",
            "unique_irs_count",
            "dominant_irs_index",
            "dominant_irs_rate",
            "irs_oracle_match_rate",
            "oracle_tx_gap_mean",
        ],
    )
    write_csv(
        f"{output_prefix}_slot_stats.csv",
        slot_rows,
        [
            "policy",
            "slot",
            "episode_count",
            "active_episode_count",
            "tx_mean_padded",
            "tx_std_padded",
            "total_tx_mean_padded",
            "power_mean_active",
            "irs_oracle_match_rate_active",
            "oracle_tx_gap_mean_active",
        ],
    )
    write_csv(
        f"{output_prefix}_episodes.csv",
        episode_rows,
        [
            "policy",
            "run_idx",
            "run_seed",
            "episode_idx",
            "episode_seed",
            "success_nodes",
            "perfect",
            "missed_nodes",
            "slots_used",
            "episode_reward",
            "avg_power",
            "total_energy",
        ],
    )
    write_csv(
        f"{output_prefix}_steps.csv",
        step_rows,
        [
            "policy",
            "run_idx",
            "run_seed",
            "episode_idx",
            "episode_seed",
            "slot",
            "g_th",
            "alpha_th",
            "irs_index",
            "tx_this_slot",
            "total_tx",
            "power_avg",
            "reward",
            "oracle_irs_index",
            "oracle_tx_this_slot",
            "irs_matches_oracle",
            "oracle_tx_gap",
            "termination_reason",
        ],
    )

    if not args.no_plots:
        plot_irs_hist(step_rows, args, output_prefix)
        plot_latency(episode_rows, args, output_prefix)
        plot_slot_curves(slot_rows, args, output_prefix)


if __name__ == "__main__":
    main()
