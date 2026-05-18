"""
共享随机种子的多策略对比脚本。

这是当前项目最重要的主评估脚本之一，用于在完全相同的 episode seeds / 信道样本下，
比较学习型策略和 rule-based baseline：

- SAC：完整三维动作 `[g_th, alpha_th, irs]`。
- SAC Fixed g/a：固定传输参数，只保留 SAC 学到的 IRS 选择。
- Codebook-Aware SAC：单独训练的 IRS-only selector。
- Feature Argmax IRS：直接选择 codebook quality feature 最大的 IRS 码本。
- Feature Argmax PowerTie IRS：只在最大 feature 并列候选中额外比较功率。
- Greedy IRS：每时隙 preview 全部码本并按调度数/功率排序。
- No IRS / Random IRS：主线基础 baseline。
- Random Fixed IRS / validation-selected Best Fixed IRS / Fixed IRS：显式开启的静态 IRS 消融。

脚本的核心设计是“共享随机性”：每个策略在同一批 episode seeds 上评估，
从而让差异主要来自策略本身，而不是信道随机采样。
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from ms_aircomp.experiment_utils import (
    action_to_alpha,
    codebook_index_to_action,
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seed_list,
    make_episode_seeds,
    make_run_seeds,
    physical_to_action,
    update_energy,
)
from test_env import MSAirCompEnv
from train_codebook_aware_agent import FixedTransmissionActionWrapper


# 多 seed 聚合时需要拼接/求均值的数值字段集中放在这里，减少后续漏字段风险。
NUMERIC_RESULT_KEYS = (
    "success_nodes",
    "avg_mse",
    "avg_power",
    "episode_reward",
    "slots_used",
    "total_energy",
    "decision_preview_calls_per_slot",
    "tie_candidates_mean",
)

BEST_FIXED_VALIDATION_SEED_OFFSET = 0x5BF15ED


def codebook_feature_preview_cost(args, include_codebook_features=True):
    """处理码本、特征、预览、cost相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    return float(args.num_codebook_states) if include_codebook_features else 0.0


def make_best_fixed_validation_seeds(args, run_seed):
    """构建best、固定、validation、随机种子所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    if run_seed is None or int(run_seed) < 0:
        return make_episode_seed_list(None, args.episodes)
    return make_episode_seed_list(int(run_seed) + BEST_FIXED_VALIDATION_SEED_OFFSET, args.episodes)


def parse_args():
    """
    解析主策略对比的命令行参数。

    默认只跑完整 SAC 和 rule-based baseline；Codebook-Aware SAC 及其 ablation
    通过显式 flag 打开，避免本地缺少对应模型文件时影响普通 baseline 评估。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare no-IRS, random IRS, feature-argmax IRS, feature-argmax "
            "power-tie IRS, greedy IRS, and optional learning/static baselines "
            "on shared channel seeds."
        )
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=1, help="Number of independent base seeds.")
    parser.add_argument("--seed-stride", type=int, default=1000, help="Offset between deterministic base seeds.")
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument(
        "--codebook-feature-noise-std",
        type=float,
        default=0.0,
        help=(
            "Gaussian noise std added to normalized codebook quality features. "
            "Default 0 keeps the exact-feature baseline."
        ),
    )
    parser.add_argument("--fixed-irs-index", type=int, default=7)
    parser.add_argument(
        "--include-fixed-irs-baselines",
        action="store_true",
        help=(
            "Also evaluate static IRS ablations: Random Fixed IRS, Best Fixed IRS, "
            "and Fixed IRS. These are excluded from the default main-baseline table."
        ),
    )
    parser.add_argument("--model-dir", default="./rl_models")
    parser.add_argument("--model-name", default="sac_final_model_v3.zip")
    parser.add_argument("--stats-name", default="vec_normalize.pkl")
    parser.add_argument("--include-codebook-aware-sac", action="store_true")
    parser.add_argument(
        "--include-codebook-aware-ablation",
        action="store_true",
        help="Also evaluate the codebook-aware SAC model with normalized codebook features zeroed.",
    )
    parser.add_argument(
        "--include-irs-selector-no-features",
        action="store_true",
        help="Evaluate an IRS-only SAC selector trained without codebook quality features.",
    )
    parser.add_argument("--codebook-aware-model-name", default="sac_codebook_aware_irs_selector.zip")
    parser.add_argument(
        "--codebook-aware-stats-name",
        default="vec_normalize_codebook_aware_irs_selector.pkl",
    )
    parser.add_argument("--irs-selector-no-features-model-name", default="sac_irs_selector_no_codebook_features.zip")
    parser.add_argument(
        "--irs-selector-no-features-stats-name",
        default="vec_normalize_irs_selector_no_codebook_features.pkl",
    )
    parser.add_argument("--skip-sac", action="store_true", help="Only run non-learning baselines.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the saved comparison plot. Defaults to a parameterized filename.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Path for the saved summary CSV. Defaults to a parameterized filename.",
    )
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
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")
    if args.codebook_feature_noise_std < 0:
        raise ValueError("--codebook-feature-noise-std must be non-negative")


def make_output_suffix(args):
    """根据关键实验参数生成默认输出文件名后缀。"""
    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    parts = [f"ep{args.episodes}", f"runs{args.num_seeds}", seed_label, "featargmax", "powertie"]
    if args.codebook_feature_noise_std > 0:
        parts.append(f"featnoise{format_float_for_suffix(args.codebook_feature_noise_std)}")
    if args.skip_sac:
        parts.append("nosac")
    if args.include_codebook_aware_sac:
        parts.append("cbsac")
    if args.include_codebook_aware_ablation:
        parts.append("zerofeat")
    if args.include_irs_selector_no_features:
        parts.append("nofeat")
    if args.include_fixed_irs_baselines:
        parts.append("staticfixed")
    return "_".join(parts)


def resolve_output_paths(args):
    """如果用户未指定输出路径，就使用参数化文件名自动命名。"""
    suffix = make_output_suffix(args)
    if args.output is None:
        args.output = os.path.join("results", "policy_comparison", f"policy_comparison_results_{suffix}.png")
    if args.csv_output is None:
        args.csv_output = os.path.join("results", "policy_comparison", f"policy_comparison_summary_{suffix}.csv")
    ensure_parent_dir(args.output)
    ensure_parent_dir(args.csv_output)


def make_env(args, irs_phase_mode):
    """创建不带 codebook features 的基础评估环境。"""
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode=irs_phase_mode,
    )


def make_feature_env(args):
    """创建带 C 维 codebook quality features 的评估环境。"""
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode="codebook",
        include_codebook_features=True,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
        codebook_feature_noise_std=args.codebook_feature_noise_std,
    )


def make_random_fixed_index_rng(episode_seed):
    """
    为 Random Fixed IRS 生成独立 RNG。

    这里不直接复用环境 seed，而是做一个确定性偏移，避免“随机固定 IRS 索引”
    和“环境信道随机数”使用同一随机流。
    """
    if episode_seed is None:
        return np.random.default_rng()
    return np.random.default_rng((int(episode_seed) + 0x9E3779B1) % (2**32))


def append_common_episode_metrics(
    avg_mse,
    avg_power,
    rewards,
    slots_used,
    total_energy,
    episode_mse,
    episode_power,
    episode_reward,
    episode_slots,
    episode_energy,
):
    """
    把一个 episode 的通用指标追加到策略级列表。

    多个评估函数都需要记录 MSE、平均功率、奖励、时延和能耗，
    统一在这里追加可以保持所有策略的统计口径一致。
    """
    avg_mse.append(float(np.mean(episode_mse)))
    avg_power.append(float(np.mean(episode_power)) if episode_power else 0.0)
    rewards.append(float(episode_reward))
    slots_used.append(int(episode_slots))
    total_energy.append(float(episode_energy))


def finalize_result(
    name,
    success_nodes,
    avg_mse,
    avg_power,
    rewards,
    tx_per_slot,
    slots_used,
    total_energy,
    **extra,
):
    """
    将策略评估产生的 Python 列表整理为统一结果字典。

    `extra` 用于附带策略特有信息，例如 IRS 索引序列或最佳固定 IRS index。
    """
    episode_count = len(success_nodes)
    decision_preview_calls_per_slot = extra.pop(
        "decision_preview_calls_per_slot",
        np.zeros(episode_count, dtype=float),
    )
    tie_candidates_mean = extra.pop(
        "tie_candidates_mean",
        np.zeros(episode_count, dtype=float),
    )
    result = {
        "name": name,
        "success_nodes": np.array(success_nodes),
        "avg_mse": np.array(avg_mse),
        "avg_power": np.array(avg_power),
        "episode_reward": np.array(rewards),
        "tx_per_slot": tx_per_slot,
        "slots_used": np.array(slots_used),
        "total_energy": np.array(total_energy),
        "decision_preview_calls_per_slot": np.array(decision_preview_calls_per_slot, dtype=float),
        "tie_candidates_mean": np.array(tie_candidates_mean, dtype=float),
    }
    result.update(extra)
    return result


def evaluate_fixed_action_policy(name, irs_phase_mode, action, episode_seeds, args, show_progress=True):
    """
    评估固定动作策略。

    适用对象：
    - Fixed IRS：固定 `g_th/alpha_th/irs_index`。
    - Random IRS：固定 `g_th/alpha_th`，IRS 相位由环境随机生成。
    - No IRS：固定 `g_th/alpha_th`，环境关闭 IRS 级联链路。
    """
    env = make_env(args, irs_phase_mode)
    success_nodes = []
    avg_mse = []
    avg_power = []
    rewards = []
    tx_per_slot = []
    slots_used = []
    total_energy = []

    if show_progress:
        print(f"Running {name}...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        episode_mse = []
        episode_power = []
        episode_reward = 0.0
        episode_tx = []
        episode_slots = args.num_slots
        episode_energy = 0.0
        total_tx = 0

        for _slot in range(args.num_slots):
            _obs, reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            episode_tx.append(int(info["tx_this_slot"]))
            episode_reward += float(reward)
            episode_slots = int(info.get("slots_used", len(episode_tx)))
            episode_energy = update_energy(episode_energy, info)

            alpha_th = action_to_alpha(action)
            episode_mse.append(env.noise_var / (alpha_th**2))
            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            if terminated or truncated:
                break

        success_nodes.append(total_tx)
        append_common_episode_metrics(
            avg_mse,
            avg_power,
            rewards,
            slots_used,
            total_energy,
            episode_mse,
            episode_power,
            episode_reward,
            episode_slots,
            episode_energy,
        )
        tx_per_slot.append(episode_tx)

        if show_progress:
            print_progress(name, ep, args.episodes, success_nodes, args.num_nodes)

    return finalize_result(
        name,
        success_nodes,
        avg_mse,
        avg_power,
        rewards,
        tx_per_slot,
        slots_used,
        total_energy,
    )


def evaluate_sac_policy(episode_seeds, args, fixed_action_prefix=None, name="SAC"):
    """
    评估完整 SAC 模型。

    Args:
        fixed_action_prefix: 若不为 None，则强制覆盖 SAC 输出的前两维
            `[g_th, alpha_th]`，只保留 SAC 的 IRS 选择。这就是 `SAC Fixed g/a` baseline。
        name: 输出表格中的策略名称。
    """
    model_path = os.path.join(args.model_dir, args.model_name)
    stats_path = os.path.join(args.model_dir, args.stats_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAC model not found: {model_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"VecNormalize stats not found: {stats_path}")

    # 评估管线必须复刻训练时的包装顺序：先构造向量环境，再做帧堆叠，最后加载归一化统计量。
    raw_venv = DummyVecEnv([lambda: make_env(args, "codebook")])
    stacked_venv = VecFrameStack(raw_venv, n_stack=4)
    venv = VecNormalize.load(stats_path, stacked_venv)
    venv.training = False
    venv.norm_reward = False
    model = SAC.load(model_path, env=venv)

    success_nodes = []
    avg_mse = []
    avg_power = []
    rewards = []
    tx_per_slot = []
    slots_used = []
    total_energy = []

    print(f"Running {name}...")
    try:
        for ep, episode_seed in enumerate(episode_seeds, start=1):
            venv.seed(episode_seed)
            obs = venv.reset()
            episode_mse = []
            episode_power = []
            episode_reward = 0.0
            episode_tx = []
            episode_slots = args.num_slots
            episode_energy = 0.0
            total_tx = 0

            for _slot in range(args.num_slots):
                action, _states = model.predict(obs, deterministic=True)
                if fixed_action_prefix is not None:
                    # 固定传输参数的 SAC 对照：只测试 SAC 学到的 IRS 维度，排除传输参数偏移的影响。
                    action = action.copy()
                    action[0, 0] = fixed_action_prefix[0]
                    action[0, 1] = fixed_action_prefix[1]
                alpha_th = action_to_alpha(action[0])
                obs, reward, done, info_list = venv.step(action)
                info = info_list[0]

                total_tx = int(info["total_tx"])
                episode_tx.append(int(info["tx_this_slot"]))
                episode_reward += float(reward[0])
                episode_slots = int(info.get("slots_used", len(episode_tx)))
                episode_energy = update_energy(episode_energy, info)
                episode_mse.append(1e-9 / (alpha_th**2))
                if info["tx_this_slot"] > 0:
                    episode_power.append(float(info["power_avg"]))

                if done[0]:
                    break

            success_nodes.append(total_tx)
            append_common_episode_metrics(
                avg_mse,
                avg_power,
                rewards,
                slots_used,
                total_energy,
                episode_mse,
                episode_power,
                episode_reward,
                episode_slots,
                episode_energy,
            )
            tx_per_slot.append(episode_tx)

            print_progress(name, ep, args.episodes, success_nodes, args.num_nodes)
    finally:
        venv.close()

    return finalize_result(
        name,
        success_nodes,
        avg_mse,
        avg_power,
        rewards,
        tx_per_slot,
        slots_used,
        total_energy,
    )


def zero_codebook_features(obs, args):
    """
    将堆叠观测中的 codebook quality features 全部置零。

    Codebook-Aware SAC 使用 4 帧堆叠，观测结构是多个 `[7 + C]` frame 拼接。
    因此需要逐帧定位第 7 维之后的 C 个 feature，而不是只处理最后一帧。
    """
    frame_dim = 7 + args.num_codebook_states
    if obs.shape[-1] % frame_dim != 0:
        raise ValueError(
            f"Unexpected observation shape {obs.shape}; cannot locate codebook feature frames."
        )

    ablated = obs.copy()
    num_frames = obs.shape[-1] // frame_dim
    for frame_idx in range(num_frames):
        start = frame_idx * frame_dim + 7
        stop = start + args.num_codebook_states
        ablated[..., start:stop] = 0.0
    return ablated


def evaluate_codebook_aware_sac_policy(
    episode_seeds,
    args,
    ablate_codebook_features=False,
    name="Codebook-Aware SAC",
    model_name=None,
    stats_name=None,
    include_codebook_features=True,
):
    """
    评估 IRS-only SAC selector。

    这个策略通过 `FixedTransmissionActionWrapper` 固定 `g_th/alpha_th`，
    模型只输出一维 IRS 动作。可选的 `ablate_codebook_features=True`
    会在模型预测前把 codebook features 置零，用于检测模型是否真正依赖这些特征。
    """
    model_path = os.path.join(args.model_dir, model_name or args.codebook_aware_model_name)
    stats_path = os.path.join(args.model_dir, stats_name or args.codebook_aware_stats_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{name} model not found: {model_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"{name} VecNormalize stats not found: {stats_path}")

    def make_wrapped_env():
        """创建与 IRS-only selector 训练时一致的 wrapped env。"""
        env = MSAirCompEnv(
            num_nodes=args.num_nodes,
            num_slots=args.num_slots,
            num_irs_elements=args.num_irs_elements,
            num_codebook_states=args.num_codebook_states,
            irs_phase_mode="codebook",
            include_codebook_features=include_codebook_features,
            codebook_feature_g_th=args.g_th,
            codebook_feature_alpha_th=args.alpha_th,
            codebook_feature_noise_std=(
                args.codebook_feature_noise_std if include_codebook_features else 0.0
            ),
        )
        return FixedTransmissionActionWrapper(env, g_th=args.g_th, alpha_th=args.alpha_th)

    raw_venv = DummyVecEnv([make_wrapped_env])
    stacked_venv = VecFrameStack(raw_venv, n_stack=4)
    venv = VecNormalize.load(stats_path, stacked_venv)
    venv.training = False
    venv.norm_reward = False
    model = SAC.load(model_path, env=venv)

    success_nodes = []
    avg_mse = []
    avg_power = []
    rewards = []
    tx_per_slot = []
    slots_used = []
    total_energy = []
    decision_preview_calls_per_slot = []
    tie_candidates_mean = []

    print(f"Running {name}...")
    try:
        for ep, episode_seed in enumerate(episode_seeds, start=1):
            venv.seed(episode_seed)
            obs = venv.reset()
            episode_mse = []
            episode_power = []
            episode_reward = 0.0
            episode_tx = []
            episode_slots = args.num_slots
            episode_energy = 0.0
            episode_preview_calls = 0.0
            total_tx = 0

            for _slot in range(args.num_slots):
                episode_preview_calls += codebook_feature_preview_cost(args, include_codebook_features)
                policy_obs = (
                    zero_codebook_features(obs, args)
                    if ablate_codebook_features and include_codebook_features
                    else obs
                )
                action, _states = model.predict(policy_obs, deterministic=True)
                obs, reward, done, info_list = venv.step(action)
                info = info_list[0]

                total_tx = int(info["total_tx"])
                episode_tx.append(int(info["tx_this_slot"]))
                episode_reward += float(reward[0])
                episode_slots = int(info.get("slots_used", len(episode_tx)))
                episode_energy = update_energy(episode_energy, info)
                episode_mse.append(1e-9 / (args.alpha_th**2))
                if info["tx_this_slot"] > 0:
                    episode_power.append(float(info["power_avg"]))

                if done[0]:
                    break

            success_nodes.append(total_tx)
            append_common_episode_metrics(
                avg_mse,
                avg_power,
                rewards,
                slots_used,
                total_energy,
                episode_mse,
                episode_power,
                episode_reward,
                episode_slots,
                episode_energy,
            )
            tx_per_slot.append(episode_tx)
            active_slots = max(len(episode_tx), 1)
            decision_preview_calls_per_slot.append(float(episode_preview_calls) / active_slots)
            tie_candidates_mean.append(0.0)

            print_progress(name, ep, args.episodes, success_nodes, args.num_nodes)
    finally:
        venv.close()

    return finalize_result(
        name,
        success_nodes,
        avg_mse,
        avg_power,
        rewards,
        tx_per_slot,
        slots_used,
        total_energy,
        decision_preview_calls_per_slot=decision_preview_calls_per_slot,
        tie_candidates_mean=tie_candidates_mean,
    )


def evaluate_greedy_irs_policy(episode_seeds, args, base_action):
    """
    评估局部 Greedy IRS baseline。

    每个时隙都 preview 全部 C 个码本，然后选择：
    1. 可调度节点数最多；
    2. 若并列，平均功率最低；
    3. 若仍并列，剩余节点平均信道增益最高。

    因为它完整预览所有码本，所以它更接近 oracle-style baseline，
    但复杂度也高于 Feature Argmax。
    """
    env = make_env(args, "codebook")
    success_nodes = []
    avg_mse = []
    avg_power = []
    rewards = []
    tx_per_slot = []
    slots_used = []
    total_energy = []
    selected_indices = []
    decision_preview_calls_per_slot = []
    tie_candidates_mean = []
    g_th = args.g_th
    alpha_th = args.alpha_th

    print("Running Greedy IRS...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        episode_mse = []
        episode_power = []
        episode_reward = 0.0
        episode_tx = []
        episode_indices = []
        episode_slots = args.num_slots
        episode_energy = 0.0
        total_tx = 0

        for _slot in range(args.num_slots):
            candidates = [
                env.preview_codebook_index(codebook_index, g_th, alpha_th)
                for codebook_index in range(args.num_codebook_states)
            ]

            def candidate_key(candidate):
                """贪心排序键：先比较调度数量，再用低功率和剩余增益打破并列。"""
                tx_count = int(candidate["tx_this_slot"])
                power_avg = float(candidate["power_avg"])
                mean_gain = float(candidate["mean_gain_remaining"])
                power_tiebreak = -power_avg if tx_count > 0 else 0.0
                return tx_count, power_tiebreak, mean_gain

            best_candidate = max(candidates, key=candidate_key)
            greedy_action = base_action.copy()
            greedy_action[2] = codebook_index_to_action(
                best_candidate["irs_index"], args.num_codebook_states
            )

            _obs, reward, terminated, truncated, info = env.step(greedy_action)
            total_tx = int(info["total_tx"])
            episode_tx.append(int(info["tx_this_slot"]))
            episode_indices.append(int(info["irs_index"]))
            episode_reward += float(reward)
            episode_slots = int(info.get("slots_used", len(episode_tx)))
            episode_energy = update_energy(episode_energy, info)

            episode_mse.append(env.noise_var / (alpha_th**2))
            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            if terminated or truncated:
                break

        success_nodes.append(total_tx)
        append_common_episode_metrics(
            avg_mse,
            avg_power,
            rewards,
            slots_used,
            total_energy,
            episode_mse,
            episode_power,
            episode_reward,
            episode_slots,
            episode_energy,
        )
        tx_per_slot.append(episode_tx)
        selected_indices.append(episode_indices)
        decision_preview_calls_per_slot.append(float(args.num_codebook_states))
        tie_candidates_mean.append(float(args.num_codebook_states))

        print_progress("Greedy IRS", ep, args.episodes, success_nodes, args.num_nodes)

    return finalize_result(
        "Greedy IRS",
        success_nodes,
        avg_mse,
        avg_power,
        rewards,
        tx_per_slot,
        slots_used,
        total_energy,
        selected_indices=selected_indices,
        decision_preview_calls_per_slot=decision_preview_calls_per_slot,
        tie_candidates_mean=tie_candidates_mean,
    )


def feature_argmax_candidates(obs, args):
    """
    返回 codebook features 中所有并列最大候选索引。

    feature 的数值是预计可调度节点比例，所以并列最大值代表这些码本
    在当前时隙拥有相同的最大预计调度节点数。
    """
    features = obs[7 : 7 + args.num_codebook_states]
    max_feature = float(np.max(features))
    return np.flatnonzero(np.isclose(features, max_feature, rtol=0.0, atol=1e-7))


def feature_argmax_power_tie_index(env, args, obs):
    """
    在 Feature Argmax 的最大调度候选集合内执行功率 tie-break。

    Returns:
        `(irs_index, extra_preview_calls, tie_count)`。这里的 preview_calls 只统计
        在 codebook features 已经可用之后，决策阶段额外调用 preview 的次数。
    """
    candidate_indices = feature_argmax_candidates(obs, args)
    tie_count = int(len(candidate_indices))
    if tie_count <= 1:
        return int(candidate_indices[0]), 0, tie_count

    candidates = [
        env.preview_codebook_index(codebook_index, args.g_th, args.alpha_th)
        for codebook_index in candidate_indices
    ]

    def candidate_key(candidate):
        """功率并列排序键：先比较调度数量，再用低功率和剩余增益打破并列。"""
        tx_count = int(candidate["tx_this_slot"])
        power_avg = float(candidate["power_avg"])
        mean_gain = float(candidate["mean_gain_remaining"])
        power_tiebreak = -power_avg if tx_count > 0 else 0.0
        return tx_count, power_tiebreak, mean_gain

    best_candidate = max(candidates, key=candidate_key)
    return int(best_candidate["irs_index"]), tie_count, tie_count


def evaluate_feature_argmax_irs_policy(episode_seeds, args, base_action):
    """
    评估 Feature Argmax IRS baseline。

    环境观测尾部的 C 维 codebook features 表示每个码本预计可调度节点比例。
    本策略直接选择最大 feature 对应的码本，不做神经网络推理，也不额外 preview 功率。
    C 维 codebook feature acquisition 本身按每 slot C 次 preview 计入成本。
    """
    env = make_feature_env(args)
    success_nodes = []
    avg_mse = []
    avg_power = []
    rewards = []
    tx_per_slot = []
    slots_used = []
    total_energy = []
    selected_indices = []
    decision_preview_calls_per_slot = []
    tie_candidates_mean = []

    print("Running Feature Argmax IRS...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        obs, _reset_info = env.reset(seed=episode_seed)
        episode_mse = []
        episode_power = []
        episode_reward = 0.0
        episode_tx = []
        episode_indices = []
        episode_tie_candidates = []
        episode_slots = args.num_slots
        episode_energy = 0.0
        total_tx = 0

        for _slot in range(args.num_slots):
            candidate_indices = feature_argmax_candidates(obs, args)
            irs_index = int(candidate_indices[0])
            episode_tie_candidates.append(int(len(candidate_indices)))
            action = base_action.copy()
            action[2] = codebook_index_to_action(irs_index, args.num_codebook_states)

            obs, reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            episode_tx.append(int(info["tx_this_slot"]))
            episode_indices.append(int(info["irs_index"]))
            episode_reward += float(reward)
            episode_slots = int(info.get("slots_used", len(episode_tx)))
            episode_energy = update_energy(episode_energy, info)

            episode_mse.append(env.noise_var / (args.alpha_th**2))
            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            if terminated or truncated:
                break

        success_nodes.append(total_tx)
        append_common_episode_metrics(
            avg_mse,
            avg_power,
            rewards,
            slots_used,
            total_energy,
            episode_mse,
            episode_power,
            episode_reward,
            episode_slots,
            episode_energy,
        )
        tx_per_slot.append(episode_tx)
        selected_indices.append(episode_indices)
        decision_preview_calls_per_slot.append(codebook_feature_preview_cost(args))
        tie_candidates_mean.append(float(np.mean(episode_tie_candidates)) if episode_tie_candidates else 0.0)

        print_progress("Feature Argmax IRS", ep, args.episodes, success_nodes, args.num_nodes)

    return finalize_result(
        "Feature Argmax IRS",
        success_nodes,
        avg_mse,
        avg_power,
        rewards,
        tx_per_slot,
        slots_used,
        total_energy,
        selected_indices=selected_indices,
        decision_preview_calls_per_slot=decision_preview_calls_per_slot,
        tie_candidates_mean=tie_candidates_mean,
    )


def evaluate_feature_argmax_power_tie_irs_policy(episode_seeds, args, base_action):
    """
    评估 Feature Argmax PowerTie IRS baseline。

    该策略先用观测中的 codebook features 找到最大预计调度数量的候选集合，
    只在这些并列候选里额外 preview 平均功率，并选择 Greedy tie-break 下最优的码本。
    总 preview 成本计为 codebook feature acquisition 的 C 次，加上 tie-break preview。
    """
    env = make_feature_env(args)
    success_nodes = []
    avg_mse = []
    avg_power = []
    rewards = []
    tx_per_slot = []
    slots_used = []
    total_energy = []
    selected_indices = []
    decision_preview_calls_per_slot = []
    tie_candidates_mean = []

    print("Running Feature Argmax PowerTie IRS...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        obs, _reset_info = env.reset(seed=episode_seed)
        episode_mse = []
        episode_power = []
        episode_reward = 0.0
        episode_tx = []
        episode_indices = []
        episode_slots = args.num_slots
        episode_energy = 0.0
        episode_preview_calls = 0
        episode_tie_candidates = []
        total_tx = 0

        for _slot in range(args.num_slots):
            irs_index, preview_calls, tie_count = feature_argmax_power_tie_index(env, args, obs)
            episode_preview_calls += codebook_feature_preview_cost(args) + int(preview_calls)
            episode_tie_candidates.append(int(tie_count))
            action = base_action.copy()
            action[2] = codebook_index_to_action(irs_index, args.num_codebook_states)

            obs, reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            episode_tx.append(int(info["tx_this_slot"]))
            episode_indices.append(int(info["irs_index"]))
            episode_reward += float(reward)
            episode_slots = int(info.get("slots_used", len(episode_tx)))
            episode_energy = update_energy(episode_energy, info)

            episode_mse.append(env.noise_var / (args.alpha_th**2))
            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            if terminated or truncated:
                break

        success_nodes.append(total_tx)
        append_common_episode_metrics(
            avg_mse,
            avg_power,
            rewards,
            slots_used,
            total_energy,
            episode_mse,
            episode_power,
            episode_reward,
            episode_slots,
            episode_energy,
        )
        tx_per_slot.append(episode_tx)
        selected_indices.append(episode_indices)
        active_slots = max(len(episode_tie_candidates), 1)
        decision_preview_calls_per_slot.append(float(episode_preview_calls) / active_slots)
        tie_candidates_mean.append(float(np.mean(episode_tie_candidates)) if episode_tie_candidates else 0.0)

        print_progress("Feature Argmax PowerTie IRS", ep, args.episodes, success_nodes, args.num_nodes)

    return finalize_result(
        "Feature Argmax PowerTie IRS",
        success_nodes,
        avg_mse,
        avg_power,
        rewards,
        tx_per_slot,
        slots_used,
        total_energy,
        selected_indices=selected_indices,
        decision_preview_calls_per_slot=decision_preview_calls_per_slot,
        tie_candidates_mean=tie_candidates_mean,
    )


def evaluate_episode_random_fixed_irs_policy(episode_seeds, args, base_action):
    """
    每个 episode 随机选择一个 DFT 码本索引，并在该 episode 的所有时隙保持固定。

    它和 `Random IRS` 不同：Random IRS 是每个时隙随机相位；
    Random Fixed IRS 更像“随机选一个固定波束”的弱 baseline。
    """
    env = make_env(args, "codebook")
    success_nodes = []
    avg_mse = []
    avg_power = []
    rewards = []
    tx_per_slot = []
    slots_used = []
    total_energy = []
    selected_indices = []

    print("Running Random Fixed IRS...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        index_rng = make_random_fixed_index_rng(episode_seed)
        irs_index = int(index_rng.integers(0, args.num_codebook_states))
        action = base_action.copy()
        action[2] = codebook_index_to_action(irs_index, args.num_codebook_states)

        episode_mse = []
        episode_power = []
        episode_reward = 0.0
        episode_tx = []
        episode_slots = args.num_slots
        episode_energy = 0.0
        total_tx = 0

        for _slot in range(args.num_slots):
            _obs, reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            episode_tx.append(int(info["tx_this_slot"]))
            episode_reward += float(reward)
            episode_slots = int(info.get("slots_used", len(episode_tx)))
            episode_energy = update_energy(episode_energy, info)

            episode_mse.append(env.noise_var / (args.alpha_th**2))
            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            if terminated or truncated:
                break

        success_nodes.append(total_tx)
        append_common_episode_metrics(
            avg_mse,
            avg_power,
            rewards,
            slots_used,
            total_energy,
            episode_mse,
            episode_power,
            episode_reward,
            episode_slots,
            episode_energy,
        )
        tx_per_slot.append(episode_tx)
        selected_indices.append(irs_index)

        print_progress("Random Fixed IRS", ep, args.episodes, success_nodes, args.num_nodes)

    return finalize_result(
        "Random Fixed IRS",
        success_nodes,
        avg_mse,
        avg_power,
        rewards,
        tx_per_slot,
        slots_used,
        total_energy,
        selected_indices=selected_indices,
    )


def select_best_fixed_irs_index(validation_episode_seeds, args, base_action):
    """按照best、固定、IRS、索引规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    print("Running Best Fixed IRS validation search...")
    candidate_results = []
    for irs_index in range(args.num_codebook_states):
        action = base_action.copy()
        action[2] = codebook_index_to_action(irs_index, args.num_codebook_states)
        result = evaluate_fixed_action_policy(
            f"Fixed IRS {irs_index} validation",
            "codebook",
            action,
            validation_episode_seeds,
            args,
            show_progress=False,
        )
        result["fixed_irs_index"] = irs_index
        candidate_results.append(result)

    def candidate_key(result):
        """最佳固定 IRS 排序键：覆盖率优先，其次比较完成时延和总能耗。"""
        success = float(np.mean(result["success_nodes"]))
        perfect_rate = float(np.mean(result["success_nodes"] == args.num_nodes))
        latency = float(np.mean(result["slots_used"]))
        energy = float(np.mean(result["total_energy"]))
        return success, perfect_rate, -latency, -energy

    best_result = max(candidate_results, key=candidate_key)
    return int(best_result["fixed_irs_index"]), best_result


def evaluate_best_fixed_irs_policy(episode_seeds, args, base_action, validation_episode_seeds=None):
    """评估最佳固定 IRS 策略，返回后续聚合和报告生成所需的指标。"""
    if validation_episode_seeds is None:
        validation_episode_seeds = make_best_fixed_validation_seeds(args, args.seed)
    best_index, validation_result = select_best_fixed_irs_index(
        validation_episode_seeds,
        args,
        base_action,
    )
    best_action = base_action.copy()
    best_action[2] = codebook_index_to_action(best_index, args.num_codebook_states)
    best_result = evaluate_fixed_action_policy(
        f"Best Fixed IRS {best_index} (val-selected)",
        "codebook",
        best_action,
        episode_seeds,
        args,
        show_progress=False,
    )
    best_result["fixed_irs_index"] = best_index
    best_result["best_fixed_validation_episodes"] = len(validation_episode_seeds)
    print(
        f"  Best Fixed IRS validation-selected: index={best_index}, "
        f"validation mean success={np.mean(validation_result['success_nodes']):.2f}/{args.num_nodes}, "
        f"test mean success={np.mean(best_result['success_nodes']):.2f}/{args.num_nodes}"
    )
    return best_result


def print_progress(name, ep, episodes, success_nodes, num_nodes):
    """按 10% 间隔打印最近一段 episode 的平均成功节点数。"""
    interval = max(episodes // 10, 1)
    if ep % interval == 0 or ep == episodes:
        recent = np.mean(success_nodes[-interval:])
        print(f"  {name}: [{ep:04d}/{episodes:04d}] recent success {recent:.2f}/{num_nodes}")


def seed_summary(result):
    """把一个 run 内的策略结果压缩为 seed-level 均值。"""
    return {key: float(np.mean(result[key])) for key in NUMERIC_RESULT_KEYS}


def aggregate_seed_results(seed_result_sets):
    """
    合并多个 run seed 的评估结果。

    episode 级数组会直接拼接；同时保留每个 seed 的 summary，
    用于后续计算跨 seed 的 95% 置信区间。
    """
    if not seed_result_sets:
        return []

    aggregated_results = []
    result_count = len(seed_result_sets[0])
    for result_idx in range(result_count):
        parts = [seed_results[result_idx] for seed_results in seed_result_sets]
        aggregated = {"name": parts[0]["name"], "tx_per_slot": []}

        for key in NUMERIC_RESULT_KEYS:
            aggregated[key] = np.concatenate([part[key] for part in parts])

        for part in parts:
            aggregated["tx_per_slot"].extend(part["tx_per_slot"])

        if any("selected_indices" in part for part in parts):
            selected_indices = []
            for part in parts:
                selected_indices.extend(part.get("selected_indices", []))
            aggregated["selected_indices"] = selected_indices

        if "fixed_irs_index" in parts[0]:
            aggregated["fixed_irs_index"] = parts[0]["fixed_irs_index"]

        aggregated["seed_summaries"] = [seed_summary(part) for part in parts]
        aggregated_results.append(aggregated)

    return aggregated_results


def metric_mean_ci(result, key):
    """
    计算总体均值和基于 run seed 的 95% CI。

    如果只有一个 seed，则 CI 记为 0，避免错误地用 episode 方差冒充 seed 不确定性。
    """
    seed_values = np.array(
        [summary[key] for summary in result.get("seed_summaries", [seed_summary(result)])],
        dtype=float,
    )
    mean_value = float(np.mean(result[key]))
    if len(seed_values) <= 1:
        return mean_value, 0.0
    ci95 = 1.96 * float(np.std(seed_values, ddof=1)) / np.sqrt(len(seed_values))
    return mean_value, ci95


def run_policy_suite(
    episode_seeds,
    args,
    fixed_action,
    random_action,
    g_action,
    alpha_action,
    include_best_fixed=True,
    best_fixed_validation_seeds=None,
):
    """
    在同一批 episode seeds 上执行一整套策略。

    这是共享随机评估的核心：所有策略都使用同一批信道样本，
    使策略之间的差异具有可比性。
    """
    results = []
    if not args.skip_sac:
        results.append(evaluate_sac_policy(episode_seeds, args, name="SAC"))
        results.append(
            evaluate_sac_policy(
                episode_seeds,
                args,
                fixed_action_prefix=np.array([g_action, alpha_action], dtype=np.float32),
                name="SAC Fixed g/a",
            )
        )

    if args.include_codebook_aware_sac:
        results.append(evaluate_codebook_aware_sac_policy(episode_seeds, args))

    if args.include_codebook_aware_ablation:
        results.append(
            evaluate_codebook_aware_sac_policy(
                episode_seeds,
                args,
                ablate_codebook_features=True,
                name="Codebook-Aware SAC ZeroFeat",
            )
        )

    if args.include_irs_selector_no_features:
        results.append(
            evaluate_codebook_aware_sac_policy(
                episode_seeds,
                args,
                name="IRS Selector NoFeat",
                model_name=args.irs_selector_no_features_model_name,
                stats_name=args.irs_selector_no_features_stats_name,
                include_codebook_features=False,
            )
        )

    results.append(evaluate_fixed_action_policy("No IRS", "none", random_action, episode_seeds, args))
    results.append(evaluate_fixed_action_policy("Random IRS", "random", random_action, episode_seeds, args))
    results.append(evaluate_feature_argmax_irs_policy(episode_seeds, args, fixed_action))
    results.append(evaluate_feature_argmax_power_tie_irs_policy(episode_seeds, args, fixed_action))
    results.append(evaluate_greedy_irs_policy(episode_seeds, args, fixed_action))

    if args.include_fixed_irs_baselines:
        results.append(evaluate_episode_random_fixed_irs_policy(episode_seeds, args, fixed_action))
        if include_best_fixed:
            results.append(
                evaluate_best_fixed_irs_policy(
                    episode_seeds,
                    args,
                    fixed_action,
                    validation_episode_seeds=best_fixed_validation_seeds,
                )
            )
        results.append(
            evaluate_fixed_action_policy(
                f"Fixed IRS {args.fixed_irs_index}",
                "codebook",
                fixed_action,
                episode_seeds,
                args,
            )
        )
    return results


def evaluate_global_best_fixed_irs_policy(
    episode_seed_sets,
    args,
    base_action,
    validation_episode_seed_sets=None,
):
    """
    多 seed 场景下搜索全局 Best Fixed IRS。

    The fixed index is selected on held-out validation seeds and then evaluated
    on the report/test seeds for each run.
    """
    if validation_episode_seed_sets is None:
        validation_episode_seed_sets = [
            make_best_fixed_validation_seeds(args, args.seed)
        ]
    validation_episode_seeds = [
        episode_seed
        for seeds in validation_episode_seed_sets
        for episode_seed in seeds
    ]
    best_index, validation_result = select_best_fixed_irs_index(
        validation_episode_seeds,
        args,
        base_action,
    )
    best_action = base_action.copy()
    best_action[2] = codebook_index_to_action(best_index, args.num_codebook_states)

    seed_result_sets = []
    for episode_seeds in episode_seed_sets:
        seed_result_sets.append(
            [
                evaluate_fixed_action_policy(
                    f"Best Fixed IRS {best_index} (val-selected)",
                    "codebook",
                    best_action,
                    episode_seeds,
                    args,
                    show_progress=False,
                )
            ]
    )

    aggregated = aggregate_seed_results(seed_result_sets)[0]
    aggregated["fixed_irs_index"] = best_index
    aggregated["best_fixed_validation_episodes"] = len(validation_episode_seeds)
    print(
        f"  Global Best Fixed IRS validation-selected: index={best_index}, "
        f"validation mean success={np.mean(validation_result['success_nodes']):.2f}/{args.num_nodes}"
    )
    return aggregated


def summarize(results, args):
    """
    打印并保存最终策略对比表。

    CSV 中包含成功节点、置信区间、完美覆盖率、时延、MSE、功率、总能耗和奖励。
    这些字段对应论文表格中最常用的性能指标。
    """
    rows = []
    print("=" * 132)
    print("Policy Comparison Summary")
    print("=" * 132)
    print(
        f"{'Policy':<30} {'Success':>12} {'CI95':>8} {'Std':>8} "
        f"{'Perfect %':>10} {'Slots':>8} {'Slot CI':>8} {'Energy':>12} "
        f"{'Power W':>12} {'Preview':>9} {'Tie':>7} {'Reward':>10}"
    )

    for result in results:
        success = result["success_nodes"]
        success_mean, success_ci95 = metric_mean_ci(result, "success_nodes")
        slots_mean, slots_ci95 = metric_mean_ci(result, "slots_used")
        energy_mean, energy_ci95 = metric_mean_ci(result, "total_energy")
        row = {
            "policy": result["name"],
            "codebook_feature_noise_std": args.codebook_feature_noise_std,
            "success_mean": success_mean,
            "success_ci95": success_ci95,
            "success_std": float(np.std(success)),
            "perfect_rate": float(np.mean(success == args.num_nodes) * 100.0),
            "avg_mse": float(np.mean(result["avg_mse"])),
            "avg_power": float(np.mean(result["avg_power"])),
            "slots_mean": slots_mean,
            "slots_ci95": slots_ci95,
            "total_energy_mean": energy_mean,
            "total_energy_ci95": energy_ci95,
            "decision_preview_calls_per_slot_mean": float(np.mean(result["decision_preview_calls_per_slot"])),
            "tie_candidates_mean": float(np.mean(result["tie_candidates_mean"])),
            "avg_reward": float(np.mean(result["episode_reward"])),
        }
        rows.append(row)
        print(
            f"{row['policy']:<30} {row['success_mean']:>12.2f} {row['success_ci95']:>8.2f} "
            f"{row['success_std']:>8.2f} {row['perfect_rate']:>9.1f}% "
            f"{row['slots_mean']:>8.2f} {row['slots_ci95']:>8.2f} "
            f"{row['total_energy_mean']:>12.4e} {row['avg_power']:>12.4e} "
            f"{row['decision_preview_calls_per_slot_mean']:>9.2f} {row['tie_candidates_mean']:>7.2f} "
            f"{row['avg_reward']:>10.2f}"
        )

    with open(args.csv_output, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "policy",
            "codebook_feature_noise_std",
            "success_mean",
            "success_ci95",
            "success_std",
            "perfect_rate",
            "slots_mean",
            "slots_ci95",
            "avg_mse",
            "avg_power",
            "total_energy_mean",
            "total_energy_ci95",
            "decision_preview_calls_per_slot_mean",
            "tie_candidates_mean",
            "avg_reward",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV saved to: {args.csv_output}")


def plot_results(results, args):
    """
    绘制四宫格策略对比图。

    左上：平均成功节点数；右上：成功节点数 CDF；
    左下：完成时延；右下：总发射能耗代理。
    """
    names = [result["name"] for result in results]
    success_means = [np.mean(result["success_nodes"]) for result in results]
    success_ci95 = [metric_mean_ci(result, "success_nodes")[1] for result in results]
    latency_means = [np.mean(result["slots_used"]) for result in results]
    latency_ci95 = [metric_mean_ci(result, "slots_used")[1] for result in results]
    energies = [np.mean(result["total_energy"]) for result in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(idx % 20) for idx in range(len(results))]

    axes[0, 0].bar(names, success_means, yerr=success_ci95, color=colors, capsize=4)
    axes[0, 0].axhline(args.num_nodes, color="#444444", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Average Success Nodes")
    axes[0, 0].set_ylabel("Success Nodes")
    axes[0, 0].set_ylim(0, args.num_nodes + 2)
    axes[0, 0].grid(True, axis="y", linestyle="--", alpha=0.4)

    for result, color in zip(results, colors):
        sorted_success = np.sort(result["success_nodes"])
        yvals = np.arange(1, len(sorted_success) + 1) / len(sorted_success)
        axes[0, 1].plot(sorted_success, yvals, label=result["name"], color=color, linewidth=1.6)
    axes[0, 1].set_title("Success Nodes CDF")
    axes[0, 1].set_xlabel("Success Nodes")
    axes[0, 1].set_ylabel("Cumulative Probability")
    axes[0, 1].set_xlim(0, args.num_nodes + 1)
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend()

    axes[1, 0].bar(names, latency_means, yerr=latency_ci95, color=colors, capsize=4)
    axes[1, 0].set_title("Completion Latency")
    axes[1, 0].set_ylabel("Slots Used")
    axes[1, 0].set_ylim(0, args.num_slots + 1)
    axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[1, 1].bar(names, energies, color=colors)
    axes[1, 1].set_title("Total Transmit Energy Proxy")
    axes[1, 1].set_ylabel("Sum Power Across Slots")
    axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.4)

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Comparison plot saved to: {args.output}")


def main():
    """
    脚本入口。

    负责准备固定动作、生成 run/episode seeds、逐 seed 运行策略套件、
    聚合结果，并最终写 CSV 与图像。
    """
    args = parse_args()
    validate_args(args)
    resolve_output_paths(args)
    fixed_index = int(np.clip(args.fixed_irs_index, 0, args.num_codebook_states - 1))
    args.fixed_irs_index = fixed_index

    g_action = physical_to_action(args.g_th, low=0.001, scale=0.05)
    alpha_action = physical_to_action(args.alpha_th, low=0.05, scale=0.05)
    fixed_irs_action = codebook_index_to_action(fixed_index, args.num_codebook_states)
    fixed_action = np.array([g_action, alpha_action, fixed_irs_action], dtype=np.float32)
    random_action = np.array([g_action, alpha_action, 0.0], dtype=np.float32)
    run_seeds = make_run_seeds(args)

    print("=" * 96)
    print(
        f"Shared-seed policy comparison: episodes={args.episodes}, "
        f"num_seeds={len(run_seeds)}, base_seed={args.seed}"
    )
    print(f"Transmission parameters: g_th={args.g_th}, alpha_th={args.alpha_th}")
    if args.include_fixed_irs_baselines:
        print(f"Static IRS ablation enabled: fixed IRS index={fixed_index}")
    print(f"Codebook feature noise std: {args.codebook_feature_noise_std}")
    print("=" * 96)

    seed_result_sets = []
    episode_seed_sets = []
    validation_episode_seed_sets = []
    include_best_fixed_per_run = args.include_fixed_irs_baselines and len(run_seeds) == 1
    for run_idx, run_seed in enumerate(run_seeds, start=1):
        print("=" * 96)
        print(f"Run seed [{run_idx}/{len(run_seeds)}]: {run_seed}")
        print("=" * 96)
        episode_seeds = make_episode_seeds(args, run_seed)
        best_fixed_validation_seeds = (
            make_best_fixed_validation_seeds(args, run_seed)
            if args.include_fixed_irs_baselines
            else None
        )
        episode_seed_sets.append(episode_seeds)
        if best_fixed_validation_seeds is not None:
            validation_episode_seed_sets.append(best_fixed_validation_seeds)
        seed_result_sets.append(
            run_policy_suite(
                episode_seeds,
                args,
                fixed_action,
                random_action,
                g_action,
                alpha_action,
                include_best_fixed=include_best_fixed_per_run,
                best_fixed_validation_seeds=best_fixed_validation_seeds,
            )
        )

    results = aggregate_seed_results(seed_result_sets)
    if args.include_fixed_irs_baselines and not include_best_fixed_per_run:
        best_fixed_result = evaluate_global_best_fixed_irs_policy(
            episode_seed_sets,
            args,
            fixed_action,
            validation_episode_seed_sets=validation_episode_seed_sets,
        )
        insert_at = next(
            (idx + 1 for idx, result in enumerate(results) if result["name"] == "Random Fixed IRS"),
            len(results),
        )
        results.insert(insert_at, best_fixed_result)
    summarize(results, args)
    plot_results(results, args)


if __name__ == "__main__":
    main()
