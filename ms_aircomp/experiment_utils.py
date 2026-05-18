"""提供实验脚本共享的小工具，包括路径创建、动作映射、seed 生成、参数校验和能耗累计。"""

import os

import numpy as np

__all__ = [
    "action_to_alpha",
    "codebook_index_to_action",
    "ensure_parent_dir",
    "format_float_for_suffix",
    "make_episode_seed_list",
    "make_episode_seeds",
    "make_run_seeds",
    "physical_to_action",
    "validate_common_experiment_args",
    "validate_nonempty_values",
    "validate_nonnegative_values",
    "validate_positive_values",
    "validate_probe_budget_values",
    "validate_probability_values",
    "update_energy",
]


def ensure_parent_dir(path):
    """处理ensure、parent、dir相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def physical_to_action(value, low, scale):
    """处理physical、to、action相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    action_value = (value - low) / scale - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def codebook_index_to_action(index, num_codebook_states):
    """处理码本、索引、to、action相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if num_codebook_states <= 1:
        return 0.0
    action_value = 2.0 * index / (num_codebook_states - 1) - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def action_to_alpha(action):
    """处理action、to、alpha相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    return float(0.05 + (float(action[1]) + 1.0) * 0.05)


def format_float_for_suffix(value):
    """格式化浮点数、for、suffix显示文本，保证控制台、CSV 和 Markdown 中的数值表达一致。"""
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def make_run_seeds(args):
    """构建运行、随机种子所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    if args.num_seeds <= 1:
        return [None if args.seed < 0 else args.seed]

    if args.seed < 0:
        rng = np.random.default_rng()
        return [int(seed) for seed in rng.integers(0, 2**31 - 1, size=args.num_seeds)]

    return [args.seed + idx * args.seed_stride for idx in range(args.num_seeds)]


def make_episode_seeds(args, run_seed):
    """构建回合、随机种子所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    seed = None if run_seed is None else int(run_seed)
    return make_episode_seed_list(seed, args.episodes)


def make_episode_seed_list(seed, episodes):
    """构建回合、随机种子、列表所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**31 - 1, size=episodes)]


def _option_name(attribute_name):
    """处理option、name相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    return f"--{attribute_name.replace('_', '-')}"


def validate_common_experiment_args(args):
    """校验通用实验参数，包括节点数、时隙数、码本规模、seed 数和 seed 步长。"""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if hasattr(args, "seed_stride") and args.seed_stride <= 0:
        raise ValueError("--seed-stride must be positive")

    for name in ("num_nodes", "num_slots", "num_irs_elements", "num_codebook_states"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{_option_name(name)} must be positive")
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")
    if args.g_th <= 0.0:
        raise ValueError("--g-th must be positive")
    if args.alpha_th <= 0.0:
        raise ValueError("--alpha-th must be positive")


def validate_nonempty_values(values, option_name):
    """校验解析后的列表不为空，避免空参数导致实验矩阵没有有效取值。"""
    values = list(values)
    if not values:
        raise ValueError(f"{option_name} must not be empty")
    return values


def _validate_finite_values(values, option_name):
    """校验数值列表中的每一项都是有限数，避免 NaN 或无穷值进入实验。"""
    values = validate_nonempty_values(values, option_name)
    if any(not np.isfinite(float(value)) for value in values):
        raise ValueError(f"{option_name} must contain finite values")
    return values


def validate_nonnegative_values(values, option_name):
    """校验数值列表中的每一项都是非负有限数。"""
    values = _validate_finite_values(values, option_name)
    if any(float(value) < 0.0 for value in values):
        raise ValueError(f"{option_name} must be non-negative")
    return values


def validate_positive_values(values, option_name):
    """校验数值列表中的每一项都是正的有限数。"""
    values = _validate_finite_values(values, option_name)
    if any(float(value) <= 0.0 for value in values):
        raise ValueError(f"{option_name} must be positive")
    return values


def validate_probability_values(values, option_name):
    """校验概率列表中的每一项都位于 [0, 1] 区间。"""
    values = _validate_finite_values(values, option_name)
    if any(float(value) < 0.0 or float(value) > 1.0 for value in values):
        raise ValueError(f"{option_name} must be in [0, 1]")
    return values


def validate_probe_budget_values(values, num_codebook_states, option_name):
    """校验 probe budget 列表，确保预算非负且不会超过码本大小。"""
    budgets = [int(value) for value in validate_nonempty_values(values, option_name)]
    if any(value <= 0 for value in budgets):
        raise ValueError(f"{option_name} must contain positive integers")
    if any(value > num_codebook_states for value in budgets):
        raise ValueError(f"{option_name} must not exceed --num-codebook-states")
    return sorted(set(budgets))


def update_energy(episode_energy, info):
    """更新energy相关状态、历史记录或结果行，保证后续时隙和聚合阶段能继续累积信息。"""
    return episode_energy + float(info["power_avg"]) * int(info["tx_this_slot"])
