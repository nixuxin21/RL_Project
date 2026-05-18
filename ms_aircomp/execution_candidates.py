"""构造执行阶段候选和隐藏 current-channel oracle，用于比较决策信道与执行信道的偏差。"""

import numpy as np

import ms_aircomp.limited_csi as limited
from ms_aircomp.channel_models import drift_channels, execution_rng

__all__ = [
    "choose_execution_oracle",
    "execution_candidate_for_decision",
    "execution_candidates",
    "execution_oracle_candidate",
]


def execution_candidates(env, args, indices=None, execution_error_std=0.0, slot_idx=0, no_irs=False):
    """处理执行阶段、候选集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    episode_seed = getattr(env, "_last_seed", None)
    if no_irs:
        rng = execution_rng(
            episode_seed,
            execution_error_std,
            slot_idx,
            no_irs=True,
            candidate_index=-2,
        )
        h_ref = limited.effective_channels(env, no_irs=True)
        h_exec = drift_channels(h_ref, execution_error_std, rng)
        return [limited.build_candidate(env, args, -2, h_exec[0], no_irs=True)]

    clean_indices = [int(np.clip(index, 0, args.num_codebook_states - 1)) for index in indices]
    h_ref = limited.effective_channels(env, clean_indices)
    candidates = []
    for row_idx, index in enumerate(clean_indices):
        rng = execution_rng(
            episode_seed,
            execution_error_std,
            slot_idx,
            candidate_index=index,
        )
        h_exec = drift_channels(h_ref[row_idx : row_idx + 1], execution_error_std, rng)[0]
        candidates.append(limited.build_candidate(env, args, index, h_exec))
    return candidates


def execution_candidate_for_decision(env, args, decision_candidate, execution_error_std, slot_idx):
    """处理执行阶段、候选、for、决策相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    irs_index = int(decision_candidate["irs_index"])
    if irs_index == -2:
        return execution_candidates(
            env,
            args,
            execution_error_std=execution_error_std,
            slot_idx=slot_idx,
            no_irs=True,
        )[0]
    return execution_candidates(
        env,
        args,
        indices=[irs_index],
        execution_error_std=execution_error_std,
        slot_idx=slot_idx,
    )[0]


def execution_oracle_candidate(env, args, execution_error_std, slot_idx):
    """处理执行阶段、oracle 诊断上界、候选相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    candidates = execution_candidates(
        env,
        args,
        indices=range(args.num_codebook_states),
        execution_error_std=execution_error_std,
        slot_idx=slot_idx,
    )
    return limited.best_candidate(candidates)


def choose_execution_oracle(env, args, execution_error_std, slot_idx):
    """按照执行阶段、oracle 诊断上界规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    oracle = execution_oracle_candidate(env, args, execution_error_std, slot_idx)
    return oracle, args.num_codebook_states, args.num_codebook_states
