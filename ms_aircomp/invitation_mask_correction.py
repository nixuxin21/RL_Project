"""实现邀请掩码修正的纯函数核心，包括 target count、rerank mode 和 mask 更新。"""

import numpy as np

__all__ = [
    "apply_invitation_mask_correction",
    "corrected_target_count",
    "rank_remaining_by_stale_gain",
    "MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN",
    "MASK_CORRECTION_MODE_PRUNE_ONLY",
    "VALID_MASK_CORRECTION_RERANK_MODES",
    "validate_mask_correction_rerank_mode",
]

MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN = "global_stale_gain"
MASK_CORRECTION_MODE_PRUNE_ONLY = "prune_only"
VALID_MASK_CORRECTION_RERANK_MODES = (
    MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN,
    MASK_CORRECTION_MODE_PRUNE_ONLY,
)


def validate_mask_correction_rerank_mode(rerank_mode):
    """校验邀请掩码修正的 rerank mode，并把合法模式归一化为内部字符串。"""
    mode = str(rerank_mode).strip()
    if mode not in VALID_MASK_CORRECTION_RERANK_MODES:
        valid = ", ".join(VALID_MASK_CORRECTION_RERANK_MODES)
        raise ValueError(
            f"unknown invitation-mask rerank mode {mode!r}; expected one of: {valid}"
        )
    return mode


def rank_remaining_by_stale_gain(candidate, remaining_mask):
    """对remaining、by、过时、增益进行打分或排序，为候选选择、诊断归因或学习标签提供比较依据。"""
    h_gain = np.asarray(candidate["h_gain"], dtype=float)
    required_power = np.asarray(candidate["required_power"], dtype=float)
    indices = [int(index) for index in np.flatnonzero(remaining_mask)]
    return sorted(
        indices,
        key=lambda index: (
            float(h_gain[index]),
            -float(required_power[index]),
            -int(index),
        ),
        reverse=True,
    )


def corrected_target_count(
    args,
    remaining_count,
    stale_count,
    feedback_count,
    strength,
    deadband_z,
    max_delta,
):
    """处理corrected、target、count相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    raw_delta = float(feedback_count - stale_count)
    noise_count_std = float(args.confirmation_feedback_noise_std) * float(args.num_nodes)
    deadband = max(0.0, float(deadband_z) * noise_count_std)
    if abs(raw_delta) <= deadband:
        corrected_delta = 0.0
    else:
        corrected_delta = np.sign(raw_delta) * (abs(raw_delta) - deadband)
    corrected_delta *= float(strength)
    if float(max_delta) >= 0.0:
        corrected_delta = float(
            np.clip(corrected_delta, -float(max_delta), float(max_delta))
        )
    return int(
        np.clip(
            round(float(stale_count) + corrected_delta),
            0,
            int(remaining_count),
        )
    )


def apply_invitation_mask_correction(
    candidate,
    args,
    env,
    feedback,
    strength,
    deadband_z,
    max_delta,
    rerank_mode=MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN,
):
    """处理apply、invitation、mask、correction相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    rerank_mode = validate_mask_correction_rerank_mode(rerank_mode)
    adjusted = dict(candidate)
    stale_mask = np.asarray(candidate["valid_mask"], dtype=bool).copy()
    remaining_mask = ~env.transmitted_flags
    stale_remaining_mask = stale_mask & remaining_mask
    remaining_count = int(np.sum(remaining_mask))
    stale_count = int(np.sum(stale_remaining_mask))
    feedback_count = int(
        np.clip(
            round(float(feedback["observed_tx_fraction"]) * float(args.num_nodes)),
            0,
            remaining_count,
        )
    )
    requested_target_count = corrected_target_count(
        args,
        remaining_count,
        stale_count,
        feedback_count,
        strength,
        deadband_z,
        max_delta,
    )
    if rerank_mode == MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN:
        if requested_target_count == stale_count:
            corrected_mask = stale_remaining_mask
        else:
            corrected_mask = np.zeros_like(stale_mask, dtype=bool)
            ranked_nodes = rank_remaining_by_stale_gain(candidate, remaining_mask)
            corrected_mask[ranked_nodes[:requested_target_count]] = True
    elif rerank_mode == MASK_CORRECTION_MODE_PRUNE_ONLY:
        if requested_target_count >= stale_count:
            corrected_mask = stale_remaining_mask
        else:
            corrected_mask = np.zeros_like(stale_mask, dtype=bool)
            ranked_stale_nodes = rank_remaining_by_stale_gain(
                candidate,
                stale_remaining_mask,
            )
            corrected_mask[ranked_stale_nodes[:requested_target_count]] = True
    else:
        raise AssertionError(f"unhandled invitation-mask rerank mode: {rerank_mode}")

    target_count = int(np.sum(corrected_mask))
    correction_applied = int(not np.array_equal(corrected_mask, stale_remaining_mask))
    unmet_additions = max(0, int(requested_target_count - target_count))
    if rerank_mode == MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN:
        assert unmet_additions == 0
    added_mask = corrected_mask & (~stale_mask) & remaining_mask
    pruned_mask = stale_mask & (~corrected_mask) & remaining_mask
    scheduled_power = np.asarray(candidate["required_power"], dtype=float)[corrected_mask]
    adjusted["valid_mask"] = corrected_mask
    adjusted["tx_this_slot"] = target_count
    adjusted["power_avg"] = float(np.mean(scheduled_power)) if scheduled_power.size else 0.0
    adjusted["mask_correction_strength"] = float(strength)
    adjusted["mask_correction_noise_deadband_z"] = float(deadband_z)
    adjusted["mask_correction_max_delta"] = float(max_delta)
    adjusted["mask_correction_rerank_mode"] = rerank_mode
    adjusted["mask_correction_stale_count"] = stale_count
    adjusted["mask_correction_feedback_count"] = feedback_count
    adjusted["mask_correction_requested_target_count"] = int(requested_target_count)
    adjusted["mask_correction_target_count"] = target_count
    adjusted["mask_correction_added"] = int(np.sum(added_mask))
    adjusted["mask_correction_pruned"] = int(np.sum(pruned_mask))
    adjusted["mask_correction_requested_delta"] = int(requested_target_count - stale_count)
    adjusted["mask_correction_target_delta"] = int(target_count - stale_count)
    adjusted["mask_correction_unmet_additions"] = unmet_additions
    adjusted["mask_correction_applied"] = correction_applied
    return adjusted
