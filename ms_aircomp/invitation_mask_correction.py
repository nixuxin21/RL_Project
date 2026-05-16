"""Invitation-mask correction helpers for aggregate current feedback."""

import numpy as np

__all__ = [
    "apply_invitation_mask_correction",
    "corrected_target_count",
    "rank_remaining_by_stale_gain",
]


def rank_remaining_by_stale_gain(candidate, remaining_mask):
    """Rank remaining nodes from strongest to weakest stale gain."""
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


def corrected_target_count(args, remaining_count, stale_count, feedback_count, strength, deadband_z, max_delta):
    """Return noise-aware target count for invitation-mask correction."""
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
):
    """
    Correct a confirmed candidate's stale invitation mask with aggregate feedback.

    `strength=0` leaves the original stale mask unchanged. Direct correction
    uses `deadband_z=0` and `max_delta=-1`; noise-aware variants shrink the
    count delta by a feedback-noise deadband and optionally clip the correction.
    """
    adjusted = dict(candidate)
    stale_mask = np.asarray(candidate["valid_mask"], dtype=bool).copy()
    remaining_mask = ~env.transmitted_flags
    remaining_count = int(np.sum(remaining_mask))
    stale_count = int(np.sum(stale_mask & remaining_mask))
    feedback_count = int(
        np.clip(
            round(float(feedback["observed_tx_fraction"]) * float(args.num_nodes)),
            0,
            remaining_count,
        )
    )
    target_count = corrected_target_count(
        args,
        remaining_count,
        stale_count,
        feedback_count,
        strength,
        deadband_z,
        max_delta,
    )
    if target_count == stale_count:
        corrected_mask = stale_mask & remaining_mask
    else:
        corrected_mask = np.zeros_like(stale_mask, dtype=bool)
        ranked_nodes = rank_remaining_by_stale_gain(candidate, remaining_mask)
        corrected_mask[ranked_nodes[:target_count]] = True

    added_mask = corrected_mask & (~stale_mask) & remaining_mask
    pruned_mask = stale_mask & (~corrected_mask) & remaining_mask
    scheduled_power = np.asarray(candidate["required_power"], dtype=float)[corrected_mask]
    adjusted["valid_mask"] = corrected_mask
    adjusted["tx_this_slot"] = int(np.sum(corrected_mask))
    adjusted["power_avg"] = float(np.mean(scheduled_power)) if scheduled_power.size else 0.0
    adjusted["mask_correction_strength"] = float(strength)
    adjusted["mask_correction_noise_deadband_z"] = float(deadband_z)
    adjusted["mask_correction_max_delta"] = float(max_delta)
    adjusted["mask_correction_stale_count"] = stale_count
    adjusted["mask_correction_feedback_count"] = feedback_count
    adjusted["mask_correction_target_count"] = target_count
    adjusted["mask_correction_added"] = int(np.sum(added_mask))
    adjusted["mask_correction_pruned"] = int(np.sum(pruned_mask))
    adjusted["mask_correction_target_delta"] = int(target_count - stale_count)
    adjusted["mask_correction_applied"] = int(target_count != stale_count)
    return adjusted
