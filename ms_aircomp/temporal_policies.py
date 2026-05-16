"""Temporal stale-CSI policy helpers for execution-mismatch experiments."""

import numpy as np

import evaluate_limited_csi_ms_aircomp as limited
from ms_aircomp.channel_models import apply_channel_state, capture_channel_state
from ms_aircomp.execution_candidates import execution_candidates

__all__ = [
    "choose_temporal_deviation_oracle_decision",
    "choose_temporal_reliability_decision",
    "temporal_reliability_candidate",
    "temporal_reliability_candidate_key",
]


def temporal_reliability_candidate(
    env,
    args,
    candidate,
    decision_error_std,
    temporal_error_std,
    risk_weight=0.5,
    risk_power_weight=0.1,
    quantile_z=1.0,
):
    """Score a stale-CSI candidate by expected current schedulability."""
    adjusted = dict(candidate)
    estimated_valid_mask = np.asarray(candidate["valid_mask"], dtype=bool)
    h_gain = np.asarray(candidate["h_gain"], dtype=float)
    h_abs = np.sqrt(np.maximum(h_gain, 0.0))
    total_error_std = float(np.hypot(float(decision_error_std), float(temporal_error_std)))
    rms = float(np.sqrt(np.mean(h_gain))) if h_gain.size else 0.0
    error_scale = total_error_std * max(rms, 1e-12)
    reliability = limited.estimate_success_reliability(env, args, h_abs, error_scale)

    amp_lower = np.maximum(h_abs - float(quantile_z) * error_scale, 0.0)
    lower_gain = amp_lower**2
    lower_required_power = (float(args.alpha_th) ** 2) / (lower_gain + 1e-12)
    remaining = ~env.transmitted_flags
    quantile_valid_mask = (
        remaining
        & (lower_gain >= float(args.g_th))
        & (lower_required_power <= float(env.P_max))
    )
    quantile_invited_mask = estimated_valid_mask & quantile_valid_mask

    tx_count = int(np.sum(estimated_valid_mask))
    scheduled_power = np.asarray(candidate["required_power"], dtype=float)[estimated_valid_mask]
    power_avg = float(np.mean(scheduled_power)) if tx_count > 0 else 0.0
    expected_success = float(np.sum(reliability[estimated_valid_mask]))
    risk_mass = float(np.sum(1.0 - reliability[estimated_valid_mask]))
    quantile_count = int(np.sum(quantile_invited_mask))
    score = (
        expected_success
        - float(risk_weight) * risk_mass
        - float(risk_power_weight) * power_avg
    )

    adjusted["success_reliability"] = reliability
    adjusted["temporal_reliability_error_std"] = total_error_std
    adjusted["temporal_reliability_error_scale"] = error_scale
    adjusted["temporal_reliability_z"] = float(quantile_z)
    adjusted["temporal_quantile_valid_count"] = quantile_count
    adjusted["expected_success"] = expected_success
    adjusted["risk_mass"] = risk_mass
    adjusted["temporal_reliability_score"] = float(score)
    adjusted["effective_risk_weight"] = float(risk_weight)
    return adjusted


def temporal_reliability_candidate_key(candidate):
    """Rank temporal reliability candidates without hard false-reject filtering."""
    return (
        float(candidate["temporal_reliability_score"]),
        int(candidate["temporal_quantile_valid_count"]),
        float(candidate["expected_success"]),
        int(candidate["tx_this_slot"]),
        -float(candidate["risk_mass"]),
        -float(candidate["power_avg"]) if int(candidate["tx_this_slot"]) > 0 else 0.0,
        float(candidate["mean_gain_remaining"]),
    )


def choose_temporal_reliability_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    temporal_error_std,
    episode_seed,
    risk_weight=0.5,
    risk_power_weight=0.1,
    quantile_z=1.0,
):
    """Choose rotating candidates using temporal schedulability reliability."""
    budget = min(int(budget), args.num_codebook_states)
    random_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        budget,
        salt=17 + slot_idx,
    )
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        budget,
        salt=19 + slot_idx,
    )
    indices = limited.select_indices(
        limited.POLICY_ROTATING_GRID,
        args,
        budget,
        slot_idx,
        random_rng,
    )
    candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=indices,
        error_std=decision_error_std,
        rng=error_rng,
    )
    adjusted = [
        temporal_reliability_candidate(
            env,
            args,
            candidate,
            decision_error_std,
            temporal_error_std,
            risk_weight=risk_weight,
            risk_power_weight=risk_power_weight,
            quantile_z=quantile_z,
        )
        for candidate in candidates
    ]
    return max(adjusted, key=temporal_reliability_candidate_key), len(indices), len(indices)


def choose_temporal_deviation_oracle_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
):
    """
    Choose a B-sized probe set using hidden execution-channel quality.

    This is a diagnostic upper bound for probe-set selection: the oracle may
    pick the current top-B IRS indices, but the actual invitation mask is still
    built from the stale/estimated decision channel.
    """
    budget = min(int(budget), args.num_codebook_states)
    decision_snapshot = capture_channel_state(env)
    if execution_state is not None:
        apply_channel_state(env, execution_state)
    hidden_candidates = execution_candidates(
        env,
        args,
        indices=range(args.num_codebook_states),
        execution_error_std=execution_error_std,
        slot_idx=slot_idx,
    )
    hidden_ranked = sorted(hidden_candidates, key=limited.candidate_key, reverse=True)
    selected_indices = [int(candidate["irs_index"]) for candidate in hidden_ranked[:budget]]

    apply_channel_state(env, decision_snapshot)
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        budget,
        salt=29 + slot_idx,
    )
    decision_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=selected_indices,
        error_std=decision_error_std,
        rng=error_rng,
    )
    decision = limited.best_candidate(decision_candidates)
    decision["deviation_hidden_best_tx"] = int(hidden_ranked[0]["tx_this_slot"]) if hidden_ranked else 0
    decision["deviation_hidden_selected_count"] = len(selected_indices)
    return decision, len(selected_indices), len(selected_indices)
