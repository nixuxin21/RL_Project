"""Execution-risk and opportunity-cost policy helpers."""

import numpy as np

import ms_aircomp.limited_csi as limited

POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID = "Adaptive Execution-Risk Rotating Grid"

__all__ = [
    "POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID",
    "candidate_with_execution_reliability",
    "choose_execution_risk_decision",
    "choose_opportunity_execution_risk_decision",
    "execution_risk_candidate_set",
    "execution_risk_error_std",
    "execution_urgency",
    "opportunity_cost_candidate",
    "opportunity_cost_candidate_key",
]


def execution_risk_error_std(decision_error_std, execution_error_std):
    """Combine independent decision-estimation and execution-drift uncertainty."""
    return float(np.hypot(float(decision_error_std), float(execution_error_std)))


def candidate_with_execution_reliability(
    env,
    args,
    candidate,
    decision_error_std,
    execution_error_std,
):
    """
    Re-score a decision candidate with execution-drift reliability.

    Limited-CSI risk-aware candidates normally derive reliability only from
    noisy decision CSI. In execution-mismatch experiments, a stale decision can
    be exact while the execution slot still drifts. This helper exposes that
    drift as a posterior reliability term without revealing the realized
    execution channel.
    """
    adjusted = dict(candidate)
    h_gain = np.asarray(candidate["h_gain"], dtype=float)
    h_proxy = np.sqrt(np.maximum(h_gain, 0.0))
    combined_std = execution_risk_error_std(decision_error_std, execution_error_std)
    rms = float(np.sqrt(np.mean(h_gain))) if h_gain.size else 0.0
    error_scale = combined_std * max(rms, 1e-12)
    adjusted["success_reliability"] = limited.estimate_success_reliability(
        env,
        args,
        h_proxy,
        error_scale,
    )
    adjusted["execution_risk_error_std"] = combined_std
    adjusted["execution_risk_error_scale"] = error_scale
    return adjusted


def execution_risk_candidate_set(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
):
    """Return rotating-grid candidates re-scored with execution-drift reliability."""
    budget = min(int(budget), args.num_codebook_states)
    random_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        budget,
        salt=2 + slot_idx,
    )
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        budget,
        salt=3 + slot_idx,
    )
    indices = limited.select_indices(
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
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
    candidates = [
        candidate_with_execution_reliability(
            env,
            args,
            candidate,
            decision_error_std,
            execution_error_std,
        )
        for candidate in candidates
    ]
    return candidates, len(indices)


def execution_urgency(env, args, slot_idx, deadline_gain=0.5, backlog_gain=0.5):
    """Return a multiplier that raises missed-opportunity cost near deadline/backlog."""
    if args.num_slots <= 1:
        deadline_pressure = 1.0
    else:
        deadline_pressure = float(slot_idx) / float(max(args.num_slots - 1, 1))
    remaining_count = int(np.sum(~env.transmitted_flags))
    remaining_ratio = float(remaining_count) / float(max(args.num_nodes, 1))
    slots_left = max(int(args.num_slots) - int(slot_idx), 1)
    schedule_ratio = float(slots_left) / float(max(args.num_slots, 1))
    backlog_pressure = max(0.0, remaining_ratio - schedule_ratio) / max(schedule_ratio, 1e-12)
    backlog_pressure = min(backlog_pressure, 2.0)
    return 1.0 + float(deadline_gain) * deadline_pressure + float(backlog_gain) * backlog_pressure


def opportunity_cost_candidate(
    env,
    candidate,
    args,
    slot_idx,
    failure_cost=1.0,
    missed_cost=1.0,
    power_weight=0.1,
    deadline_gain=0.5,
    backlog_gain=0.5,
):
    """Build an execution-risk candidate using false-accept and false-reject costs."""
    adjusted = dict(candidate)
    estimated_valid_mask = np.asarray(candidate["valid_mask"], dtype=bool)
    reliability = np.asarray(candidate["success_reliability"], dtype=float)
    required_power = np.asarray(candidate["required_power"], dtype=float)
    urgency = execution_urgency(
        env,
        args,
        slot_idx,
        deadline_gain=deadline_gain,
        backlog_gain=backlog_gain,
    )
    success_value = urgency
    false_reject_cost = float(missed_cost) * urgency
    false_accept_cost = float(failure_cost)
    normalized_power = required_power / max(float(env.P_max), 1e-12)
    invite_utility = (
        reliability * success_value
        - (1.0 - reliability) * false_accept_cost
        - float(power_weight) * normalized_power
    )
    skip_utility = -false_reject_cost * reliability
    invite_mask = estimated_valid_mask & (invite_utility >= skip_utility)

    scheduled_power = required_power[invite_mask]
    tx_count = int(np.sum(invite_mask))
    power_avg = float(np.mean(scheduled_power)) if tx_count > 0 else 0.0
    expected_success = float(np.sum(reliability[invite_mask]))
    failure_mass = float(np.sum(1.0 - reliability[invite_mask]))
    rejected_mask = estimated_valid_mask & (~invite_mask)
    missed_mass = float(np.sum(reliability[rejected_mask]))
    policy_value = float(
        np.sum(invite_utility[invite_mask])
        + np.sum(skip_utility[rejected_mask])
    )

    adjusted["estimated_valid_mask"] = estimated_valid_mask
    adjusted["valid_mask"] = invite_mask
    adjusted["tx_this_slot"] = tx_count
    adjusted["power_avg"] = power_avg
    adjusted["expected_success"] = expected_success
    adjusted["risk_mass"] = failure_mass
    adjusted["missed_reliability_mass"] = missed_mass
    adjusted["opportunity_score"] = policy_value
    adjusted["opportunity_urgency"] = urgency
    adjusted["opportunity_false_accept_cost"] = false_accept_cost
    adjusted["opportunity_false_reject_cost"] = false_reject_cost
    adjusted["effective_risk_weight"] = false_accept_cost
    adjusted["risk_rejected_count"] = int(np.sum(rejected_mask))
    return adjusted


def opportunity_cost_candidate_key(candidate):
    """Rank opportunity-cost candidates by expected utility and reliable progress."""
    return (
        float(candidate["opportunity_score"]),
        float(candidate["expected_success"]),
        int(candidate["tx_this_slot"]),
        -float(candidate["risk_mass"]),
        -float(candidate["power_avg"]) if int(candidate["tx_this_slot"]) > 0 else 0.0,
        float(candidate["mean_gain_remaining"]),
    )


def choose_opportunity_execution_risk_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    failure_cost=1.0,
    missed_cost=1.0,
    power_weight=0.1,
    deadline_gain=0.5,
    backlog_gain=0.5,
):
    """Choose rotating-grid candidates by expected utility under execution drift."""
    candidates, preview_calls = execution_risk_candidate_set(
        env,
        args,
        budget,
        slot_idx,
        decision_error_std,
        execution_error_std,
        episode_seed,
    )
    adjusted = [
        opportunity_cost_candidate(
            env,
            candidate,
            args,
            slot_idx,
            failure_cost=failure_cost,
            missed_cost=missed_cost,
            power_weight=power_weight,
            deadline_gain=deadline_gain,
            backlog_gain=backlog_gain,
        )
        for candidate in candidates
    ]
    return max(adjusted, key=opportunity_cost_candidate_key), preview_calls, preview_calls


def choose_execution_risk_decision(
    env,
    args,
    policy_name,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    risk_weight=0.5,
    risk_power_weight=0.1,
    risk_invite_threshold=0.5,
    adaptive_risk_error_ref=0.3,
    adaptive_risk_error_gain=1.0,
    adaptive_risk_deadline_relief=0.6,
    adaptive_risk_backlog_relief=0.8,
):
    """Choose rotating-grid candidates using execution-drift risk statistics."""
    candidates, preview_calls = execution_risk_candidate_set(
        env,
        args,
        budget,
        slot_idx,
        decision_error_std,
        execution_error_std,
        episode_seed,
    )
    effective_risk = float(risk_weight)
    if policy_name == POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID:
        effective_risk = limited.adaptive_risk_weight(
            env,
            args,
            execution_risk_error_std(decision_error_std, execution_error_std),
            slot_idx,
            base_weight=risk_weight,
            error_ref=adaptive_risk_error_ref,
            error_gain=adaptive_risk_error_gain,
            deadline_relief=adaptive_risk_deadline_relief,
            backlog_relief=adaptive_risk_backlog_relief,
        )
    return (
        limited.best_risk_aware_candidate(
            candidates,
            args,
            slot_idx,
            risk_weight=effective_risk,
            risk_power_weight=risk_power_weight,
            risk_invite_threshold=risk_invite_threshold,
        ),
        preview_calls,
        preview_calls,
    )
