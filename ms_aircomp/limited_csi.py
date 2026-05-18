"""Reusable limited-CSI policy helpers for MS-AirComp experiments.

This module contains the policy constants and candidate-construction helpers
shared by limited-CSI, execution-mismatch, and learned-shortlist experiments.
Top-level scripts keep CLI parsing, orchestration, and reporting.
"""

import numpy as np

from test_env import MSAirCompEnv


POLICY_EXACT_GREEDY = "Exact Greedy Full CSI"
POLICY_EST_GREEDY = "Estimated Greedy Full Preview"
POLICY_RANDOM_PROBE = "Estimated Random Probe"
POLICY_ROTATING_GRID = "Estimated Rotating Grid"
POLICY_ROBUST_ROTATING_GRID = "Robust Rotating Grid"
POLICY_RISK_AWARE_ROTATING_GRID = "Risk-Aware Rotating Grid"
POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID = "Adaptive Risk-Aware Rotating Grid"
POLICY_FIXED_IRS = "Estimated Fixed IRS"
POLICY_NO_IRS = "Estimated No IRS"


POLICY_OFFSETS = {
    POLICY_EXACT_GREEDY: 0x243F6A88,
    POLICY_EST_GREEDY: 0x85A308D3,
    POLICY_RANDOM_PROBE: 0x13198A2E,
    POLICY_ROTATING_GRID: 0x03707344,
    POLICY_ROBUST_ROTATING_GRID: 0xA4093822,
    POLICY_RISK_AWARE_ROTATING_GRID: 0x452821E6,
    POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID: 0xBE5466CF,
    POLICY_FIXED_IRS: 0x299F31D0,
    POLICY_NO_IRS: 0x082EFA98,
}


def parse_int_list(value):
    """Parse a comma-separated integer list such as '1,2,4,8'."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    """Parse a comma-separated float list such as '0,0.05,0.1'."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def unique_fill(indices, budget, num_codebook_states):
    """Return up to budget unique indices, filling deterministically when needed."""
    selected = []
    seen = set()
    for index in indices:
        clean_index = int(index) % num_codebook_states
        if clean_index not in seen:
            selected.append(clean_index)
            seen.add(clean_index)
        if len(selected) >= budget:
            return selected

    for index in range(num_codebook_states):
        if index not in seen:
            selected.append(index)
            seen.add(index)
        if len(selected) >= budget:
            break
    return selected


def grid_indices(num_codebook_states, budget, offset=0):
    """Select roughly evenly spaced codebook indices with a circular offset."""
    budget = min(int(budget), num_codebook_states)
    if budget >= num_codebook_states:
        return list(range(num_codebook_states))
    raw = np.floor(np.arange(budget) * num_codebook_states / budget).astype(int)
    return unique_fill(raw + int(offset), budget, num_codebook_states)


def stable_rng(episode_seed, error_std, policy_name, budget, salt=0, gain_margin=1.0, power_margin=1.0):
    """Create deterministic RNG streams for CSI errors and random probing."""
    if episode_seed is None:
        return np.random.default_rng()
    error_tag = int(round(float(error_std) * 1_000_000))
    gain_tag = int(round(float(gain_margin) * 10_000))
    power_tag = int(round(float(power_margin) * 10_000))
    seed = (
        int(episode_seed)
        + POLICY_OFFSETS[policy_name]
        + int(budget) * 0x9E3779B1
        + error_tag * 0x85EBCA6B
        + gain_tag * 0xC2B2AE35
        + power_tag * 0x27D4EB2F
        + int(salt) * 0x165667B1
    ) % (2**32)
    return np.random.default_rng(seed)


def make_env(args):
    """Create the base codebook environment used for limited-CSI evaluation."""
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode="codebook",
    )


def print_progress(name, error_std, budget, ep, episodes, success_nodes, num_nodes):
    """Print limited-CSI policy progress at 10 percent intervals."""
    interval = max(episodes // 10, 1)
    if ep % interval == 0 or ep == episodes:
        recent = np.mean(success_nodes[-interval:])
        print(
            f"  {name} err={error_std:g} B={budget}: "
            f"[{ep:04d}/{episodes:04d}] recent success {recent:.2f}/{num_nodes}"
        )


def effective_channels(env, indices=None, no_irs=False):
    """
    Compute true equivalent channels for codebook indices or the no-IRS link.

    Returns an array with shape (num_candidates, K).
    """
    if no_irs:
        return env.h_d[np.newaxis, :]

    clean_indices = np.asarray(indices, dtype=int)
    weighted_reflection = env.h_r * env.h_bs_r
    cascade = weighted_reflection @ env.codebook[clean_indices].T
    cascade = (cascade.T / np.sqrt(env.M)) * 0.05
    return env.h_d[np.newaxis, :] + cascade


def success_gain_threshold(env, args):
    """Return the true gain threshold implied by channel and power constraints."""
    power_limited_gain = (float(args.alpha_th) ** 2) / max(float(env.P_max), 1e-12)
    return max(float(args.g_th), power_limited_gain)


def estimate_success_reliability(env, args, h_total, error_scale):
    """
    Estimate per-node success probability from noisy equivalent-channel distance.

    This is a heuristic posterior: nodes far above the feasibility amplitude
    threshold receive reliability close to one; borderline nodes remain risky.
    """
    h_abs = np.abs(h_total)
    amp_threshold = np.sqrt(success_gain_threshold(env, args))
    if float(error_scale) <= 1e-12:
        return (h_abs >= amp_threshold).astype(float)

    normalized_excess = (h_abs - amp_threshold) / max(float(error_scale), 1e-12)
    normalized_excess = np.clip(normalized_excess, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-normalized_excess))


def build_candidate(
    env,
    args,
    irs_index,
    h_total,
    gain_margin=1.0,
    power_margin=1.0,
    no_irs=False,
    error_scale=0.0,
):
    """Build a schedulability candidate from one equivalent channel vector."""
    h_gain = np.abs(h_total) ** 2
    required_power = (args.alpha_th**2) / (h_gain + 1e-12)
    remaining = ~env.transmitted_flags
    valid_mask = (
        remaining
        & (h_gain >= args.g_th * float(gain_margin))
        & (required_power <= env.P_max * float(power_margin))
    )
    tx_count = int(np.sum(valid_mask))
    power_avg = float(np.mean(required_power[valid_mask])) if tx_count > 0 else 0.0
    mean_gain_remaining = float(np.mean(h_gain[remaining])) if np.any(remaining) else 0.0
    return {
        "irs_index": -2 if no_irs else int(irs_index),
        "valid_mask": valid_mask,
        "tx_this_slot": tx_count,
        "required_power": required_power,
        "h_gain": h_gain,
        "success_reliability": estimate_success_reliability(env, args, h_total, error_scale),
        "success_margin": h_gain / max(success_gain_threshold(env, args), 1e-12),
        "power_avg": power_avg,
        "mean_gain_remaining": mean_gain_remaining,
    }


def true_preview_candidates(env, args, indices=None, no_irs=False):
    """Preview candidates with the true channel."""
    if no_irs:
        return [build_candidate(env, args, -2, effective_channels(env, no_irs=True)[0], no_irs=True)]

    clean_indices = [int(np.clip(index, 0, args.num_codebook_states - 1)) for index in indices]
    channels = effective_channels(env, clean_indices)
    return [
        build_candidate(env, args, index, channels[row_idx])
        for row_idx, index in enumerate(clean_indices)
    ]


def estimated_preview_candidates(
    env,
    args,
    indices=None,
    error_std=0.0,
    rng=None,
    gain_margin=1.0,
    power_margin=1.0,
    no_irs=False,
):
    """Preview candidates with noisy estimated equivalent channels."""
    if rng is None:
        rng = np.random.default_rng()

    if no_irs:
        clean_indices = [-2]
        h_total = effective_channels(env, no_irs=True)
    else:
        clean_indices = [int(np.clip(index, 0, args.num_codebook_states - 1)) for index in indices]
        h_total = effective_channels(env, clean_indices)

    error_scales = np.zeros((len(clean_indices), 1), dtype=float)
    if error_std > 0.0:
        rms = np.sqrt(np.mean(np.abs(h_total) ** 2, axis=1, keepdims=True))
        noise = (rng.normal(size=h_total.shape) + 1j * rng.normal(size=h_total.shape)) / np.sqrt(2.0)
        h_total = h_total + float(error_std) * np.maximum(rms, 1e-12) * noise
        estimated_rms = np.sqrt(np.mean(np.abs(h_total) ** 2, axis=1, keepdims=True))
        error_scales = float(error_std) * np.maximum(estimated_rms, 1e-12)

    return [
        build_candidate(
            env,
            args,
            clean_indices[row_idx],
            h_total[row_idx],
            gain_margin=gain_margin,
            power_margin=power_margin,
            no_irs=no_irs,
            error_scale=float(error_scales[row_idx, 0]),
        )
        for row_idx in range(len(clean_indices))
    ]


def candidate_key(candidate):
    """Greedy ranking key: scheduled nodes first, then lower power, then gain."""
    tx_count = int(candidate["tx_this_slot"])
    power_avg = float(candidate["power_avg"])
    mean_gain = float(candidate["mean_gain_remaining"])
    power_tiebreak = -power_avg if tx_count > 0 else 0.0
    return tx_count, power_tiebreak, mean_gain


def best_candidate(candidates):
    """Return the best candidate by the project greedy ranking rule."""
    return max(candidates, key=candidate_key)


def effective_risk_invite_threshold(args, slot_idx, risk_invite_threshold):
    """
    Lower the reliability cutoff as the deadline approaches.

    Early slots can avoid borderline nodes; late slots should become less
    conservative because unserved nodes have fewer future chances.
    """
    if args.num_slots <= 1:
        return min(float(risk_invite_threshold), 0.5)
    progress = float(slot_idx) / float(max(args.num_slots - 1, 1))
    return max(0.5, float(risk_invite_threshold) - 0.1 * progress)


def adaptive_risk_weight(
    env,
    args,
    error_std,
    slot_idx,
    base_weight=0.5,
    error_ref=0.3,
    error_gain=1.0,
    deadline_relief=0.6,
    backlog_relief=0.8,
):
    """
    Adapt risk aversion to CSI uncertainty and multi-slot urgency.

    Higher CSI error increases risk aversion. As the episode approaches the
    deadline, or the remaining-node backlog is high for the available slots,
    the policy becomes less conservative to avoid missed opportunities.
    """
    if float(base_weight) <= 0.0:
        return 0.0

    error_pressure = min(float(error_std) / max(float(error_ref), 1e-12), 2.0)
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

    numerator = float(base_weight) * (1.0 + float(error_gain) * error_pressure)
    relief = 1.0 + float(deadline_relief) * deadline_pressure + float(backlog_relief) * backlog_pressure
    return max(0.0, numerator / max(relief, 1e-12))


def risk_aware_candidate(
    candidate,
    args,
    slot_idx,
    risk_weight=0.5,
    risk_power_weight=0.1,
    risk_invite_threshold=0.5,
):
    """Convert an estimated candidate into a risk-aware invitation decision."""
    estimated_valid_mask = np.asarray(candidate["valid_mask"], dtype=bool)
    reliability = np.asarray(candidate["success_reliability"], dtype=float)
    threshold = effective_risk_invite_threshold(args, slot_idx, risk_invite_threshold)
    invite_mask = estimated_valid_mask & (reliability >= threshold)

    scheduled_power = candidate["required_power"][invite_mask]
    tx_count = int(np.sum(invite_mask))
    power_avg = float(np.mean(scheduled_power)) if tx_count > 0 else 0.0
    expected_success = float(np.sum(reliability[invite_mask]))
    risk_mass = float(np.sum(1.0 - reliability[invite_mask]))
    risk_score = expected_success - float(risk_weight) * risk_mass - float(risk_power_weight) * power_avg

    adjusted = dict(candidate)
    adjusted["estimated_valid_mask"] = estimated_valid_mask
    adjusted["valid_mask"] = invite_mask
    adjusted["tx_this_slot"] = tx_count
    adjusted["power_avg"] = power_avg
    adjusted["expected_success"] = expected_success
    adjusted["risk_mass"] = risk_mass
    adjusted["risk_score"] = risk_score
    adjusted["effective_risk_invite_threshold"] = threshold
    adjusted["effective_risk_weight"] = float(risk_weight)
    adjusted["risk_rejected_count"] = int(np.sum(estimated_valid_mask & (~invite_mask)))
    return adjusted


def risk_aware_candidate_key(candidate):
    """Rank risk-aware candidates by expected reliable progress."""
    return (
        float(candidate["risk_score"]),
        float(candidate["expected_success"]),
        int(candidate["tx_this_slot"]),
        -float(candidate["power_avg"]) if int(candidate["tx_this_slot"]) > 0 else 0.0,
        float(candidate["mean_gain_remaining"]),
    )


def best_risk_aware_candidate(
    candidates,
    args,
    slot_idx,
    risk_weight=0.5,
    risk_power_weight=0.1,
    risk_invite_threshold=0.5,
):
    """Return the best candidate after reliability-aware invitation filtering."""
    adjusted = [
        risk_aware_candidate(
            candidate,
            args,
            slot_idx,
            risk_weight=risk_weight,
            risk_power_weight=risk_power_weight,
            risk_invite_threshold=risk_invite_threshold,
        )
        for candidate in candidates
    ]
    return max(adjusted, key=risk_aware_candidate_key)


def select_indices(policy_name, args, budget, slot_idx, rng):
    """Select codebook indices to probe for one limited-CSI policy."""
    budget = min(int(budget), args.num_codebook_states)
    if policy_name in {POLICY_EXACT_GREEDY, POLICY_EST_GREEDY}:
        return list(range(args.num_codebook_states))
    if policy_name == POLICY_FIXED_IRS:
        return [args.fixed_irs_index]
    if policy_name == POLICY_RANDOM_PROBE:
        return [int(index) for index in rng.choice(args.num_codebook_states, size=budget, replace=False)]
    if policy_name in {
        POLICY_ROTATING_GRID,
        POLICY_ROBUST_ROTATING_GRID,
        POLICY_RISK_AWARE_ROTATING_GRID,
        POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
    }:
        return grid_indices(args.num_codebook_states, budget, offset=slot_idx)
    raise ValueError(f"Policy does not use IRS codebook indices: {policy_name}")


def choose_policy_candidate(
    env,
    args,
    policy_name,
    budget,
    slot_idx,
    error_std,
    episode_seed,
    gain_margin=1.0,
    power_margin=1.0,
    risk_weight=0.5,
    risk_power_weight=0.1,
    risk_invite_threshold=0.5,
    adaptive_risk_error_ref=0.3,
    adaptive_risk_error_gain=1.0,
    adaptive_risk_deadline_relief=0.6,
    adaptive_risk_backlog_relief=0.8,
):
    """
    Choose an IRS/no-IRS candidate and return decision metadata.

    The returned candidate supplies the scheduled mask. True execution is handled
    separately by execute_limited_csi_slot().
    """
    if policy_name == POLICY_NO_IRS:
        rng = stable_rng(episode_seed, error_std, policy_name, 0, salt=1 + slot_idx)
        estimated = estimated_preview_candidates(env, args, error_std=error_std, rng=rng, no_irs=True)
        return estimated[0], 0, 0

    if policy_name == POLICY_EXACT_GREEDY:
        candidates = true_preview_candidates(env, args, range(args.num_codebook_states))
        return best_candidate(candidates), args.num_codebook_states, args.num_codebook_states

    random_rng = stable_rng(
        episode_seed,
        error_std,
        policy_name,
        budget,
        salt=2 + slot_idx,
        gain_margin=gain_margin,
        power_margin=power_margin,
    )
    error_rng = stable_rng(
        episode_seed,
        error_std,
        policy_name,
        budget,
        salt=3 + slot_idx,
        gain_margin=gain_margin,
        power_margin=power_margin,
    )
    indices = select_indices(policy_name, args, budget, slot_idx, random_rng)
    candidates = estimated_preview_candidates(
        env,
        args,
        indices=indices,
        error_std=error_std,
        rng=error_rng,
        gain_margin=gain_margin,
        power_margin=power_margin,
    )
    preview_calls = len(indices)
    if policy_name in {POLICY_RISK_AWARE_ROTATING_GRID, POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID}:
        effective_risk = float(risk_weight)
        if policy_name == POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID:
            effective_risk = adaptive_risk_weight(
                env,
                args,
                error_std,
                slot_idx,
                base_weight=risk_weight,
                error_ref=adaptive_risk_error_ref,
                error_gain=adaptive_risk_error_gain,
                deadline_relief=adaptive_risk_deadline_relief,
                backlog_relief=adaptive_risk_backlog_relief,
            )
        return (
            best_risk_aware_candidate(
                candidates,
                args,
                slot_idx,
                risk_weight=effective_risk,
                risk_power_weight=risk_power_weight,
                risk_invite_threshold=risk_invite_threshold,
            ),
            preview_calls,
            len(indices),
        )
    return best_candidate(candidates), preview_calls, len(indices)


def execute_limited_csi_slot(env, args, decision_candidate, true_candidate):
    """
    Execute a limited-CSI decision against the true channel.

    Only nodes invited by decision_candidate["valid_mask"] may transmit. A node
    is successful only if it is also valid under true_candidate["valid_mask"].
    """
    remaining = ~env.transmitted_flags
    scheduled_mask = decision_candidate["valid_mask"] & remaining
    true_valid_mask = true_candidate["valid_mask"] & remaining
    success_mask = scheduled_mask & true_valid_mask
    failed_mask = scheduled_mask & (~true_valid_mask)
    missed_mask = true_valid_mask & (~scheduled_mask)

    scheduled_count = int(np.sum(scheduled_mask))
    success_count = int(np.sum(success_mask))
    failed_count = int(np.sum(failed_mask))
    missed_count = int(np.sum(missed_mask))
    true_opportunity_count = int(np.sum(true_valid_mask))
    scheduled_power = decision_candidate["required_power"][scheduled_mask]
    power_avg = float(np.mean(scheduled_power)) if scheduled_count > 0 else 0.0
    attempted_energy = float(np.sum(scheduled_power)) if scheduled_count > 0 else 0.0

    env.transmitted_flags |= success_mask
    env.current_slot += 1
    total_tx = int(np.sum(env.transmitted_flags))
    all_done = total_tx >= args.num_nodes
    time_limit = env.current_slot >= args.num_slots
    done = all_done or time_limit

    reward = success_count * 2.0
    if scheduled_count > 0:
        reward -= 0.5 * power_avg
    if done:
        missed_nodes = args.num_nodes - total_tx
        reward -= missed_nodes**2 * 0.5

    return {
        "tx_this_slot": success_count,
        "scheduled_this_slot": scheduled_count,
        "failed_this_slot": failed_count,
        "missed_opportunity_this_slot": missed_count,
        "true_opportunity_this_slot": true_opportunity_count,
        "total_tx": total_tx,
        "slots_used": int(env.current_slot),
        "power_avg": power_avg,
        "attempted_energy": attempted_energy,
        "reward": float(reward),
        "is_complete": all_done,
        "termination_reason": "complete" if all_done else "time_limit" if time_limit else "running",
    }, done


def oracle_candidate(env, args):
    """Return the true full-CSI greedy candidate for diagnostics."""
    return best_candidate(true_preview_candidates(env, args, range(args.num_codebook_states)))


def true_candidate_for_decision(env, args, decision_candidate):
    """Return the true candidate corresponding to a chosen IRS/no-IRS decision."""
    irs_index = int(decision_candidate["irs_index"])
    if irs_index == -2:
        return true_preview_candidates(env, args, no_irs=True)[0]
    return true_preview_candidates(env, args, [irs_index])[0]


__all__ = [
    "POLICY_EXACT_GREEDY",
    "POLICY_EST_GREEDY",
    "POLICY_RANDOM_PROBE",
    "POLICY_ROTATING_GRID",
    "POLICY_ROBUST_ROTATING_GRID",
    "POLICY_RISK_AWARE_ROTATING_GRID",
    "POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID",
    "POLICY_FIXED_IRS",
    "POLICY_NO_IRS",
    "POLICY_OFFSETS",
    "adaptive_risk_weight",
    "best_candidate",
    "best_risk_aware_candidate",
    "build_candidate",
    "candidate_key",
    "choose_policy_candidate",
    "effective_channels",
    "effective_risk_invite_threshold",
    "estimate_success_reliability",
    "estimated_preview_candidates",
    "execute_limited_csi_slot",
    "grid_indices",
    "make_env",
    "oracle_candidate",
    "parse_float_list",
    "parse_int_list",
    "print_progress",
    "risk_aware_candidate",
    "risk_aware_candidate_key",
    "select_indices",
    "stable_rng",
    "success_gain_threshold",
    "true_candidate_for_decision",
    "true_preview_candidates",
    "unique_fill",
]
