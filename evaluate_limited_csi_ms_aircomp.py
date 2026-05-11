"""
Limited-CSI IRS-assisted multi-slot AirComp evaluation.

This script separates what a policy can infer from limited/noisy CSI from what
actually succeeds on the true channel. A policy first probes a subset of IRS
codebook indices, builds an estimated schedulable-node mask, chooses an IRS
index, and invites only those estimated-valid nodes. Execution is then verified
against the true channel; nodes that were not invited are not auto-scheduled.

The goal is to evaluate whether dynamic IRS selection still helps multi-slot
AirComp when full per-codebook CSI/features are unavailable.
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_partial_probing_sweep import grid_indices
from evaluate_policy_comparison import (
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
)
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


NUMERIC_RESULT_KEYS = (
    "success_nodes",
    "avg_power",
    "episode_reward",
    "slots_used",
    "total_energy",
    "scheduled_nodes",
    "failed_nodes",
    "missed_opportunities",
    "true_opportunities",
    "failure_slots",
    "decision_preview_calls_per_slot",
    "oracle_tx_gap_mean",
    "effective_risk_weight",
)


def parse_int_list(value):
    """Parse a comma-separated integer list such as '1,2,4,8'."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    """Parse a comma-separated float list such as '0,0.05,0.1'."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    """Parse limited-CSI evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate IRS-assisted MS-AirComp under limited/noisy CSI."
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--probe-budgets", default="2,4,8")
    parser.add_argument("--error-std-values", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--robust-gain-margins", default="1.25")
    parser.add_argument("--robust-power-margins", default="0.9")
    parser.add_argument("--risk-weights", default="0.5")
    parser.add_argument("--risk-power-weights", default="0.1")
    parser.add_argument("--risk-invite-thresholds", default="0.5")
    parser.add_argument("--adaptive-risk-base-weights", default="0.5")
    parser.add_argument("--adaptive-risk-error-ref", type=float, default=0.3)
    parser.add_argument("--adaptive-risk-error-gain", type=float, default=1.0)
    parser.add_argument("--adaptive-risk-deadline-relief", type=float, default=0.6)
    parser.add_argument("--adaptive-risk-backlog-relief", type=float, default=0.8)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--fixed-irs-index", type=int, default=7)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """Validate sizes, budgets, error levels, and robust margins."""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    for name in ("num_nodes", "num_slots", "num_irs_elements", "num_codebook_states"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.g_th <= 0:
        raise ValueError("--g-th must be positive")
    if args.alpha_th <= 0:
        raise ValueError("--alpha-th must be positive")

    budgets = parse_int_list(args.probe_budgets)
    if not budgets:
        raise ValueError("--probe-budgets must contain at least one value")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("--probe-budgets must contain positive integers")
    args.probe_budgets = sorted({min(int(budget), args.num_codebook_states) for budget in budgets})

    args.error_std_values = parse_float_list(args.error_std_values)
    if not args.error_std_values:
        raise ValueError("--error-std-values must contain at least one value")
    if any(value < 0.0 for value in args.error_std_values):
        raise ValueError("--error-std-values must be non-negative")

    args.robust_gain_margins = parse_float_list(args.robust_gain_margins)
    args.robust_power_margins = parse_float_list(args.robust_power_margins)
    if not args.robust_gain_margins or not args.robust_power_margins:
        raise ValueError("robust margin lists must not be empty")
    if any(value <= 0.0 for value in args.robust_gain_margins):
        raise ValueError("--robust-gain-margins must be positive")
    if any(value <= 0.0 for value in args.robust_power_margins):
        raise ValueError("--robust-power-margins must be positive")

    args.risk_weights = parse_float_list(args.risk_weights)
    args.risk_power_weights = parse_float_list(args.risk_power_weights)
    args.risk_invite_thresholds = parse_float_list(args.risk_invite_thresholds)
    if not args.risk_weights or not args.risk_power_weights or not args.risk_invite_thresholds:
        raise ValueError("risk-aware parameter lists must not be empty")
    if any(value < 0.0 for value in args.risk_weights):
        raise ValueError("--risk-weights must be non-negative")
    if any(value < 0.0 for value in args.risk_power_weights):
        raise ValueError("--risk-power-weights must be non-negative")
    if any(value < 0.0 or value > 1.0 for value in args.risk_invite_thresholds):
        raise ValueError("--risk-invite-thresholds must be in [0, 1]")

    args.adaptive_risk_base_weights = parse_float_list(args.adaptive_risk_base_weights)
    if not args.adaptive_risk_base_weights:
        raise ValueError("--adaptive-risk-base-weights must contain at least one value")
    if any(value < 0.0 for value in args.adaptive_risk_base_weights):
        raise ValueError("--adaptive-risk-base-weights must be non-negative")
    if args.adaptive_risk_error_ref <= 0.0:
        raise ValueError("--adaptive-risk-error-ref must be positive")
    if args.adaptive_risk_error_gain < 0.0:
        raise ValueError("--adaptive-risk-error-gain must be non-negative")
    if args.adaptive_risk_deadline_relief < 0.0:
        raise ValueError("--adaptive-risk-deadline-relief must be non-negative")
    if args.adaptive_risk_backlog_relief < 0.0:
        raise ValueError("--adaptive-risk-backlog-relief must be non-negative")

    args.fixed_irs_index = int(np.clip(args.fixed_irs_index, 0, args.num_codebook_states - 1))


def resolve_output_prefix(args):
    """Resolve output prefix for CSVs and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    budget_label = "-".join(format_float_for_suffix(value) for value in args.probe_budgets)
    error_label = "-".join(format_float_for_suffix(value) for value in args.error_std_values)
    gain_label = "-".join(format_float_for_suffix(value) for value in args.robust_gain_margins)
    power_label = "-".join(format_float_for_suffix(value) for value in args.robust_power_margins)
    risk_label = "-".join(format_float_for_suffix(value) for value in args.risk_weights)
    risk_power_label = "-".join(format_float_for_suffix(value) for value in args.risk_power_weights)
    risk_threshold_label = "-".join(format_float_for_suffix(value) for value in args.risk_invite_thresholds)
    adaptive_risk_label = "-".join(
        format_float_for_suffix(value) for value in args.adaptive_risk_base_weights
    )
    suffix = (
        f"ep{args.episodes}_runs{args.num_seeds}_{seed_label}_b{budget_label}_"
        f"err{error_label}_gm{gain_label}_pm{power_label}_"
        f"rw{risk_label}_rp{risk_power_label}_rt{risk_threshold_label}_"
        f"arw{adaptive_risk_label}"
    )
    output_prefix = os.path.join("results", "limited_csi", f"limited_csi_ms_aircomp_{suffix}")
    ensure_parent_dir(output_prefix)
    return output_prefix


def make_env(args):
    """Create the base codebook environment used for limited-CSI evaluation."""
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode="codebook",
    )


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

    Only nodes invited by decision_candidate["valid_mask"] may transmit. A node is
    successful only if it is also valid under true_candidate["valid_mask"].
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


def evaluate_policy(
    episode_seeds,
    args,
    error_std,
    policy_name,
    budget,
    gain_margin=1.0,
    power_margin=1.0,
    risk_weight=0.0,
    risk_power_weight=0.0,
    risk_invite_threshold=0.0,
    adaptive_risk_error_ref=0.3,
    adaptive_risk_error_gain=1.0,
    adaptive_risk_deadline_relief=0.6,
    adaptive_risk_backlog_relief=0.8,
):
    """Evaluate one limited-CSI policy for one error level and budget."""
    env = make_env(args)
    success_nodes = []
    avg_power = []
    rewards = []
    slots_used = []
    total_energy = []
    scheduled_nodes = []
    failed_nodes = []
    missed_opportunities = []
    true_opportunities = []
    failure_slots = []
    preview_calls_per_slot = []
    oracle_tx_gap_mean = []
    effective_risk_weights = []

    print(
        f"Running {policy_name} err={error_std:g} B={budget} "
        f"gm={gain_margin:g} pm={power_margin:g} "
        f"rw={risk_weight:g} rp={risk_power_weight:g} rt={risk_invite_threshold:g}..."
    )
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        episode_power = []
        episode_reward = 0.0
        episode_energy = 0.0
        episode_scheduled = 0
        episode_failed = 0
        episode_missed = 0
        episode_true_opportunities = 0
        episode_failure_slots = 0
        episode_preview_calls = []
        episode_oracle_gaps = []
        episode_effective_risk_weights = []
        total_tx = 0
        episode_slots = args.num_slots

        for slot_idx in range(args.num_slots):
            oracle = oracle_candidate(env, args)
            decision, preview_calls, _candidate_count = choose_policy_candidate(
                env,
                args,
                policy_name,
                budget,
                slot_idx,
                error_std,
                episode_seed,
                gain_margin=gain_margin,
                power_margin=power_margin,
                risk_weight=risk_weight,
                risk_power_weight=risk_power_weight,
                risk_invite_threshold=risk_invite_threshold,
                adaptive_risk_error_ref=adaptive_risk_error_ref,
                adaptive_risk_error_gain=adaptive_risk_error_gain,
                adaptive_risk_deadline_relief=adaptive_risk_deadline_relief,
                adaptive_risk_backlog_relief=adaptive_risk_backlog_relief,
            )
            true_selected = true_candidate_for_decision(env, args, decision)
            info, done = execute_limited_csi_slot(env, args, decision, true_selected)

            total_tx = int(info["total_tx"])
            episode_slots = int(info["slots_used"])
            episode_reward += float(info["reward"])
            episode_energy += float(info["attempted_energy"])
            episode_scheduled += int(info["scheduled_this_slot"])
            episode_failed += int(info["failed_this_slot"])
            episode_missed += int(info["missed_opportunity_this_slot"])
            episode_true_opportunities += int(info["true_opportunity_this_slot"])
            episode_failure_slots += int(info["failed_this_slot"] > 0)
            episode_preview_calls.append(int(preview_calls))
            episode_effective_risk_weights.append(float(decision.get("effective_risk_weight", 0.0)))
            episode_oracle_gaps.append(
                max(0.0, float(oracle["tx_this_slot"]) - float(true_selected["tx_this_slot"]))
            )
            if info["scheduled_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            if done:
                break

        success_nodes.append(total_tx)
        avg_power.append(float(np.mean(episode_power)) if episode_power else 0.0)
        rewards.append(float(episode_reward))
        slots_used.append(int(episode_slots))
        total_energy.append(float(episode_energy))
        scheduled_nodes.append(float(episode_scheduled))
        failed_nodes.append(float(episode_failed))
        missed_opportunities.append(float(episode_missed))
        true_opportunities.append(float(episode_true_opportunities))
        failure_slots.append(float(episode_failure_slots))
        active_slots = max(len(episode_preview_calls), 1)
        preview_calls_per_slot.append(float(sum(episode_preview_calls)) / active_slots)
        oracle_tx_gap_mean.append(float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0)
        effective_risk_weights.append(
            float(np.mean(episode_effective_risk_weights)) if episode_effective_risk_weights else 0.0
        )

        print_progress(policy_name, error_std, budget, ep, args.episodes, success_nodes, args.num_nodes)

    return {
        "name": policy_name,
        "error_std": float(error_std),
        "probe_budget": int(budget),
        "gain_margin": float(gain_margin),
        "power_margin": float(power_margin),
        "risk_weight": float(risk_weight),
        "risk_power_weight": float(risk_power_weight),
        "risk_invite_threshold": float(risk_invite_threshold),
        "success_nodes": np.asarray(success_nodes, dtype=float),
        "avg_power": np.asarray(avg_power, dtype=float),
        "episode_reward": np.asarray(rewards, dtype=float),
        "slots_used": np.asarray(slots_used, dtype=float),
        "total_energy": np.asarray(total_energy, dtype=float),
        "scheduled_nodes": np.asarray(scheduled_nodes, dtype=float),
        "failed_nodes": np.asarray(failed_nodes, dtype=float),
        "missed_opportunities": np.asarray(missed_opportunities, dtype=float),
        "true_opportunities": np.asarray(true_opportunities, dtype=float),
        "failure_slots": np.asarray(failure_slots, dtype=float),
        "decision_preview_calls_per_slot": np.asarray(preview_calls_per_slot, dtype=float),
        "oracle_tx_gap_mean": np.asarray(oracle_tx_gap_mean, dtype=float),
        "effective_risk_weight": np.asarray(effective_risk_weights, dtype=float),
    }


def print_progress(name, error_std, budget, ep, episodes, success_nodes, num_nodes):
    """Print progress at 10 percent intervals."""
    interval = max(episodes // 10, 1)
    if ep % interval == 0 or ep == episodes:
        recent = np.mean(success_nodes[-interval:])
        print(
            f"  {name} err={error_std:g} B={budget}: "
            f"[{ep:04d}/{episodes:04d}] recent success {recent:.2f}/{num_nodes}"
        )


def policy_suite_for_error(args, episode_seeds, error_std):
    """Run all policy/budget combinations for one error level and run seed."""
    results = [
        evaluate_policy(
            episode_seeds,
            args,
            error_std,
            POLICY_NO_IRS,
            budget=0,
        ),
        evaluate_policy(
            episode_seeds,
            args,
            error_std,
            POLICY_FIXED_IRS,
            budget=1,
        ),
        evaluate_policy(
            episode_seeds,
            args,
            error_std,
            POLICY_EXACT_GREEDY,
            budget=args.num_codebook_states,
        ),
        evaluate_policy(
            episode_seeds,
            args,
            error_std,
            POLICY_EST_GREEDY,
            budget=args.num_codebook_states,
        ),
    ]

    for budget in args.probe_budgets:
        results.append(evaluate_policy(episode_seeds, args, error_std, POLICY_RANDOM_PROBE, budget=budget))
        results.append(evaluate_policy(episode_seeds, args, error_std, POLICY_ROTATING_GRID, budget=budget))
        for gain_margin in args.robust_gain_margins:
            for power_margin in args.robust_power_margins:
                results.append(
                    evaluate_policy(
                        episode_seeds,
                        args,
                        error_std,
                        POLICY_ROBUST_ROTATING_GRID,
                        budget=budget,
                        gain_margin=gain_margin,
                        power_margin=power_margin,
                    )
                )
        for risk_weight in args.risk_weights:
            for risk_power_weight in args.risk_power_weights:
                for risk_invite_threshold in args.risk_invite_thresholds:
                    results.append(
                        evaluate_policy(
                            episode_seeds,
                            args,
                            error_std,
                            POLICY_RISK_AWARE_ROTATING_GRID,
                            budget=budget,
                            risk_weight=risk_weight,
                            risk_power_weight=risk_power_weight,
                            risk_invite_threshold=risk_invite_threshold,
                        )
                    )
        for risk_weight in args.adaptive_risk_base_weights:
            for risk_power_weight in args.risk_power_weights:
                for risk_invite_threshold in args.risk_invite_thresholds:
                    results.append(
                        evaluate_policy(
                            episode_seeds,
                            args,
                            error_std,
                            POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
                            budget=budget,
                            risk_weight=risk_weight,
                            risk_power_weight=risk_power_weight,
                            risk_invite_threshold=risk_invite_threshold,
                            adaptive_risk_error_ref=args.adaptive_risk_error_ref,
                            adaptive_risk_error_gain=args.adaptive_risk_error_gain,
                            adaptive_risk_deadline_relief=args.adaptive_risk_deadline_relief,
                            adaptive_risk_backlog_relief=args.adaptive_risk_backlog_relief,
                        )
                    )
    return results


def seed_summary(result):
    """Compress one run seed result to seed-level means."""
    return {key: float(np.mean(result[key])) for key in NUMERIC_RESULT_KEYS}


def aggregate_seed_results(seed_result_sets):
    """Aggregate result lists across run seeds."""
    if not seed_result_sets:
        return []

    aggregated_results = []
    result_count = len(seed_result_sets[0])
    for result_idx in range(result_count):
        parts = [seed_results[result_idx] for seed_results in seed_result_sets]
        aggregated = {
            "name": parts[0]["name"],
            "error_std": parts[0]["error_std"],
            "probe_budget": parts[0]["probe_budget"],
            "gain_margin": parts[0]["gain_margin"],
            "power_margin": parts[0]["power_margin"],
            "risk_weight": parts[0]["risk_weight"],
            "risk_power_weight": parts[0]["risk_power_weight"],
            "risk_invite_threshold": parts[0]["risk_invite_threshold"],
        }
        for key in NUMERIC_RESULT_KEYS:
            aggregated[key] = np.concatenate([part[key] for part in parts])
        aggregated["seed_summaries"] = [seed_summary(part) for part in parts]
        aggregated_results.append(aggregated)
    return aggregated_results


def metric_mean_ci(result, key):
    """Compute overall mean and run-seed 95 percent CI."""
    seed_values = np.asarray(
        [summary[key] for summary in result.get("seed_summaries", [seed_summary(result)])],
        dtype=float,
    )
    mean_value = float(np.mean(result[key]))
    if len(seed_values) <= 1:
        return mean_value, 0.0
    ci95 = 1.96 * float(np.std(seed_values, ddof=1)) / np.sqrt(len(seed_values))
    return mean_value, ci95


def safe_rate(numerator, denominator):
    """Return a percentage rate with a zero-denominator guard."""
    total_denominator = float(np.sum(denominator))
    if total_denominator <= 0.0:
        return 0.0
    return float(np.sum(numerator) / total_denominator * 100.0)


def summarize_results(args, results):
    """Convert aggregated results to CSV summary rows."""
    rows = []
    for result in results:
        success_mean, success_ci95 = metric_mean_ci(result, "success_nodes")
        slots_mean, slots_ci95 = metric_mean_ci(result, "slots_used")
        energy_mean, energy_ci95 = metric_mean_ci(result, "total_energy")
        scheduled_mean = float(np.mean(result["scheduled_nodes"]))
        failed_mean = float(np.mean(result["failed_nodes"]))
        true_opp_mean = float(np.mean(result["true_opportunities"]))
        missed_opp_mean = float(np.mean(result["missed_opportunities"]))
        failure_slots_mean = float(np.mean(result["failure_slots"]))
        rows.append(
            {
                "error_std": float(result["error_std"]),
                "probe_budget": int(result["probe_budget"]),
                "policy": result["name"],
                "gain_margin": float(result["gain_margin"]),
                "power_margin": float(result["power_margin"]),
                "risk_weight": float(result["risk_weight"]),
                "risk_power_weight": float(result["risk_power_weight"]),
                "risk_invite_threshold": float(result["risk_invite_threshold"]),
                "episodes": len(result["success_nodes"]),
                "num_seeds": args.num_seeds,
                "success_mean": success_mean,
                "success_ci95": success_ci95,
                "success_rate_mean": success_mean / args.num_nodes,
                "perfect_rate": float(np.mean(result["success_nodes"] == args.num_nodes) * 100.0),
                "slots_mean": slots_mean,
                "slots_ci95": slots_ci95,
                "avg_power": float(np.mean(result["avg_power"])),
                "total_energy_mean": energy_mean,
                "total_energy_ci95": energy_ci95,
                "scheduled_nodes_mean": scheduled_mean,
                "failed_nodes_mean": failed_mean,
                "true_opportunities_mean": true_opp_mean,
                "missed_opportunities_mean": missed_opp_mean,
                "false_positive_rate": safe_rate(result["failed_nodes"], result["scheduled_nodes"]),
                "execution_failure_rate": safe_rate(result["failed_nodes"], result["scheduled_nodes"]),
                "failure_slot_rate": safe_rate(result["failure_slots"], result["slots_used"]),
                "missed_opportunity_rate": safe_rate(
                    result["missed_opportunities"],
                    result["true_opportunities"],
                ),
                "decision_preview_calls_per_slot_mean": float(
                    np.mean(result["decision_preview_calls_per_slot"])
                ),
                "oracle_tx_gap_mean": float(np.mean(result["oracle_tx_gap_mean"])),
                "effective_risk_weight_mean": float(np.mean(result["effective_risk_weight"])),
                "avg_reward": float(np.mean(result["episode_reward"])),
            }
        )
    return rows


def write_csv(path, rows):
    """Write limited-CSI summary CSV."""
    ensure_parent_dir(path)
    fieldnames = [
        "error_std",
        "probe_budget",
        "policy",
        "gain_margin",
        "power_margin",
        "risk_weight",
        "risk_power_weight",
        "risk_invite_threshold",
        "episodes",
        "num_seeds",
        "success_mean",
        "success_ci95",
        "success_rate_mean",
        "perfect_rate",
        "slots_mean",
        "slots_ci95",
        "avg_power",
        "total_energy_mean",
        "total_energy_ci95",
        "scheduled_nodes_mean",
        "failed_nodes_mean",
        "true_opportunities_mean",
        "missed_opportunities_mean",
        "false_positive_rate",
        "execution_failure_rate",
        "failure_slot_rate",
        "missed_opportunity_rate",
        "decision_preview_calls_per_slot_mean",
        "oracle_tx_gap_mean",
        "effective_risk_weight_mean",
        "avg_reward",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def policy_label(row):
    """Build a compact plot/table label for one summary row."""
    label = row["policy"]
    if row["probe_budget"] not in {0, row.get("num_codebook_states", -1)}:
        label += f" B={int(row['probe_budget'])}"
    if row["policy"] == POLICY_ROBUST_ROTATING_GRID:
        label += f" gm={float(row['gain_margin']):g} pm={float(row['power_margin']):g}"
    if row["policy"] in {POLICY_RISK_AWARE_ROTATING_GRID, POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID}:
        label += (
            f" rw={float(row['risk_weight']):g}"
            f" rp={float(row['risk_power_weight']):g}"
            f" rt={float(row['risk_invite_threshold']):g}"
        )
    return label


def print_summary(rows):
    """Print a compact limited-CSI summary."""
    print("=" * 158)
    print("Limited-CSI MS-AirComp Summary")
    print("=" * 158)
    print(
        f"{'Err':>6} {'B':>3} {'Policy':<36} {'gm':>5} {'pm':>5} {'rw':>5} {'erw':>5} {'rt':>5} "
        f"{'Success':>9} {'Perfect%':>9} {'Slots':>8} {'Energy':>10} "
        f"{'Fail%':>8} {'MissOpp%':>9} {'Preview':>8} {'Gap':>7}"
    )
    for row in rows:
        print(
            f"{row['error_std']:>6.3f} {row['probe_budget']:>3} {row['policy']:<36} "
            f"{row['gain_margin']:>5.2f} {row['power_margin']:>5.2f} "
            f"{row['risk_weight']:>5.2f} {row['effective_risk_weight_mean']:>5.2f} "
            f"{row['risk_invite_threshold']:>5.2f} "
            f"{row['success_mean']:>9.3f} {row['perfect_rate']:>8.2f}% "
            f"{row['slots_mean']:>8.3f} {row['total_energy_mean']:>10.3f} "
            f"{row['execution_failure_rate']:>7.2f}% {row['missed_opportunity_rate']:>8.2f}% "
            f"{row['decision_preview_calls_per_slot_mean']:>8.2f} {row['oracle_tx_gap_mean']:>7.3f}"
        )


def plot_results(rows, args, output_prefix):
    """Plot perfect coverage, latency, and execution failure vs CSI error."""
    labels = []
    for row in rows:
        label = policy_label(row)
        if label not in labels:
            labels.append(label)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(idx % 20) for idx, label in enumerate(labels)}

    for label in labels:
        label_rows = sorted(
            [row for row in rows if policy_label(row) == label],
            key=lambda row: row["error_std"],
        )
        x = [row["error_std"] for row in label_rows]
        axes[0].plot(
            x,
            [row["perfect_rate"] for row in label_rows],
            marker="o",
            linewidth=1.7,
            label=label,
            color=colors[label],
        )
        axes[1].plot(
            x,
            [row["slots_mean"] for row in label_rows],
            marker="o",
            linewidth=1.7,
            label=label,
            color=colors[label],
        )
        axes[2].plot(
            x,
            [row["execution_failure_rate"] for row in label_rows],
            marker="o",
            linewidth=1.7,
            label=label,
            color=colors[label],
        )

    axes[0].set_title("Perfect Coverage vs CSI Error")
    axes[0].set_ylabel("Perfect Episodes (%)")
    axes[0].set_ylim(0.0, 103.0)
    axes[1].set_title("Latency vs CSI Error")
    axes[1].set_ylabel("Slots Used")
    axes[1].set_ylim(0.0, args.num_slots + 1)
    axes[2].set_title("Execution Failure vs CSI Error")
    axes[2].set_ylabel("Failed Invited Nodes (%)")

    for ax in axes:
        ax.set_xlabel("Equivalent Channel Error Std")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)

    fig.tight_layout()
    path = f"{output_prefix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    """Run limited-CSI MS-AirComp evaluation."""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)
    run_seeds = make_run_seeds(args)
    episode_seed_sets = [make_episode_seeds(args, run_seed) for run_seed in run_seeds]

    print("=" * 96)
    print(
        f"Limited-CSI MS-AirComp: episodes={args.episodes}, num_seeds={args.num_seeds}, "
        f"errors={args.error_std_values}, budgets={args.probe_budgets}"
    )
    print(
        f"Robust margins: gain={args.robust_gain_margins}, power={args.robust_power_margins}, "
        f"risk_weights={args.risk_weights}, risk_power={args.risk_power_weights}, "
        f"risk_thresholds={args.risk_invite_thresholds}, "
        f"adaptive_risk_base={args.adaptive_risk_base_weights}, "
        f"adaptive_error_ref={args.adaptive_risk_error_ref:g}, output_prefix={output_prefix}"
    )
    print("=" * 96)

    rows = []
    for error_std in args.error_std_values:
        print("=" * 96)
        print(f"Limited-CSI error std={error_std:g}")
        print("=" * 96)
        seed_result_sets = []
        for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
            print(f"Run seed [{run_idx}/{len(run_seeds)}]: {run_seeds[run_idx - 1]}")
            seed_result_sets.append(policy_suite_for_error(args, episode_seeds, error_std))
        rows.extend(summarize_results(args, aggregate_seed_results(seed_result_sets)))

    print_summary(rows)
    write_csv(f"{output_prefix}.csv", rows)
    if not args.no_plots:
        plot_results(rows, args, output_prefix)


if __name__ == "__main__":
    main()
