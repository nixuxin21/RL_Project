"""
Evaluate MS-AirComp when the execution channel differs from decision CSI.

Existing channel-estimation sweeps perturb only the decision preview while the
data slot still executes on the same true channel. This script models a stricter
setting: the policy decides from stale/estimated CSI, invites estimated-valid
nodes, and the actual AirComp slot succeeds only under a drifted execution
channel.

The execution oracle is kept only as an offline upper bound.
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evaluate_limited_csi_ms_aircomp as limited
from evaluate_policy_comparison import (
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
)


POLICY_EXECUTION_ORACLE = "Execution Oracle Full CSI"
POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID = "Execution-Risk Rotating Grid"
POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID = "Adaptive Execution-Risk Rotating Grid"
POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID = "Opportunity-Cost Execution-Risk Rotating Grid"
POLICY_AR1_PREDICT_ROTATING_GRID = "AR1-Predict Rotating Grid"
POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID = "Temporal-Reliability Rotating Grid"
POLICY_STALE_TOPK_FEEDBACK_GRID = "Stale-TopK Feedback Grid"
POLICY_TEMPORAL_DEVIATION_ORACLE_GRID = "Temporal Deviation Oracle"

MISMATCH_INDEPENDENT = "independent"
MISMATCH_TEMPORAL_AR1 = "temporal_ar1"
MISMATCH_CHOICES = {MISMATCH_INDEPENDENT, MISMATCH_TEMPORAL_AR1}

POLICY_CHOICES = {
    "no_irs": limited.POLICY_NO_IRS,
    "fixed": limited.POLICY_FIXED_IRS,
    "exact_greedy": limited.POLICY_EXACT_GREEDY,
    "estimated_greedy": limited.POLICY_EST_GREEDY,
    "random_probe": limited.POLICY_RANDOM_PROBE,
    "rotating": limited.POLICY_ROTATING_GRID,
    "robust_rotating": limited.POLICY_ROBUST_ROTATING_GRID,
    "risk_rotating": limited.POLICY_RISK_AWARE_ROTATING_GRID,
    "adaptive_risk_rotating": limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
    "execution_risk_rotating": POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
    "adaptive_execution_risk_rotating": POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
    "opportunity_execution_risk_rotating": POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID,
    "ar1_predict_rotating": POLICY_AR1_PREDICT_ROTATING_GRID,
    "temporal_reliability_rotating": POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID,
    "stale_topk_feedback": POLICY_STALE_TOPK_FEEDBACK_GRID,
    "stale_topk_rotating": POLICY_STALE_TOPK_FEEDBACK_GRID,
    "temporal_deviation_oracle": POLICY_TEMPORAL_DEVIATION_ORACLE_GRID,
    "execution_oracle": POLICY_EXECUTION_ORACLE,
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

CSV_FIELDS = [
    "decision_error_std",
    "execution_error_std",
    "mismatch_model",
    "channel_rho",
    "csi_delay_slots",
    "probe_budget",
    "policy",
    "episodes",
    "num_seeds",
    "num_nodes",
    "num_slots",
    "num_irs_elements",
    "num_codebook_states",
    "g_th",
    "alpha_th",
    "gain_margin",
    "power_margin",
    "risk_weight",
    "risk_power_weight",
    "risk_invite_threshold",
    "adaptive_risk_base_weight",
    "opportunity_failure_cost",
    "opportunity_missed_cost",
    "opportunity_deadline_gain",
    "opportunity_backlog_gain",
    "temporal_reliability_z",
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
    "missed_opportunities_mean",
    "true_opportunities_mean",
    "failure_slot_rate",
    "decision_preview_calls_per_slot_mean",
    "oracle_tx_gap_mean",
    "effective_risk_weight_mean",
    "avg_reward",
]


def parse_csv_items(value):
    """Parse comma-separated non-empty strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    """Parse execution mismatch arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate limited-CSI MS-AirComp with execution-stage channel mismatch."
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--probe-budgets", default="1,2,4")
    parser.add_argument("--mismatch-models", default="independent")
    parser.add_argument("--channel-rho-values", default="0.9")
    parser.add_argument("--csi-delay-slots", default="1")
    parser.add_argument("--decision-error-std-values", default="0.0")
    parser.add_argument("--execution-error-std-values", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--confirmation-feedback-noise-std", type=float, default=0.0)
    parser.add_argument("--confirmation-feedback-power-weight", type=float, default=0.05)
    parser.add_argument(
        "--policies",
        default="no_irs,fixed,execution_oracle,exact_greedy,estimated_greedy,rotating,robust_rotating,risk_rotating,adaptive_risk_rotating,execution_risk_rotating,adaptive_execution_risk_rotating,opportunity_execution_risk_rotating",
    )
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
    parser.add_argument("--opportunity-failure-costs", default="1.0")
    parser.add_argument("--opportunity-missed-costs", default="1.0")
    parser.add_argument("--opportunity-deadline-gains", default="0.5")
    parser.add_argument("--opportunity-backlog-gains", default="0.5")
    parser.add_argument("--temporal-reliability-z-values", default="1.0")
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
    """Validate arguments and parse list-like options."""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    for name in ("num_nodes", "num_slots", "num_irs_elements", "num_codebook_states"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.g_th <= 0.0:
        raise ValueError("--g-th must be positive")
    if args.alpha_th <= 0.0:
        raise ValueError("--alpha-th must be positive")

    args.probe_budgets = sorted(
        {min(int(value), args.num_codebook_states) for value in limited.parse_int_list(args.probe_budgets)}
    )
    if not args.probe_budgets or any(value <= 0 for value in args.probe_budgets):
        raise ValueError("--probe-budgets must contain positive integers")

    args.mismatch_models = parse_csv_items(args.mismatch_models)
    unknown_models = [name for name in args.mismatch_models if name not in MISMATCH_CHOICES]
    if not args.mismatch_models:
        raise ValueError("--mismatch-models must not be empty")
    if unknown_models:
        raise ValueError(f"Unknown mismatch models: {unknown_models}")
    args.channel_rho_values = limited.parse_float_list(args.channel_rho_values)
    args.csi_delay_slots = limited.parse_int_list(args.csi_delay_slots)
    if not args.channel_rho_values:
        raise ValueError("--channel-rho-values must not be empty")
    if not args.csi_delay_slots:
        raise ValueError("--csi-delay-slots must not be empty")
    if any(value < 0.0 or value > 1.0 for value in args.channel_rho_values):
        raise ValueError("--channel-rho-values must be in [0, 1]")
    if any(value < 0 for value in args.csi_delay_slots):
        raise ValueError("--csi-delay-slots must be non-negative")

    args.decision_error_std_values = limited.parse_float_list(args.decision_error_std_values)
    args.execution_error_std_values = limited.parse_float_list(args.execution_error_std_values)
    if not args.decision_error_std_values or not args.execution_error_std_values:
        raise ValueError("error std lists must not be empty")
    if any(value < 0.0 for value in args.decision_error_std_values + args.execution_error_std_values):
        raise ValueError("error std values must be non-negative")
    if args.confirmation_feedback_noise_std < 0.0:
        raise ValueError("--confirmation-feedback-noise-std must be non-negative")
    if args.confirmation_feedback_power_weight < 0.0:
        raise ValueError("--confirmation-feedback-power-weight must be non-negative")

    args.policies = parse_csv_items(args.policies)
    unknown_policies = [name for name in args.policies if name not in POLICY_CHOICES]
    if unknown_policies:
        raise ValueError(f"Unknown policies: {unknown_policies}")

    args.robust_gain_margins = limited.parse_float_list(args.robust_gain_margins)
    args.robust_power_margins = limited.parse_float_list(args.robust_power_margins)
    args.risk_weights = limited.parse_float_list(args.risk_weights)
    args.risk_power_weights = limited.parse_float_list(args.risk_power_weights)
    args.risk_invite_thresholds = limited.parse_float_list(args.risk_invite_thresholds)
    args.adaptive_risk_base_weights = limited.parse_float_list(args.adaptive_risk_base_weights)
    args.opportunity_failure_costs = limited.parse_float_list(args.opportunity_failure_costs)
    args.opportunity_missed_costs = limited.parse_float_list(args.opportunity_missed_costs)
    args.opportunity_deadline_gains = limited.parse_float_list(args.opportunity_deadline_gains)
    args.opportunity_backlog_gains = limited.parse_float_list(args.opportunity_backlog_gains)
    args.temporal_reliability_z_values = limited.parse_float_list(args.temporal_reliability_z_values)
    for name in (
        "robust_gain_margins",
        "robust_power_margins",
        "risk_weights",
        "risk_power_weights",
        "risk_invite_thresholds",
        "adaptive_risk_base_weights",
        "opportunity_failure_costs",
        "opportunity_missed_costs",
        "opportunity_deadline_gains",
        "opportunity_backlog_gains",
        "temporal_reliability_z_values",
    ):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} must not be empty")
    if any(value <= 0.0 for value in args.robust_gain_margins):
        raise ValueError("--robust-gain-margins must be positive")
    if any(value <= 0.0 for value in args.robust_power_margins):
        raise ValueError("--robust-power-margins must be positive")
    if any(value < 0.0 for value in args.risk_weights + args.risk_power_weights):
        raise ValueError("risk weights must be non-negative")
    if any(value < 0.0 or value > 1.0 for value in args.risk_invite_thresholds):
        raise ValueError("--risk-invite-thresholds must be in [0, 1]")
    if any(value < 0.0 for value in args.adaptive_risk_base_weights):
        raise ValueError("--adaptive-risk-base-weights must be non-negative")
    if any(value < 0.0 for value in args.opportunity_failure_costs):
        raise ValueError("--opportunity-failure-costs must be non-negative")
    if any(value < 0.0 for value in args.opportunity_missed_costs):
        raise ValueError("--opportunity-missed-costs must be non-negative")
    if any(value < 0.0 for value in args.opportunity_deadline_gains):
        raise ValueError("--opportunity-deadline-gains must be non-negative")
    if any(value < 0.0 for value in args.opportunity_backlog_gains):
        raise ValueError("--opportunity-backlog-gains must be non-negative")
    if any(value < 0.0 for value in args.temporal_reliability_z_values):
        raise ValueError("--temporal-reliability-z-values must be non-negative")

    args.fixed_irs_index = int(np.clip(args.fixed_irs_index, 0, args.num_codebook_states - 1))


def resolve_output_prefix(args):
    """Resolve output prefix for CSV and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    budget_label = "-".join(format_float_for_suffix(value) for value in args.probe_budgets)
    model_label = "-".join(args.mismatch_models)
    rho_label = "-".join(format_float_for_suffix(value) for value in args.channel_rho_values)
    delay_label = "-".join(str(value) for value in args.csi_delay_slots)
    decision_label = "-".join(format_float_for_suffix(value) for value in args.decision_error_std_values)
    execution_label = "-".join(format_float_for_suffix(value) for value in args.execution_error_std_values)
    suffix = (
        f"ep{args.episodes}_runs{args.num_seeds}_{seed_label}_b{budget_label}_"
        f"model{model_label}_rho{rho_label}_delay{delay_label}_"
        f"decerr{decision_label}_execerr{execution_label}"
    )
    output_prefix = os.path.join("results", "execution_mismatch", f"execution_mismatch_{suffix}")
    ensure_parent_dir(output_prefix)
    return output_prefix


def execution_rng(episode_seed, execution_error_std, slot_idx, no_irs=False):
    """Create a policy-independent RNG for execution channel drift."""
    if episode_seed is None:
        return np.random.default_rng()
    error_tag = int(round(float(execution_error_std) * 1_000_000))
    mode_tag = 0x4CF5AD43 if no_irs else 0
    seed = (
        int(episode_seed)
        + 0xD1B54A32
        + error_tag * 0x9E3779B1
        + int(slot_idx) * 0x85EBCA6B
        + mode_tag
    ) % (2**32)
    return np.random.default_rng(seed)


def drift_channels(h_total, execution_error_std, rng):
    """Apply per-slot execution drift to equivalent channels."""
    clean = np.asarray(h_total, dtype=np.complex128)
    if float(execution_error_std) <= 0.0:
        return clean.copy()
    rms = np.sqrt(np.mean(np.abs(clean) ** 2, axis=1, keepdims=True))
    noise = (rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)) / np.sqrt(2.0)
    return clean + float(execution_error_std) * np.maximum(rms, 1e-12) * noise


def capture_channel_state(env):
    """Copy the current physical channels from the environment."""
    return {
        "h_d": np.asarray(env.h_d, dtype=np.complex128).copy(),
        "h_r": np.asarray(env.h_r, dtype=np.complex128).copy(),
        "h_bs_r": np.asarray(env.h_bs_r, dtype=np.complex128).copy(),
    }


def apply_channel_state(env, state):
    """Apply a copied physical-channel state to the environment."""
    env.h_d = np.asarray(state["h_d"], dtype=np.complex128).copy()
    env.h_r = np.asarray(state["h_r"], dtype=np.complex128).copy()
    env.h_bs_r = np.asarray(state["h_bs_r"], dtype=np.complex128).copy()
    env.avg_large_scale = float(np.mean(np.abs(env.h_d) ** 2))


def temporal_rng(episode_seed, channel_rho):
    """Create a policy-independent RNG for temporal channel evolution."""
    if episode_seed is None:
        return np.random.default_rng()
    rho_tag = int(round(float(channel_rho) * 1_000_000))
    seed = (int(episode_seed) + 0xA511E9B3 + rho_tag * 0x9E3779B1) % (2**32)
    return np.random.default_rng(seed)


def complex_normal(rng, shape, scale=1.0):
    """Return circular complex Gaussian samples with the requested scale."""
    return float(scale) * (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)


def build_temporal_channel_states(env, args, episode_seed, channel_rho):
    """Generate one AR(1) physical-channel state per slot."""
    rho = float(channel_rho)
    innovation_weight = np.sqrt(max(0.0, 1.0 - rho**2))
    rng = temporal_rng(episode_seed, rho)
    states = [capture_channel_state(env)]
    for _slot_idx in range(1, int(args.num_slots)):
        prev = states[-1]
        states.append(
            {
                "h_d": rho * prev["h_d"]
                + innovation_weight * complex_normal(rng, prev["h_d"].shape, scale=0.1),
                "h_r": rho * prev["h_r"]
                + innovation_weight * complex_normal(rng, prev["h_r"].shape),
                "h_bs_r": rho * prev["h_bs_r"]
                + innovation_weight * complex_normal(rng, prev["h_bs_r"].shape),
            }
        )
    return states


def delayed_channel_state(states, slot_idx, csi_delay_slots):
    """Return the stale CSI state visible to the decision policy."""
    delayed_idx = max(0, int(slot_idx) - int(csi_delay_slots))
    delayed_idx = min(delayed_idx, len(states) - 1)
    return states[delayed_idx]


def ar1_predict_channel_state(delayed_state, channel_rho, csi_delay_slots):
    """Predict the current channel mean from delayed AR(1) CSI."""
    delay = max(int(csi_delay_slots), 0)
    rho = float(channel_rho)
    direct_factor = rho**delay
    return {
        "h_d": direct_factor * delayed_state["h_d"],
        "h_r": direct_factor * delayed_state["h_r"],
        "h_bs_r": direct_factor * delayed_state["h_bs_r"],
    }


def temporal_uncertainty_std(channel_rho, csi_delay_slots, use_ar1_prediction=False):
    """Return relative CSI uncertainty induced by temporal delay."""
    delay = max(int(csi_delay_slots), 0)
    if delay == 0:
        return 0.0
    rho_delay = float(channel_rho) ** delay
    if use_ar1_prediction:
        return float(np.sqrt(max(0.0, 1.0 - rho_delay**2)))
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - rho_delay))))


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
    # Reuse a known stable_rng policy offset; temporal reliability keeps its own salts below.
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


def execution_candidates(env, args, indices=None, execution_error_std=0.0, slot_idx=0, no_irs=False):
    """Build drifted execution candidates for selected indices or no-IRS."""
    rng = execution_rng(getattr(env, "_last_seed", None), execution_error_std, slot_idx, no_irs=no_irs)
    if no_irs:
        h_ref = limited.effective_channels(env, no_irs=True)
        h_exec = drift_channels(h_ref, execution_error_std, rng)
        return [limited.build_candidate(env, args, -2, h_exec[0], no_irs=True)]

    clean_indices = [int(np.clip(index, 0, args.num_codebook_states - 1)) for index in indices]
    h_ref = limited.effective_channels(env, clean_indices)
    h_exec = drift_channels(h_ref, execution_error_std, rng)
    return [
        limited.build_candidate(env, args, index, h_exec[row_idx])
        for row_idx, index in enumerate(clean_indices)
    ]


def execution_candidate_for_decision(env, args, decision_candidate, execution_error_std, slot_idx):
    """Return drifted execution candidate matching a decision."""
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
    """Return hidden oracle candidate under the drifted execution channel."""
    candidates = execution_candidates(
        env,
        args,
        indices=range(args.num_codebook_states),
        execution_error_std=execution_error_std,
        slot_idx=slot_idx,
    )
    return limited.best_candidate(candidates)


def choose_execution_oracle(env, args, execution_error_std, slot_idx):
    """Choose and invite using the hidden execution channel."""
    oracle = execution_oracle_candidate(env, args, execution_error_std, slot_idx)
    return oracle, args.num_codebook_states, args.num_codebook_states


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


def ordered_unique_prefix(indices, budget, num_codebook_states):
    """Return a clipped unique prefix preserving priority order."""
    selected = []
    seen = set()
    for raw_index in indices:
        index = int(np.clip(raw_index, 0, num_codebook_states - 1))
        if index in seen:
            continue
        selected.append(index)
        seen.add(index)
        if len(selected) >= int(budget):
            break
    return selected


def confirmation_feedback(candidate, args, feedback_noise_std, feedback_power_weight, rng):
    """Return noisy aggregate feedback for one current execution candidate."""
    observed_tx_fraction = float(candidate["tx_this_slot"]) / float(max(args.num_nodes, 1))
    if float(feedback_noise_std) > 0.0:
        observed_tx_fraction += float(rng.normal(0.0, float(feedback_noise_std)))
    observed_tx_fraction = float(np.clip(observed_tx_fraction, 0.0, 1.0))

    observed_power = float(candidate["power_avg"])
    observed_score = observed_tx_fraction - float(feedback_power_weight) * observed_power
    observed_score += float(rng.uniform(0.0, 1e-9))
    return {
        "irs_index": int(candidate["irs_index"]),
        "observed_tx_fraction": observed_tx_fraction,
        "observed_power": observed_power,
        "observed_score": float(observed_score),
    }


def choose_stale_topk_feedback_decision(
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
    Pick an active stale-CSI probe set, then confirm it with current feedback.

    The top half of the budget comes from a full stale-codebook ranking and the
    rest preserves rotating-grid coverage. The final IRS index is selected from
    B aggregate current-channel feedback probes, not from full execution CSI.
    """
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_decision(
            env,
            args,
            limited.POLICY_NO_IRS,
            0,
            slot_idx,
            decision_error_std,
            episode_seed,
        )

    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        args.num_codebook_states,
        salt=37 + slot_idx,
    )
    full_stale_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=range(args.num_codebook_states),
        error_std=decision_error_std,
        rng=error_rng,
    )
    ranked_indices = [
        int(candidate["irs_index"])
        for candidate in sorted(full_stale_candidates, key=limited.candidate_key, reverse=True)
    ]
    topk_budget = max(1, budget // 2)
    rotating_indices = limited.grid_indices(args.num_codebook_states, budget, offset=slot_idx)
    selected_indices = ordered_unique_prefix(
        ranked_indices[:topk_budget] + rotating_indices + ranked_indices,
        budget,
        args.num_codebook_states,
    )

    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in full_stale_candidates}
    decision_snapshot = capture_channel_state(env)
    if execution_state is not None:
        apply_channel_state(env, execution_state)
    current_candidates = [
        execution_candidates(
            env,
            args,
            indices=[index],
            execution_error_std=execution_error_std,
            slot_idx=slot_idx,
        )[0]
        for index in selected_indices
    ]
    feedback_rng = limited.stable_rng(
        episode_seed,
        args.confirmation_feedback_noise_std,
        limited.POLICY_RANDOM_PROBE,
        budget,
        salt=41 + slot_idx,
    )
    feedbacks = [
        confirmation_feedback(
            candidate,
            args,
            args.confirmation_feedback_noise_std,
            args.confirmation_feedback_power_weight,
            feedback_rng,
        )
        for candidate in current_candidates
    ]
    feedback_by_index = {int(feedback["irs_index"]): feedback for feedback in feedbacks}
    confirmed_index = max(
        selected_indices,
        key=lambda index: (
            float(feedback_by_index[index]["observed_score"]),
            float(feedback_by_index[index]["observed_tx_fraction"]),
            -float(feedback_by_index[index]["observed_power"]),
        ),
    )

    apply_channel_state(env, decision_snapshot)
    decision = candidate_by_index[int(confirmed_index)]
    decision["stale_topk_count"] = int(topk_budget)
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["stale_full_preview_count"] = int(args.num_codebook_states)
    decision["confirmation_feedback_count"] = int(len(selected_indices))
    return decision, args.num_codebook_states + len(selected_indices), len(selected_indices)


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


def choose_decision(
    env,
    args,
    policy_name,
    budget,
    slot_idx,
    decision_error_std,
    episode_seed,
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
    """Choose a limited-CSI decision from the stale/estimated decision channel."""
    choice_policy_name = policy_name
    if policy_name == POLICY_AR1_PREDICT_ROTATING_GRID:
        choice_policy_name = limited.POLICY_ROTATING_GRID
    return limited.choose_policy_candidate(
        env,
        args,
        choice_policy_name,
        budget,
        slot_idx,
        decision_error_std,
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


def policy_label(
    policy_name,
    budget=0,
    gain_margin=1.0,
    power_margin=1.0,
    risk_weight=0.0,
    risk_invite_threshold=0.0,
    opportunity_failure_cost=0.0,
    opportunity_missed_cost=0.0,
    opportunity_deadline_gain=0.0,
    opportunity_backlog_gain=0.0,
    temporal_reliability_z=0.0,
):
    """Return a compact display label."""
    if policy_name in {
        limited.POLICY_RANDOM_PROBE,
        limited.POLICY_ROTATING_GRID,
        limited.POLICY_ROBUST_ROTATING_GRID,
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
        POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
        POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
        POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID,
        POLICY_AR1_PREDICT_ROTATING_GRID,
        POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID,
        POLICY_STALE_TOPK_FEEDBACK_GRID,
        POLICY_TEMPORAL_DEVIATION_ORACLE_GRID,
    }:
        label = f"{policy_name} B={int(budget)}"
    else:
        label = policy_name
    if policy_name == limited.POLICY_ROBUST_ROTATING_GRID:
        label += f" gm={gain_margin:g} pm={power_margin:g}"
    if policy_name in {
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
        POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
        POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
    }:
        label += f" rw={risk_weight:g} rt={risk_invite_threshold:g}"
    if policy_name == POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID:
        label += (
            f" fc={opportunity_failure_cost:g} mc={opportunity_missed_cost:g}"
            f" dg={opportunity_deadline_gain:g} bg={opportunity_backlog_gain:g}"
        )
    if policy_name == POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID:
        label += f" rw={risk_weight:g} qz={temporal_reliability_z:g}"
    return label


def evaluate_policy(
    episode_seeds,
    args,
    decision_error_std,
    execution_error_std,
    mismatch_model,
    channel_rho,
    csi_delay_slots,
    policy_name,
    budget=0,
    gain_margin=1.0,
    power_margin=1.0,
    risk_weight=0.0,
    risk_power_weight=0.0,
    risk_invite_threshold=0.0,
    adaptive_risk_base_weight=0.0,
    opportunity_failure_cost=0.0,
    opportunity_missed_cost=0.0,
    opportunity_deadline_gain=0.0,
    opportunity_backlog_gain=0.0,
    temporal_reliability_z=0.0,
):
    """Evaluate one policy/config under decision and execution mismatch."""
    env = limited.make_env(args)
    display_name = policy_label(
        policy_name,
        budget=budget,
        gain_margin=gain_margin,
        power_margin=power_margin,
        risk_weight=risk_weight
        if policy_name
        not in {
            limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
            POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
        }
        else adaptive_risk_base_weight,
        risk_invite_threshold=risk_invite_threshold,
        opportunity_failure_cost=opportunity_failure_cost,
        opportunity_missed_cost=opportunity_missed_cost,
        opportunity_deadline_gain=opportunity_deadline_gain,
        opportunity_backlog_gain=opportunity_backlog_gain,
        temporal_reliability_z=temporal_reliability_z,
    )
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
        f"Running {display_name} model={mismatch_model} rho={channel_rho:g} "
        f"delay={int(csi_delay_slots)} decerr={decision_error_std:g} "
        f"execerr={execution_error_std:g}..."
    )
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        env._last_seed = episode_seed  # local metadata for policy-independent execution drift
        temporal_states = None
        if mismatch_model == MISMATCH_TEMPORAL_AR1:
            temporal_states = build_temporal_channel_states(env, args, episode_seed, channel_rho)
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
            if temporal_states is not None:
                execution_state = temporal_states[min(slot_idx, len(temporal_states) - 1)]
                stale_state = delayed_channel_state(temporal_states, slot_idx, csi_delay_slots)
                if policy_name == POLICY_AR1_PREDICT_ROTATING_GRID:
                    decision_state = ar1_predict_channel_state(stale_state, channel_rho, csi_delay_slots)
                    temporal_error_std = temporal_uncertainty_std(
                        channel_rho,
                        csi_delay_slots,
                        use_ar1_prediction=True,
                    )
                else:
                    decision_state = stale_state
                    temporal_error_std = temporal_uncertainty_std(
                        channel_rho,
                        csi_delay_slots,
                        use_ar1_prediction=False,
                    )
                apply_channel_state(env, execution_state)
            else:
                decision_state = None
                execution_state = None
                temporal_error_std = 0.0
            risk_execution_error_std = float(np.hypot(float(execution_error_std), temporal_error_std))

            execution_oracle = execution_oracle_candidate(env, args, execution_error_std, slot_idx)
            if policy_name == POLICY_EXECUTION_ORACLE:
                decision, preview_calls, _candidate_count = choose_execution_oracle(
                    env,
                    args,
                    execution_error_std,
                    slot_idx,
                )
                true_selected = decision
            else:
                effective_risk = risk_weight
                if policy_name == limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID:
                    effective_risk = adaptive_risk_base_weight
                if policy_name == POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID:
                    effective_risk = adaptive_risk_base_weight
                if decision_state is not None:
                    apply_channel_state(env, decision_state)
                if policy_name in {
                    POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
                    POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
                }:
                    decision, preview_calls, _candidate_count = choose_execution_risk_decision(
                        env,
                        args,
                        policy_name,
                        budget,
                        slot_idx,
                        decision_error_std,
                        risk_execution_error_std,
                        episode_seed,
                        risk_weight=effective_risk,
                        risk_power_weight=risk_power_weight,
                        risk_invite_threshold=risk_invite_threshold,
                        adaptive_risk_error_ref=args.adaptive_risk_error_ref,
                        adaptive_risk_error_gain=args.adaptive_risk_error_gain,
                        adaptive_risk_deadline_relief=args.adaptive_risk_deadline_relief,
                        adaptive_risk_backlog_relief=args.adaptive_risk_backlog_relief,
                    )
                elif policy_name == POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID:
                    decision, preview_calls, _candidate_count = choose_opportunity_execution_risk_decision(
                        env,
                        args,
                        budget,
                        slot_idx,
                        decision_error_std,
                        risk_execution_error_std,
                        episode_seed,
                        failure_cost=opportunity_failure_cost,
                        missed_cost=opportunity_missed_cost,
                        power_weight=risk_power_weight,
                        deadline_gain=opportunity_deadline_gain,
                        backlog_gain=opportunity_backlog_gain,
                    )
                elif policy_name == POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID:
                    decision, preview_calls, _candidate_count = choose_temporal_reliability_decision(
                        env,
                        args,
                        budget,
                        slot_idx,
                        decision_error_std,
                        risk_execution_error_std,
                        episode_seed,
                        risk_weight=effective_risk,
                        risk_power_weight=risk_power_weight,
                        quantile_z=temporal_reliability_z,
                    )
                elif policy_name == POLICY_STALE_TOPK_FEEDBACK_GRID:
                    decision, preview_calls, _candidate_count = choose_stale_topk_feedback_decision(
                        env,
                        args,
                        budget,
                        slot_idx,
                        decision_error_std,
                        execution_error_std,
                        episode_seed,
                        execution_state=execution_state,
                    )
                elif policy_name == POLICY_TEMPORAL_DEVIATION_ORACLE_GRID:
                    decision, preview_calls, _candidate_count = choose_temporal_deviation_oracle_decision(
                        env,
                        args,
                        budget,
                        slot_idx,
                        decision_error_std,
                        execution_error_std,
                        episode_seed,
                        execution_state=execution_state,
                    )
                else:
                    decision, preview_calls, _candidate_count = choose_decision(
                        env,
                        args,
                        policy_name,
                        budget,
                        slot_idx,
                        decision_error_std,
                        episode_seed,
                        gain_margin=gain_margin,
                        power_margin=power_margin,
                        risk_weight=effective_risk,
                        risk_power_weight=risk_power_weight,
                        risk_invite_threshold=risk_invite_threshold,
                        adaptive_risk_error_ref=args.adaptive_risk_error_ref,
                        adaptive_risk_error_gain=args.adaptive_risk_error_gain,
                        adaptive_risk_deadline_relief=args.adaptive_risk_deadline_relief,
                        adaptive_risk_backlog_relief=args.adaptive_risk_backlog_relief,
                    )
                if execution_state is not None:
                    apply_channel_state(env, execution_state)
                true_selected = execution_candidate_for_decision(
                    env,
                    args,
                    decision,
                    execution_error_std,
                    slot_idx,
                )

            info, done = limited.execute_limited_csi_slot(env, args, decision, true_selected)
            total_tx = int(info["total_tx"])
            episode_reward += float(info["reward"])
            episode_slots = int(info.get("slots_used", slot_idx + 1))
            episode_energy += float(info["attempted_energy"])
            episode_scheduled += int(info["scheduled_this_slot"])
            episode_failed += int(info["failed_this_slot"])
            episode_missed += int(info["missed_opportunity_this_slot"])
            episode_true_opportunities += int(info["true_opportunity_this_slot"])
            episode_failure_slots += int(info["failed_this_slot"] > 0)
            episode_preview_calls.append(int(preview_calls))
            episode_oracle_gaps.append(
                max(0.0, float(execution_oracle["tx_this_slot"]) - float(info["tx_this_slot"]))
            )
            episode_effective_risk_weights.append(float(decision.get("effective_risk_weight", 0.0)))
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
        failure_slots.append(float(episode_failure_slots) / max(float(episode_slots), 1.0))
        preview_calls_per_slot.append(
            float(sum(episode_preview_calls)) / max(float(len(episode_preview_calls)), 1.0)
        )
        oracle_tx_gap_mean.append(float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0)
        effective_risk_weights.append(
            float(np.mean(episode_effective_risk_weights)) if episode_effective_risk_weights else 0.0
        )

        print_progress(display_name, decision_error_std, execution_error_std, ep, args.episodes, success_nodes, args.num_nodes)

    return {
        "name": display_name,
        "policy": policy_name,
        "decision_error_std": float(decision_error_std),
        "execution_error_std": float(execution_error_std),
        "mismatch_model": mismatch_model,
        "channel_rho": float(channel_rho),
        "csi_delay_slots": int(csi_delay_slots),
        "probe_budget": int(budget),
        "gain_margin": float(gain_margin),
        "power_margin": float(power_margin),
        "risk_weight": float(risk_weight),
        "risk_power_weight": float(risk_power_weight),
        "risk_invite_threshold": float(risk_invite_threshold),
        "adaptive_risk_base_weight": float(adaptive_risk_base_weight),
        "opportunity_failure_cost": float(opportunity_failure_cost),
        "opportunity_missed_cost": float(opportunity_missed_cost),
        "opportunity_deadline_gain": float(opportunity_deadline_gain),
        "opportunity_backlog_gain": float(opportunity_backlog_gain),
        "temporal_reliability_z": float(temporal_reliability_z),
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


def print_progress(name, decision_error_std, execution_error_std, ep, episodes, success_nodes, num_nodes):
    """Print progress at 10 percent intervals."""
    interval = max(episodes // 10, 1)
    if ep % interval == 0 or ep == episodes:
        recent = np.mean(success_nodes[-interval:])
        print(
            f"  {name} decerr={decision_error_std:g} execerr={execution_error_std:g}: "
            f"[{ep:04d}/{episodes:04d}] recent success {recent:.2f}/{num_nodes}"
        )


def seed_summary(result):
    """Compress one run seed result into seed-level means."""
    return {key: float(np.mean(result[key])) for key in NUMERIC_RESULT_KEYS}


def aggregate_seed_results(seed_result_sets):
    """Aggregate matching result lists across run seeds."""
    if not seed_result_sets:
        return []
    aggregated_results = []
    result_count = len(seed_result_sets[0])
    for result_idx in range(result_count):
        parts = [seed_results[result_idx] for seed_results in seed_result_sets]
        aggregated = {
            "name": parts[0]["name"],
            "decision_error_std": parts[0]["decision_error_std"],
            "execution_error_std": parts[0]["execution_error_std"],
            "mismatch_model": parts[0]["mismatch_model"],
            "channel_rho": parts[0]["channel_rho"],
            "csi_delay_slots": parts[0]["csi_delay_slots"],
            "probe_budget": parts[0]["probe_budget"],
            "gain_margin": parts[0]["gain_margin"],
            "power_margin": parts[0]["power_margin"],
            "risk_weight": parts[0]["risk_weight"],
            "risk_power_weight": parts[0]["risk_power_weight"],
            "risk_invite_threshold": parts[0]["risk_invite_threshold"],
            "adaptive_risk_base_weight": parts[0]["adaptive_risk_base_weight"],
            "opportunity_failure_cost": parts[0]["opportunity_failure_cost"],
            "opportunity_missed_cost": parts[0]["opportunity_missed_cost"],
            "opportunity_deadline_gain": parts[0]["opportunity_deadline_gain"],
            "opportunity_backlog_gain": parts[0]["opportunity_backlog_gain"],
            "temporal_reliability_z": parts[0]["temporal_reliability_z"],
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


def summarize_results(args, results):
    """Convert aggregated results into CSV rows."""
    rows = []
    for result in results:
        success_mean, success_ci95 = metric_mean_ci(result, "success_nodes")
        slots_mean, slots_ci95 = metric_mean_ci(result, "slots_used")
        energy_mean, energy_ci95 = metric_mean_ci(result, "total_energy")
        rows.append(
            {
                "decision_error_std": float(result["decision_error_std"]),
                "execution_error_std": float(result["execution_error_std"]),
                "mismatch_model": result["mismatch_model"],
                "channel_rho": float(result["channel_rho"]),
                "csi_delay_slots": int(result["csi_delay_slots"]),
                "probe_budget": int(result["probe_budget"]),
                "policy": result["name"],
                "episodes": len(result["success_nodes"]),
                "num_seeds": args.num_seeds,
                "num_nodes": args.num_nodes,
                "num_slots": args.num_slots,
                "num_irs_elements": args.num_irs_elements,
                "num_codebook_states": args.num_codebook_states,
                "g_th": args.g_th,
                "alpha_th": args.alpha_th,
                "gain_margin": float(result["gain_margin"]),
                "power_margin": float(result["power_margin"]),
                "risk_weight": float(result["risk_weight"]),
                "risk_power_weight": float(result["risk_power_weight"]),
                "risk_invite_threshold": float(result["risk_invite_threshold"]),
                "adaptive_risk_base_weight": float(result["adaptive_risk_base_weight"]),
                "opportunity_failure_cost": float(result["opportunity_failure_cost"]),
                "opportunity_missed_cost": float(result["opportunity_missed_cost"]),
                "opportunity_deadline_gain": float(result["opportunity_deadline_gain"]),
                "opportunity_backlog_gain": float(result["opportunity_backlog_gain"]),
                "temporal_reliability_z": float(result["temporal_reliability_z"]),
                "success_mean": success_mean,
                "success_ci95": success_ci95,
                "success_rate_mean": success_mean / args.num_nodes,
                "perfect_rate": float(np.mean(result["success_nodes"] == args.num_nodes) * 100.0),
                "slots_mean": slots_mean,
                "slots_ci95": slots_ci95,
                "avg_power": float(np.mean(result["avg_power"])),
                "total_energy_mean": energy_mean,
                "total_energy_ci95": energy_ci95,
                "scheduled_nodes_mean": float(np.mean(result["scheduled_nodes"])),
                "failed_nodes_mean": float(np.mean(result["failed_nodes"])),
                "missed_opportunities_mean": float(np.mean(result["missed_opportunities"])),
                "true_opportunities_mean": float(np.mean(result["true_opportunities"])),
                "failure_slot_rate": float(np.mean(result["failure_slots"]) * 100.0),
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
    """Write execution mismatch summary CSV."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def print_summary(rows):
    """Print a compact execution mismatch summary."""
    print("=" * 184)
    print("Execution Channel Mismatch Summary")
    print("=" * 184)
    print(
        f"{'Model':>12} {'Rho':>5} {'Delay':>5} {'DecErr':>6} {'ExecErr':>7} "
        f"{'Policy':<44} {'Success':>9} {'Perfect%':>9} "
        f"{'Slots':>7} {'Fail':>8} {'MissOpp':>8} {'Preview':>8} {'Gap':>7}"
    )
    for row in rows:
        print(
            f"{row['mismatch_model']:>12} {row['channel_rho']:>5.2f} "
            f"{row['csi_delay_slots']:>5} {row['decision_error_std']:>6.3f} "
            f"{row['execution_error_std']:>7.3f} "
            f"{row['policy']:<44} {row['success_mean']:>9.3f} "
            f"{row['perfect_rate']:>8.2f}% {row['slots_mean']:>7.3f} "
            f"{row['failed_nodes_mean']:>8.2f} {row['missed_opportunities_mean']:>8.2f} "
            f"{row['decision_preview_calls_per_slot_mean']:>8.2f} "
            f"{row['oracle_tx_gap_mean']:>7.3f}"
        )


def plot_results(rows, args, output_prefix):
    """Plot success, failed invitations, and oracle gap vs execution error."""
    policies = []
    for row in rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    scenario_keys = sorted(
        {
            (
                row["mismatch_model"],
                row["channel_rho"],
                row["csi_delay_slots"],
                row["decision_error_std"],
            )
            for row in rows
        },
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    for mismatch_model, channel_rho, csi_delay_slots, decision_error_std in scenario_keys:
        subset = [
            row
            for row in rows
            if row["mismatch_model"] == mismatch_model
            and row["channel_rho"] == channel_rho
            and row["csi_delay_slots"] == csi_delay_slots
            and row["decision_error_std"] == decision_error_std
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cmap = plt.get_cmap("tab20")
        colors = {policy: cmap(idx % 20) for idx, policy in enumerate(policies)}
        for policy in policies:
            policy_rows = sorted(
                [row for row in subset if row["policy"] == policy],
                key=lambda row: row["execution_error_std"],
            )
            if not policy_rows:
                continue
            x = [row["execution_error_std"] for row in policy_rows]
            axes[0].plot(
                x,
                [row["success_mean"] for row in policy_rows],
                marker="o",
                linewidth=1.5,
                label=policy,
                color=colors[policy],
            )
            axes[1].plot(
                x,
                [row["failed_nodes_mean"] for row in policy_rows],
                marker="o",
                linewidth=1.5,
                label=policy,
                color=colors[policy],
            )
            axes[2].plot(
                x,
                [row["oracle_tx_gap_mean"] for row in policy_rows],
                marker="o",
                linewidth=1.5,
                label=policy,
                color=colors[policy],
            )

        scenario_title = (
            f"{mismatch_model}, rho={channel_rho:g}, delay={int(csi_delay_slots)}, "
            f"decision error={decision_error_std:g}"
        )
        axes[0].set_title(f"Success, {scenario_title}")
        axes[0].set_ylabel("Successful Nodes")
        axes[0].set_ylim(0.0, args.num_nodes + 1)
        axes[1].set_title("Failed Invited Nodes")
        axes[1].set_ylabel("Failed Nodes / Episode")
        axes[2].set_title("Per-Slot Execution Oracle Gap")
        axes[2].set_ylabel("Missed Tx Count")
        for ax in axes:
            ax.set_xlabel("Execution Channel Error Std")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(fontsize=7)
        fig.tight_layout()
        decision_suffix = format_float_for_suffix(decision_error_std)
        rho_suffix = format_float_for_suffix(channel_rho)
        path = (
            f"{output_prefix}_{mismatch_model}_rho{rho_suffix}_"
            f"delay{int(csi_delay_slots)}_decerr{decision_suffix}.png"
        )
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved: {path}")


def policy_configs(args):
    """Expand selected policy aliases into concrete parameter configurations."""
    configs = []
    for alias in args.policies:
        policy_name = POLICY_CHOICES[alias]
        if policy_name == POLICY_EXECUTION_ORACLE:
            configs.append({"policy_name": policy_name, "budget": args.num_codebook_states})
        elif policy_name in {
            limited.POLICY_NO_IRS,
            limited.POLICY_FIXED_IRS,
            limited.POLICY_EXACT_GREEDY,
            limited.POLICY_EST_GREEDY,
        }:
            budget = args.num_codebook_states if policy_name in {
                limited.POLICY_EXACT_GREEDY,
                limited.POLICY_EST_GREEDY,
            } else 0
            configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == limited.POLICY_ROBUST_ROTATING_GRID:
            for budget in args.probe_budgets:
                for gain_margin in args.robust_gain_margins:
                    for power_margin in args.robust_power_margins:
                        configs.append(
                            {
                                "policy_name": policy_name,
                                "budget": budget,
                                "gain_margin": gain_margin,
                                "power_margin": power_margin,
                            }
                        )
        elif policy_name in {
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
        }:
            for budget in args.probe_budgets:
                for risk_weight in args.risk_weights:
                    for risk_power_weight in args.risk_power_weights:
                        for risk_invite_threshold in args.risk_invite_thresholds:
                            configs.append(
                                {
                                    "policy_name": policy_name,
                                    "budget": budget,
                                    "risk_weight": risk_weight,
                                    "risk_power_weight": risk_power_weight,
                                    "risk_invite_threshold": risk_invite_threshold,
                                }
                            )
        elif policy_name in {
            limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
            POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
        }:
            for budget in args.probe_budgets:
                for adaptive_risk_base_weight in args.adaptive_risk_base_weights:
                    for risk_power_weight in args.risk_power_weights:
                        for risk_invite_threshold in args.risk_invite_thresholds:
                            configs.append(
                                {
                                    "policy_name": policy_name,
                                    "budget": budget,
                                    "risk_weight": adaptive_risk_base_weight,
                                    "adaptive_risk_base_weight": adaptive_risk_base_weight,
                                    "risk_power_weight": risk_power_weight,
                                    "risk_invite_threshold": risk_invite_threshold,
                                }
                            )
        elif policy_name == POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID:
            for budget in args.probe_budgets:
                for opportunity_failure_cost in args.opportunity_failure_costs:
                    for opportunity_missed_cost in args.opportunity_missed_costs:
                        for opportunity_deadline_gain in args.opportunity_deadline_gains:
                            for opportunity_backlog_gain in args.opportunity_backlog_gains:
                                for risk_power_weight in args.risk_power_weights:
                                    configs.append(
                                        {
                                            "policy_name": policy_name,
                                            "budget": budget,
                                            "risk_power_weight": risk_power_weight,
                                            "opportunity_failure_cost": opportunity_failure_cost,
                                            "opportunity_missed_cost": opportunity_missed_cost,
                                            "opportunity_deadline_gain": opportunity_deadline_gain,
                                            "opportunity_backlog_gain": opportunity_backlog_gain,
                                        }
                                    )
        elif policy_name == POLICY_AR1_PREDICT_ROTATING_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_STALE_TOPK_FEEDBACK_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_TEMPORAL_DEVIATION_ORACLE_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID:
            for budget in args.probe_budgets:
                for risk_weight in args.risk_weights:
                    for risk_power_weight in args.risk_power_weights:
                        for temporal_reliability_z in args.temporal_reliability_z_values:
                            configs.append(
                                {
                                    "policy_name": policy_name,
                                    "budget": budget,
                                    "risk_weight": risk_weight,
                                    "risk_power_weight": risk_power_weight,
                                    "temporal_reliability_z": temporal_reliability_z,
                                }
                            )
        else:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
    return configs


def mismatch_scenarios(args):
    """Yield concrete mismatch-model parameter tuples."""
    for mismatch_model in args.mismatch_models:
        if mismatch_model == MISMATCH_INDEPENDENT:
            yield mismatch_model, 0.0, 0
            continue
        for channel_rho in args.channel_rho_values:
            for csi_delay_slots in args.csi_delay_slots:
                yield mismatch_model, float(channel_rho), int(csi_delay_slots)


def main():
    """Run execution-stage channel mismatch evaluation."""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)
    run_seeds = make_run_seeds(args)
    episode_seed_sets = [make_episode_seeds(args, run_seed) for run_seed in run_seeds]
    configs = policy_configs(args)

    print("=" * 96)
    print(
        f"Execution channel mismatch: episodes={args.episodes}, num_seeds={args.num_seeds}, "
        f"decision_errors={args.decision_error_std_values}, execution_errors={args.execution_error_std_values}, "
        f"models={args.mismatch_models}, rhos={args.channel_rho_values}, "
        f"delays={args.csi_delay_slots}, budgets={args.probe_budgets}"
    )
    print(f"Output prefix: {output_prefix}")
    print("=" * 96)

    all_rows = []
    for mismatch_model, channel_rho, csi_delay_slots in mismatch_scenarios(args):
        for decision_error_std in args.decision_error_std_values:
            for execution_error_std in args.execution_error_std_values:
                print("=" * 96)
                print(
                    f"Mismatch={mismatch_model}, rho={channel_rho:g}, "
                    f"delay={int(csi_delay_slots)}, decision error std={decision_error_std:g}, "
                    f"execution error std={execution_error_std:g}"
                )
                print("=" * 96)
                seed_result_sets = []
                for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
                    print(f"Run seed [{run_idx}/{len(run_seeds)}]: {run_seeds[run_idx - 1]}")
                    seed_results = [
                        evaluate_policy(
                            episode_seeds,
                            args,
                            decision_error_std,
                            execution_error_std,
                            mismatch_model,
                            channel_rho,
                            csi_delay_slots,
                            **config,
                        )
                        for config in configs
                    ]
                    seed_result_sets.append(seed_results)
                all_rows.extend(summarize_results(args, aggregate_seed_results(seed_result_sets)))

    print_summary(all_rows)
    write_csv(f"{output_prefix}.csv", all_rows)
    if not args.no_plots:
        plot_results(all_rows, args, output_prefix)


if __name__ == "__main__":
    main()
