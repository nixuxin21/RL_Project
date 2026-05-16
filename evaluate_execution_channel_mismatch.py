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
import os
import numpy as np

import evaluate_limited_csi_ms_aircomp as limited
from evaluate_policy_comparison import make_episode_seeds, make_run_seeds
from ms_aircomp.channel_models import (
    apply_channel_state,
    ar1_predict_channel_state,
    build_temporal_channel_states,
    delayed_channel_state,
    temporal_uncertainty_std,
)
from ms_aircomp.execution_decision_dispatch import (
    choose_decision,
    choose_execution_mismatch_decision,
)
from ms_aircomp.execution_candidates import execution_oracle_candidate
from ms_aircomp.execution_output import (
    plot_results,
    print_progress,
    print_summary,
    resolve_output_prefix,
)
from ms_aircomp.execution_policy_registry import (
    MISMATCH_CHOICES,
    MISMATCH_INDEPENDENT,
    MISMATCH_TEMPORAL_AR1,
    POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
    POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
    POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
    POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
    POLICY_ACTIVE_DIVERSE_FEEDBACK_GRID,
    POLICY_AR1_PREDICT_ROTATING_GRID,
    POLICY_CHOICES,
    POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
    POLICY_EXECUTION_ORACLE,
    POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
    POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
    POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
    POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
    POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID,
    POLICY_ROTATING_FEEDBACK_CONFIRM_GRID,
    POLICY_SPARSE_TOPK_FEEDBACK_GRID,
    POLICY_STALE_TOPK_FEEDBACK_GRID,
    POLICY_TEMPORAL_DEVIATION_ORACLE_GRID,
    POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID,
    mismatch_scenarios,
    policy_configs,
    policy_label,
)
from ms_aircomp.execution_result_summary import (
    CSV_FIELDS,
    NUMERIC_RESULT_KEYS,
    OPTIONAL_RESULT_DEFAULTS,
    aggregate_seed_results,
    metric_mean_ci,
    result_array,
    result_value,
    seed_summary,
    summarize_results,
    write_csv,
)
from ms_aircomp.learned_shortlist import (
    load_learned_set_shortlist_model,
    load_learned_shortlist_model,
)


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
    parser.add_argument(
        "--sparse-topk-seed-multipliers",
        default="2",
        help="Comma-separated multipliers for sparse stale preview pool size; seed count is ceil(multiplier * B).",
    )
    parser.add_argument(
        "--sparse-topk-fractions",
        default="0.75",
        help="Comma-separated fractions of B reserved for sparse stale top candidates before rotating coverage.",
    )
    parser.add_argument(
        "--coverage-sparse-weights",
        default="0.5",
        help=(
            "Comma-separated marginal device-coverage weights for coverage-aware sparse top-k. "
            "Higher values favor stale candidates that cover devices not covered by already selected candidates."
        ),
    )
    parser.add_argument(
        "--coverage-sparse-power-weight",
        type=float,
        default=0.0,
        help="Power penalty used while selecting coverage-aware sparse candidates from stale previews.",
    )
    parser.add_argument(
        "--coverage-sparse-power-weights",
        default=None,
        help=(
            "Optional comma-separated power penalties for coverage-aware sparse top-k. "
            "When omitted, --coverage-sparse-power-weight is used for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--adaptive-sparse-margin-thresholds",
        default="0.05",
        help=(
            "Comma-separated normalized stale top-k margin thresholds for adaptive sparse top-k. "
            "Expand from base to expanded seed multiplier when the top-vs-kth stale tx-count margin is below threshold."
        ),
    )
    parser.add_argument(
        "--adaptive-sparse-base-multiplier",
        type=float,
        default=2.0,
        help="Base stale preview multiplier for adaptive sparse top-k.",
    )
    parser.add_argument(
        "--adaptive-sparse-expanded-multiplier",
        type=float,
        default=3.0,
        help="Expanded stale preview multiplier for adaptive sparse top-k.",
    )
    parser.add_argument(
        "--adaptive-sparse-v2-preview-costs",
        default="0.002",
        help=(
            "Comma-separated expansion cost penalties for adaptive sparse top-k v2. "
            "Each value is a normalized margin penalty per extra stale preview."
        ),
    )
    parser.add_argument(
        "--adaptive-sparse-v2-uncertainty-weight",
        type=float,
        default=0.02,
        help="Margin-threshold boost per unit stale rho/delay uncertainty for adaptive sparse top-k v2.",
    )
    parser.add_argument(
        "--adaptive-sparse-v2-urgency-weight",
        type=float,
        default=0.5,
        help="Margin-threshold boost for deadline/backlog shortfall in adaptive sparse top-k v2.",
    )
    parser.add_argument(
        "--adaptive-sparse-v2-history-weight",
        type=float,
        default=0.02,
        help="Margin-threshold reduction when recent confirmed IRS choices are stable in adaptive sparse top-k v2.",
    )
    parser.add_argument(
        "--adaptive-sparse-history-window",
        type=int,
        default=3,
        help="Number of recent confirmed IRS choices used by adaptive sparse top-k v2.",
    )
    parser.add_argument(
        "--adaptive-sparse-history-prior-threshold",
        type=float,
        default=0.67,
        help="Minimum recent winner frequency before adaptive sparse top-k v2 injects a history prior.",
    )
    parser.add_argument(
        "--adaptive-sparse-v3-neighbor-radius",
        type=int,
        default=1,
        help="Codebook-neighborhood radius around stale top candidates for adaptive sparse top-k v3.",
    )
    parser.add_argument(
        "--adaptive-sparse-v3-neighbor-count",
        type=int,
        default=2,
        help="Maximum neighbor IRS indices to consider before rotating coverage in adaptive sparse top-k v3.",
    )
    parser.add_argument(
        "--adaptive-sparse-v3-history-count",
        type=int,
        default=1,
        help="Maximum recent stable winner indices to consider before rotating coverage in adaptive sparse top-k v3.",
    )
    parser.add_argument(
        "--learned-shortlist-model",
        default="",
        help="Path to a .npz linear ranker for learned sparse shortlist feedback.",
    )
    parser.add_argument(
        "--learned-shortlist-extra-counts",
        default="1",
        help="Comma-separated numbers of learned extra stale candidates to add after the 2B sparse pool.",
    )
    parser.add_argument(
        "--learned-set-shortlist-model",
        default="",
        help="Path to a .npz set-level ranker for learned set shortlist feedback.",
    )
    parser.add_argument(
        "--learned-set-extra-counts",
        default="1",
        help="Comma-separated maximum numbers of non-seed candidates considered by learned set shortlist feedback.",
    )
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
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")
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
    args.sparse_topk_seed_multipliers = limited.parse_float_list(args.sparse_topk_seed_multipliers)
    args.sparse_topk_fractions = limited.parse_float_list(args.sparse_topk_fractions)
    args.coverage_sparse_weights = limited.parse_float_list(args.coverage_sparse_weights)
    if args.coverage_sparse_power_weights is None:
        args.coverage_sparse_power_weights = [float(args.coverage_sparse_power_weight)]
    else:
        args.coverage_sparse_power_weights = limited.parse_float_list(
            args.coverage_sparse_power_weights
        )
    args.adaptive_sparse_margin_thresholds = limited.parse_float_list(args.adaptive_sparse_margin_thresholds)
    args.adaptive_sparse_v2_preview_costs = limited.parse_float_list(args.adaptive_sparse_v2_preview_costs)
    args.learned_shortlist_extra_counts = limited.parse_int_list(args.learned_shortlist_extra_counts)
    args.learned_set_extra_counts = limited.parse_int_list(args.learned_set_extra_counts)
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
        "sparse_topk_seed_multipliers",
        "sparse_topk_fractions",
        "coverage_sparse_weights",
        "coverage_sparse_power_weights",
        "adaptive_sparse_margin_thresholds",
        "adaptive_sparse_v2_preview_costs",
        "learned_shortlist_extra_counts",
        "learned_set_extra_counts",
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
    if any(value <= 0.0 for value in args.sparse_topk_seed_multipliers):
        raise ValueError("--sparse-topk-seed-multipliers must be positive")
    if any(value <= 0.0 or value > 1.0 for value in args.sparse_topk_fractions):
        raise ValueError("--sparse-topk-fractions must be in (0, 1]")
    if any(value < 0.0 for value in args.coverage_sparse_weights):
        raise ValueError("--coverage-sparse-weights must be non-negative")
    if any(value < 0.0 for value in args.coverage_sparse_power_weights):
        raise ValueError("--coverage-sparse-power-weights must be non-negative")
    if any(value < 0.0 for value in args.adaptive_sparse_margin_thresholds):
        raise ValueError("--adaptive-sparse-margin-thresholds must be non-negative")
    if args.adaptive_sparse_base_multiplier <= 0.0:
        raise ValueError("--adaptive-sparse-base-multiplier must be positive")
    if args.adaptive_sparse_expanded_multiplier < args.adaptive_sparse_base_multiplier:
        raise ValueError("--adaptive-sparse-expanded-multiplier must be at least the base multiplier")
    if any(value < 0.0 for value in args.adaptive_sparse_v2_preview_costs):
        raise ValueError("--adaptive-sparse-v2-preview-costs must be non-negative")
    if args.adaptive_sparse_v2_uncertainty_weight < 0.0:
        raise ValueError("--adaptive-sparse-v2-uncertainty-weight must be non-negative")
    if args.adaptive_sparse_v2_urgency_weight < 0.0:
        raise ValueError("--adaptive-sparse-v2-urgency-weight must be non-negative")
    if args.adaptive_sparse_v2_history_weight < 0.0:
        raise ValueError("--adaptive-sparse-v2-history-weight must be non-negative")
    if args.adaptive_sparse_history_window <= 0:
        raise ValueError("--adaptive-sparse-history-window must be positive")
    if args.adaptive_sparse_history_prior_threshold < 0.0 or args.adaptive_sparse_history_prior_threshold > 1.0:
        raise ValueError("--adaptive-sparse-history-prior-threshold must be in [0, 1]")
    if args.adaptive_sparse_v3_neighbor_radius < 0:
        raise ValueError("--adaptive-sparse-v3-neighbor-radius must be non-negative")
    if args.adaptive_sparse_v3_neighbor_count < 0:
        raise ValueError("--adaptive-sparse-v3-neighbor-count must be non-negative")
    if args.adaptive_sparse_v3_history_count < 0:
        raise ValueError("--adaptive-sparse-v3-history-count must be non-negative")
    if any(value < 0 for value in args.learned_shortlist_extra_counts):
        raise ValueError("--learned-shortlist-extra-counts must be non-negative")
    if any(value < 0 for value in args.learned_set_extra_counts):
        raise ValueError("--learned-set-extra-counts must be non-negative")
    if any(POLICY_CHOICES.get(alias) == POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID for alias in args.policies):
        if not args.learned_shortlist_model:
            raise ValueError("--learned-shortlist-model is required for learned shortlist feedback")
        if not os.path.exists(args.learned_shortlist_model):
            raise ValueError(f"Learned shortlist model not found: {args.learned_shortlist_model}")
    if any(POLICY_CHOICES.get(alias) == POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID for alias in args.policies):
        if not args.learned_set_shortlist_model:
            raise ValueError("--learned-set-shortlist-model is required for learned set shortlist feedback")
        if not os.path.exists(args.learned_set_shortlist_model):
            raise ValueError(f"Learned set shortlist model not found: {args.learned_set_shortlist_model}")

    args.fixed_irs_index = int(np.clip(args.fixed_irs_index, 0, args.num_codebook_states - 1))
    args.learned_shortlist_model_data = (
        load_learned_shortlist_model(args.learned_shortlist_model)
        if getattr(args, "learned_shortlist_model", "")
        else None
    )
    args.learned_set_shortlist_model_data = (
        load_learned_set_shortlist_model(args.learned_set_shortlist_model)
        if getattr(args, "learned_set_shortlist_model", "")
        else None
    )


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
    sparse_topk_seed_multiplier=2.0,
    sparse_topk_fraction=0.75,
    sparse_topk_show_params=False,
    coverage_sparse_weight=0.5,
    coverage_sparse_power_weight=0.0,
    adaptive_sparse_base_multiplier=2.0,
    adaptive_sparse_expanded_multiplier=3.0,
    adaptive_sparse_margin_threshold=0.05,
    adaptive_sparse_v2_preview_cost=0.002,
    adaptive_sparse_v3_neighbor_radius=1,
    adaptive_sparse_v3_neighbor_count=2,
    adaptive_sparse_v3_history_count=1,
    learned_shortlist_extra_count=1,
    adaptive_sparse_show_params=False,
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
        sparse_topk_seed_multiplier=sparse_topk_seed_multiplier,
        sparse_topk_fraction=sparse_topk_fraction,
        sparse_topk_show_params=sparse_topk_show_params,
        coverage_sparse_weight=coverage_sparse_weight,
        coverage_sparse_power_weight=coverage_sparse_power_weight,
        adaptive_sparse_base_multiplier=adaptive_sparse_base_multiplier,
        adaptive_sparse_expanded_multiplier=adaptive_sparse_expanded_multiplier,
        adaptive_sparse_margin_threshold=adaptive_sparse_margin_threshold,
        adaptive_sparse_v2_preview_cost=adaptive_sparse_v2_preview_cost,
        adaptive_sparse_v3_neighbor_radius=adaptive_sparse_v3_neighbor_radius,
        adaptive_sparse_v3_neighbor_count=adaptive_sparse_v3_neighbor_count,
        adaptive_sparse_v3_history_count=adaptive_sparse_v3_history_count,
        learned_shortlist_extra_count=learned_shortlist_extra_count,
        adaptive_sparse_show_params=adaptive_sparse_show_params,
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
    adaptive_sparse_expanded = []
    adaptive_sparse_margins = []
    adaptive_sparse_effective_thresholds = []
    adaptive_sparse_expand_scores = []
    adaptive_sparse_history_stabilities = []
    adaptive_sparse_urgencies = []
    adaptive_sparse_cost_penalties = []
    adaptive_sparse_v3_history_prior_used = []
    adaptive_sparse_v3_neighbor_extra_preview_counts = []
    adaptive_sparse_v3_selected_extra_preview_counts = []
    learned_shortlist_selected_extra_preview_counts = []
    coverage_sparse_selected_marginal_fractions = []
    coverage_sparse_selected_overlap_fractions = []

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
        episode_adaptive_sparse_expanded = []
        episode_adaptive_sparse_margins = []
        episode_adaptive_sparse_effective_thresholds = []
        episode_adaptive_sparse_expand_scores = []
        episode_adaptive_sparse_history_stabilities = []
        episode_adaptive_sparse_urgencies = []
        episode_adaptive_sparse_cost_penalties = []
        episode_adaptive_sparse_v3_history_prior_used = []
        episode_adaptive_sparse_v3_neighbor_extra_preview_counts = []
        episode_adaptive_sparse_v3_selected_extra_preview_counts = []
        episode_learned_shortlist_selected_extra_preview_counts = []
        episode_coverage_sparse_selected_marginal_fractions = []
        episode_coverage_sparse_selected_overlap_fractions = []
        episode_confirmed_irs_history = []
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
            decision, true_selected, preview_calls, _candidate_count = choose_execution_mismatch_decision(
                env,
                args,
                policy_name,
                budget,
                slot_idx,
                decision_error_std,
                execution_error_std,
                risk_execution_error_std,
                episode_seed,
                decision_state=decision_state,
                execution_state=execution_state,
                episode_confirmed_irs_history=episode_confirmed_irs_history,
                channel_rho=channel_rho,
                csi_delay_slots=csi_delay_slots,
                gain_margin=gain_margin,
                power_margin=power_margin,
                risk_weight=risk_weight,
                risk_power_weight=risk_power_weight,
                risk_invite_threshold=risk_invite_threshold,
                adaptive_risk_base_weight=adaptive_risk_base_weight,
                opportunity_failure_cost=opportunity_failure_cost,
                opportunity_missed_cost=opportunity_missed_cost,
                opportunity_deadline_gain=opportunity_deadline_gain,
                opportunity_backlog_gain=opportunity_backlog_gain,
                temporal_reliability_z=temporal_reliability_z,
                sparse_topk_seed_multiplier=sparse_topk_seed_multiplier,
                sparse_topk_fraction=sparse_topk_fraction,
                coverage_sparse_weight=coverage_sparse_weight,
                coverage_sparse_power_weight=coverage_sparse_power_weight,
                adaptive_sparse_base_multiplier=adaptive_sparse_base_multiplier,
                adaptive_sparse_expanded_multiplier=adaptive_sparse_expanded_multiplier,
                adaptive_sparse_margin_threshold=adaptive_sparse_margin_threshold,
                adaptive_sparse_v2_preview_cost=adaptive_sparse_v2_preview_cost,
                adaptive_sparse_v3_neighbor_radius=adaptive_sparse_v3_neighbor_radius,
                adaptive_sparse_v3_neighbor_count=adaptive_sparse_v3_neighbor_count,
                adaptive_sparse_v3_history_count=adaptive_sparse_v3_history_count,
                learned_shortlist_extra_count=learned_shortlist_extra_count,
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
            episode_adaptive_sparse_expanded.append(float(decision.get("adaptive_sparse_expanded", 0.0)))
            episode_adaptive_sparse_margins.append(float(decision.get("adaptive_sparse_margin", 0.0)))
            episode_adaptive_sparse_effective_thresholds.append(
                float(decision.get("adaptive_sparse_effective_margin_threshold", 0.0))
            )
            episode_adaptive_sparse_expand_scores.append(
                float(decision.get("adaptive_sparse_expand_score", 0.0))
            )
            episode_adaptive_sparse_history_stabilities.append(
                float(decision.get("adaptive_sparse_history_stability", 0.0))
            )
            episode_adaptive_sparse_urgencies.append(float(decision.get("adaptive_sparse_urgency", 0.0)))
            episode_adaptive_sparse_cost_penalties.append(
                float(decision.get("adaptive_sparse_cost_penalty", 0.0))
            )
            episode_adaptive_sparse_v3_history_prior_used.append(
                float(decision.get("adaptive_sparse_v3_history_prior_used", 0.0))
            )
            episode_adaptive_sparse_v3_neighbor_extra_preview_counts.append(
                float(decision.get("adaptive_sparse_v3_neighbor_extra_preview_count", 0.0))
            )
            episode_adaptive_sparse_v3_selected_extra_preview_counts.append(
                float(decision.get("adaptive_sparse_v3_selected_extra_preview_count", 0.0))
            )
            episode_learned_shortlist_selected_extra_preview_counts.append(
                float(decision.get("learned_shortlist_selected_extra_preview_count", 0.0))
            )
            episode_coverage_sparse_selected_marginal_fractions.append(
                float(decision.get("coverage_sparse_selected_marginal_fraction", 0.0))
            )
            episode_coverage_sparse_selected_overlap_fractions.append(
                float(decision.get("coverage_sparse_selected_overlap_fraction", 0.0))
            )
            if int(decision.get("confirmed_irs_index", decision.get("irs_index", -1))) >= 0:
                episode_confirmed_irs_history.append(
                    int(decision.get("confirmed_irs_index", decision.get("irs_index", -1)))
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
        failure_slots.append(float(episode_failure_slots) / max(float(episode_slots), 1.0))
        preview_calls_per_slot.append(
            float(sum(episode_preview_calls)) / max(float(len(episode_preview_calls)), 1.0)
        )
        oracle_tx_gap_mean.append(float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0)
        effective_risk_weights.append(
            float(np.mean(episode_effective_risk_weights)) if episode_effective_risk_weights else 0.0
        )
        adaptive_sparse_expanded.append(
            float(np.mean(episode_adaptive_sparse_expanded)) if episode_adaptive_sparse_expanded else 0.0
        )
        adaptive_sparse_margins.append(
            float(np.mean(episode_adaptive_sparse_margins)) if episode_adaptive_sparse_margins else 0.0
        )
        adaptive_sparse_effective_thresholds.append(
            float(np.mean(episode_adaptive_sparse_effective_thresholds))
            if episode_adaptive_sparse_effective_thresholds
            else 0.0
        )
        adaptive_sparse_expand_scores.append(
            float(np.mean(episode_adaptive_sparse_expand_scores))
            if episode_adaptive_sparse_expand_scores
            else 0.0
        )
        adaptive_sparse_history_stabilities.append(
            float(np.mean(episode_adaptive_sparse_history_stabilities))
            if episode_adaptive_sparse_history_stabilities
            else 0.0
        )
        adaptive_sparse_urgencies.append(
            float(np.mean(episode_adaptive_sparse_urgencies)) if episode_adaptive_sparse_urgencies else 0.0
        )
        adaptive_sparse_cost_penalties.append(
            float(np.mean(episode_adaptive_sparse_cost_penalties))
            if episode_adaptive_sparse_cost_penalties
            else 0.0
        )
        adaptive_sparse_v3_history_prior_used.append(
            float(np.mean(episode_adaptive_sparse_v3_history_prior_used))
            if episode_adaptive_sparse_v3_history_prior_used
            else 0.0
        )
        adaptive_sparse_v3_neighbor_extra_preview_counts.append(
            float(np.mean(episode_adaptive_sparse_v3_neighbor_extra_preview_counts))
            if episode_adaptive_sparse_v3_neighbor_extra_preview_counts
            else 0.0
        )
        adaptive_sparse_v3_selected_extra_preview_counts.append(
            float(np.mean(episode_adaptive_sparse_v3_selected_extra_preview_counts))
            if episode_adaptive_sparse_v3_selected_extra_preview_counts
            else 0.0
        )
        learned_shortlist_selected_extra_preview_counts.append(
            float(np.mean(episode_learned_shortlist_selected_extra_preview_counts))
            if episode_learned_shortlist_selected_extra_preview_counts
            else 0.0
        )
        coverage_sparse_selected_marginal_fractions.append(
            float(np.mean(episode_coverage_sparse_selected_marginal_fractions))
            if episode_coverage_sparse_selected_marginal_fractions
            else 0.0
        )
        coverage_sparse_selected_overlap_fractions.append(
            float(np.mean(episode_coverage_sparse_selected_overlap_fractions))
            if episode_coverage_sparse_selected_overlap_fractions
            else 0.0
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
        "sparse_topk_seed_multiplier": float(
            sparse_topk_seed_multiplier
            if policy_name
            in {
                POLICY_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
            }
            else 0.0
        ),
        "sparse_topk_fraction": float(
            sparse_topk_fraction
            if policy_name
            in {
                POLICY_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
                POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
                POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
                POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
            }
            else 0.0
        ),
        "coverage_sparse_weight": float(
            coverage_sparse_weight
            if policy_name
            in {
                POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
            }
            else 0.0
        ),
        "coverage_sparse_power_weight": float(
            coverage_sparse_power_weight
            if policy_name
            in {
                POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
            }
            else 0.0
        ),
        "adaptive_sparse_base_multiplier": float(
            adaptive_sparse_base_multiplier
            if policy_name
            in {
                POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
                POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
                POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
                POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
            }
            else 0.0
        ),
        "adaptive_sparse_expanded_multiplier": float(
            adaptive_sparse_base_multiplier
            if policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID
            else adaptive_sparse_base_multiplier
            if policy_name in {
                POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
                POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
            }
            else adaptive_sparse_expanded_multiplier
            if policy_name
            in {
                POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
            }
            else 0.0
        ),
        "adaptive_sparse_margin_threshold": float(
            adaptive_sparse_margin_threshold
            if policy_name
            in {
                POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
                POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
            }
            else 0.0
        ),
        "adaptive_sparse_v2_preview_cost": float(
            adaptive_sparse_v2_preview_cost
            if policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID
            else 0.0
        ),
        "adaptive_sparse_v2_uncertainty_weight": float(
            args.adaptive_sparse_v2_uncertainty_weight
            if policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID
            else 0.0
        ),
        "adaptive_sparse_v2_urgency_weight": float(
            args.adaptive_sparse_v2_urgency_weight
            if policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID
            else 0.0
        ),
        "adaptive_sparse_v2_history_weight": float(
            args.adaptive_sparse_v2_history_weight
            if policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID
            else 0.0
        ),
        "adaptive_sparse_history_window": int(
            args.adaptive_sparse_history_window
            if policy_name
            in {
                POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
                POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
                POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
                POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
            }
            else 0
        ),
        "adaptive_sparse_v3_neighbor_radius": int(
            adaptive_sparse_v3_neighbor_radius
            if policy_name
            in {
                POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
                POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
            }
            else 0
        ),
        "adaptive_sparse_v3_neighbor_count": int(
            adaptive_sparse_v3_neighbor_count
            if policy_name
            in {
                POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
                POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
            }
            else 0
        ),
        "adaptive_sparse_v3_history_count": int(
            adaptive_sparse_v3_history_count
            if policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID
            else 0
        ),
        "learned_shortlist_extra_count": int(
            learned_shortlist_extra_count
            if policy_name
            in {
                POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
                POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
            }
            else 0
        ),
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
        "adaptive_sparse_expanded": np.asarray(adaptive_sparse_expanded, dtype=float),
        "adaptive_sparse_margin": np.asarray(adaptive_sparse_margins, dtype=float),
        "adaptive_sparse_effective_margin_threshold": np.asarray(
            adaptive_sparse_effective_thresholds,
            dtype=float,
        ),
        "adaptive_sparse_expand_score": np.asarray(adaptive_sparse_expand_scores, dtype=float),
        "adaptive_sparse_history_stability": np.asarray(
            adaptive_sparse_history_stabilities,
            dtype=float,
        ),
        "adaptive_sparse_urgency": np.asarray(adaptive_sparse_urgencies, dtype=float),
        "adaptive_sparse_cost_penalty": np.asarray(adaptive_sparse_cost_penalties, dtype=float),
        "adaptive_sparse_v3_history_prior_used": np.asarray(
            adaptive_sparse_v3_history_prior_used,
            dtype=float,
        ),
        "adaptive_sparse_v3_neighbor_extra_preview_count": np.asarray(
            adaptive_sparse_v3_neighbor_extra_preview_counts,
            dtype=float,
        ),
        "adaptive_sparse_v3_selected_extra_preview_count": np.asarray(
            adaptive_sparse_v3_selected_extra_preview_counts,
            dtype=float,
        ),
        "learned_shortlist_selected_extra_preview_count": np.asarray(
            learned_shortlist_selected_extra_preview_counts,
            dtype=float,
        ),
        "coverage_sparse_selected_marginal_fraction": np.asarray(
            coverage_sparse_selected_marginal_fractions,
            dtype=float,
        ),
        "coverage_sparse_selected_overlap_fraction": np.asarray(
            coverage_sparse_selected_overlap_fractions,
            dtype=float,
        ),
    }


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
        f"delays={args.csi_delay_slots}, budgets={args.probe_budgets}, "
        f"sparse_seed_multipliers={args.sparse_topk_seed_multipliers}, "
        f"sparse_topk_fractions={args.sparse_topk_fractions}, "
        f"coverage_sparse_weights={args.coverage_sparse_weights}, "
        f"coverage_sparse_power_weights={args.coverage_sparse_power_weights}, "
        f"adaptive_sparse_margins={args.adaptive_sparse_margin_thresholds}, "
        f"adaptive_sparse_v2_preview_costs={args.adaptive_sparse_v2_preview_costs}, "
        f"adaptive_sparse_v3_neighbor_radius={args.adaptive_sparse_v3_neighbor_radius}, "
        f"adaptive_sparse_v3_neighbor_count={args.adaptive_sparse_v3_neighbor_count}, "
        f"adaptive_sparse_v3_history_count={args.adaptive_sparse_v3_history_count}, "
        f"learned_shortlist_extra_counts={args.learned_shortlist_extra_counts}, "
        f"learned_set_extra_counts={args.learned_set_extra_counts}, "
        f"learned_shortlist_model={args.learned_shortlist_model or 'none'}, "
        f"learned_set_shortlist_model={args.learned_set_shortlist_model or 'none'}"
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
