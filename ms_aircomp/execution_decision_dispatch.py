"""根据 policy config 分发到具体策略 helper，让 evaluator 只负责实验编排。"""

import ms_aircomp.limited_csi as limited

from ms_aircomp.adaptive_sparse_policies import (
    choose_adaptive_sparse_topk_feedback_decision,
    choose_adaptive_sparse_topk_v2_feedback_decision,
    choose_adaptive_sparse_topk_v3_feedback_decision,
)
from ms_aircomp.channel_models import apply_channel_state
from ms_aircomp.execution_candidates import (
    choose_execution_oracle,
    execution_candidate_for_decision,
)
from ms_aircomp.execution_policies import (
    choose_active_diverse_feedback_decision,
    choose_coverage_sparse_topk_feedback_decision,
    choose_neighbor_coverage_sparse_topk_feedback_decision,
    choose_rotating_feedback_confirm_decision,
    choose_sparse_topk_feedback_decision,
    choose_stale_topk_feedback_decision,
)
from ms_aircomp.execution_policy_registry import (
    POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
    POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
    POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
    POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
    POLICY_ACTIVE_DIVERSE_FEEDBACK_GRID,
    POLICY_AR1_PREDICT_ROTATING_GRID,
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
)
from ms_aircomp.execution_risk_policies import (
    choose_execution_risk_decision,
    choose_opportunity_execution_risk_decision,
)
from ms_aircomp.learned_shortlist import (
    choose_learned_set_shortlist_feedback_decision,
    choose_learned_sparse_shortlist_feedback_decision,
)
from ms_aircomp.temporal_policies import (
    choose_temporal_deviation_oracle_decision,
    choose_temporal_reliability_decision,
)

__all__ = [
    "choose_decision",
    "choose_execution_mismatch_decision",
]


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
    """按照决策规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
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


def choose_execution_mismatch_decision(
    env,
    args,
    policy_name,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    risk_execution_error_std,
    episode_seed,
    decision_state=None,
    execution_state=None,
    episode_confirmed_irs_history=None,
    channel_rho=0.0,
    csi_delay_slots=0,
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
):
    """按照执行阶段、mismatch、决策规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    if policy_name == POLICY_EXECUTION_ORACLE:
        decision, preview_calls, candidate_count = choose_execution_oracle(
            env,
            args,
            execution_error_std,
            slot_idx,
        )
        return decision, decision, preview_calls, candidate_count

    effective_risk = risk_weight
    if policy_name == limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID:
        effective_risk = adaptive_risk_base_weight
    if policy_name == POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID:
        effective_risk = adaptive_risk_base_weight
    if decision_state is not None:
        apply_channel_state(env, decision_state)

    confirmed_history = (
        episode_confirmed_irs_history
        if episode_confirmed_irs_history is not None
        else []
    )
    if policy_name in {
        POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
        POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
    }:
        decision, preview_calls, candidate_count = choose_execution_risk_decision(
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
        decision, preview_calls, candidate_count = choose_opportunity_execution_risk_decision(
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
        decision, preview_calls, candidate_count = choose_temporal_reliability_decision(
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
    elif policy_name == POLICY_ROTATING_FEEDBACK_CONFIRM_GRID:
        decision, preview_calls, candidate_count = choose_rotating_feedback_confirm_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
        )
    elif policy_name == POLICY_STALE_TOPK_FEEDBACK_GRID:
        decision, preview_calls, candidate_count = choose_stale_topk_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
        )
    elif policy_name == POLICY_ACTIVE_DIVERSE_FEEDBACK_GRID:
        decision, preview_calls, candidate_count = choose_active_diverse_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
        )
    elif policy_name == POLICY_SPARSE_TOPK_FEEDBACK_GRID:
        decision, preview_calls, candidate_count = choose_sparse_topk_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            seed_multiplier=sparse_topk_seed_multiplier,
            topk_fraction=sparse_topk_fraction,
        )
    elif policy_name == POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID:
        decision, preview_calls, candidate_count = choose_coverage_sparse_topk_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            seed_multiplier=sparse_topk_seed_multiplier,
            topk_fraction=sparse_topk_fraction,
            coverage_weight=coverage_sparse_weight,
            power_weight=coverage_sparse_power_weight,
        )
    elif policy_name == POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID:
        decision, preview_calls, candidate_count = choose_neighbor_coverage_sparse_topk_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            seed_multiplier=sparse_topk_seed_multiplier,
            topk_fraction=sparse_topk_fraction,
            coverage_weight=coverage_sparse_weight,
            power_weight=coverage_sparse_power_weight,
            neighbor_radius=adaptive_sparse_v3_neighbor_radius,
            neighbor_count=adaptive_sparse_v3_neighbor_count,
        )
    elif policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID:
        decision, preview_calls, candidate_count = choose_adaptive_sparse_topk_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            base_multiplier=adaptive_sparse_base_multiplier,
            expanded_multiplier=adaptive_sparse_expanded_multiplier,
            topk_fraction=sparse_topk_fraction,
            margin_threshold=adaptive_sparse_margin_threshold,
        )
    elif policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID:
        decision, preview_calls, candidate_count = choose_adaptive_sparse_topk_v2_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            confirmed_history=confirmed_history,
            channel_rho=channel_rho,
            csi_delay_slots=csi_delay_slots,
            base_multiplier=adaptive_sparse_base_multiplier,
            expanded_multiplier=adaptive_sparse_expanded_multiplier,
            topk_fraction=sparse_topk_fraction,
            margin_threshold=adaptive_sparse_margin_threshold,
            preview_cost=adaptive_sparse_v2_preview_cost,
            uncertainty_weight=args.adaptive_sparse_v2_uncertainty_weight,
            urgency_weight=args.adaptive_sparse_v2_urgency_weight,
            history_weight=args.adaptive_sparse_v2_history_weight,
            history_window=args.adaptive_sparse_history_window,
            history_prior_threshold=args.adaptive_sparse_history_prior_threshold,
        )
    elif policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID:
        decision, preview_calls, candidate_count = choose_adaptive_sparse_topk_v3_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            confirmed_history=confirmed_history,
            base_multiplier=adaptive_sparse_base_multiplier,
            topk_fraction=sparse_topk_fraction,
            history_window=args.adaptive_sparse_history_window,
            history_prior_threshold=args.adaptive_sparse_history_prior_threshold,
            history_count=adaptive_sparse_v3_history_count,
            neighbor_radius=adaptive_sparse_v3_neighbor_radius,
            neighbor_count=adaptive_sparse_v3_neighbor_count,
        )
    elif policy_name == POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID:
        decision, preview_calls, candidate_count = choose_learned_sparse_shortlist_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            confirmed_history=confirmed_history,
            channel_rho=channel_rho,
            csi_delay_slots=csi_delay_slots,
            base_multiplier=adaptive_sparse_base_multiplier,
            topk_fraction=sparse_topk_fraction,
            extra_count=learned_shortlist_extra_count,
        )
    elif policy_name == POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID:
        decision, preview_calls, candidate_count = choose_learned_set_shortlist_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            confirmed_history=confirmed_history,
            channel_rho=channel_rho,
            csi_delay_slots=csi_delay_slots,
            base_multiplier=adaptive_sparse_base_multiplier,
            topk_fraction=sparse_topk_fraction,
            extra_count=learned_shortlist_extra_count,
        )
    elif policy_name == POLICY_TEMPORAL_DEVIATION_ORACLE_GRID:
        decision, preview_calls, candidate_count = choose_temporal_deviation_oracle_decision(
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
        decision, preview_calls, candidate_count = choose_decision(
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
    return decision, true_selected, preview_calls, candidate_count
