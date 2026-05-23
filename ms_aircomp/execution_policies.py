"""实现 rotating、stale-topK、sparse-topK、coverage-aware 等执行信道错配主线策略。"""

import numpy as np

import ms_aircomp.limited_csi as limited
from ms_aircomp.confirmation import confirm_index_with_current_feedback
from ms_aircomp.channel_models import apply_channel_state, capture_channel_state
from ms_aircomp.execution_candidates import execution_candidates, execution_oracle_candidate
from ms_aircomp.invitation_mask_correction import (
    MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN,
    apply_invitation_mask_correction,
)
from ms_aircomp.posterior_viability import (
    POSTERIOR_INVITATION_RULE_MEAN_TOPK,
    POSTERIOR_INVITATION_RULE_TOP_Y,
    POSTERIOR_MEAN_MODE_AR1_PREDICT,
    POSTERIOR_TARGET_POLICY_FIXED_Y,
    apply_count_conditioned_invitation_refinement,
    attach_posterior_viability,
    posterior_guided_sparse_indices,
    posterior_viability_matrix,
)
from ms_aircomp.probe_sets import (
    coverage_aware_sparse_indices,
    coverage_increment_stats,
    fill_diverse_codebook_indices,
    ordered_unique_prefix,
    posterior_greedy_probe_indices,
)

__all__ = [
    "choose_active_diverse_feedback_decision",
    "choose_count_only_mask_correction_feedback_decision",
    "choose_coverage_sparse_topk_feedback_decision",
    "choose_coverage_only_fill_feedback_decision",
    "choose_count_conditioned_invitation_feedback_decision",
    "choose_deployable_irs_oracle_invitation_decision",
    "choose_diversity_only_fill_feedback_decision",
    "choose_neighbor_coverage_sparse_topk_feedback_decision",
    "choose_no_irs_fallback",
    "choose_oracle_irs_stale_invitation_decision",
    "choose_posterior_greedy_feedback_decision",
    "choose_posterior_greedy_invitation_feedback_decision",
    "choose_posterior_guided_count_refine_feedback_decision",
    "choose_random_same_budget_feedback_decision",
    "choose_rotating_feedback_confirm_decision",
    "choose_sparse_topk_feedback_decision",
    "choose_stale_topk_same_budget_feedback_decision",
    "choose_stale_topk_feedback_decision",
    "fill_unpreviewed_indices",
    "neighbor_pool_indices",
]


def choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed):
    """构造无 IRS 的直接链路 fallback 决策，用于预算为零或需要基础对照的场景。"""
    return limited.choose_policy_candidate(
        env,
        args,
        limited.POLICY_NO_IRS,
        0,
        slot_idx,
        decision_error_std,
        episode_seed,
    )


def _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks=None):
    """Attach deployable candidate and aggregate-feedback traces without changing decisions."""
    clean_indices = [int(index) for index in selected_indices]
    feedback_by_index = {
        int(feedback["irs_index"]): feedback
        for feedback in (feedbacks or [])
    }
    decision["candidate_irs_indices"] = clean_indices
    decision["deployable_candidate_set"] = clean_indices
    decision["candidate_stale_predicted_counts"] = [
        int(candidate_by_index[int(index)]["tx_this_slot"])
        if int(index) in candidate_by_index
        else None
        for index in clean_indices
    ]
    decision["candidate_stale_power_avg"] = [
        float(candidate_by_index[int(index)]["power_avg"])
        if int(index) in candidate_by_index
        else None
        for index in clean_indices
    ]
    decision["candidate_aggregate_feedback_counts"] = [
        float(feedback_by_index[int(index)]["observed_tx_fraction"]) * float(args.num_nodes)
        if int(index) in feedback_by_index
        else None
        for index in clean_indices
    ]
    decision["candidate_aggregate_feedback_scores"] = [
        float(feedback_by_index[int(index)]["observed_score"])
        if int(index) in feedback_by_index
        else None
        for index in clean_indices
    ]
    confirmed_index = int(decision.get("confirmed_irs_index", decision.get("irs_index", -1)))
    if confirmed_index in feedback_by_index:
        decision["selected_feedback_count"] = (
            float(feedback_by_index[confirmed_index]["observed_tx_fraction"])
            * float(args.num_nodes)
        )
        decision["selected_feedback_score"] = float(
            feedback_by_index[confirmed_index]["observed_score"]
        )
    else:
        decision["selected_feedback_count"] = None
        decision["selected_feedback_score"] = None
    decision["stale_invitation_mask"] = np.asarray(
        decision.get("stale_invitation_mask", decision["valid_mask"]),
        dtype=bool,
    ).copy()
    return decision


def choose_rotating_feedback_confirm_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
):
    """按轮换网格选择少量 IRS probe，用当前聚合反馈确认最终索引，并返回决策候选及成本。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    selected_indices = limited.grid_indices(args.num_codebook_states, budget, offset=slot_idx)
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        budget,
        salt=37 + slot_idx,
    )
    decision_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=selected_indices,
        error_std=decision_error_std,
        rng=error_rng,
    )
    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=43,
    )

    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in decision_candidates}
    decision = candidate_by_index[int(confirmed_index)]
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    return decision, 2 * len(selected_indices), len(selected_indices)


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
    """先完整预览 stale 信道下的码本排序，再取 top-k 与轮换候选组合进行当前反馈确认。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

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
    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=41,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["stale_topk_count"] = int(topk_budget)
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["stale_full_preview_count"] = int(args.num_codebook_states)
    decision["confirmation_feedback_count"] = int(len(selected_indices))
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    return decision, args.num_codebook_states + len(selected_indices), len(selected_indices)


def choose_stale_topk_same_budget_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
):
    """Probe the top-B stale exhaustive IRS states, then select by aggregate feedback."""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        args.num_codebook_states,
        salt=173 + slot_idx,
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
    selected_indices = ordered_unique_prefix(ranked_indices, budget, args.num_codebook_states)
    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in full_stale_candidates}
    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=173,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["stale_same_budget_topk_count"] = int(len(selected_indices))
    decision["stale_full_preview_count"] = int(args.num_codebook_states)
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    return decision, args.num_codebook_states + len(selected_indices), len(selected_indices)


def choose_active_diverse_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
):
    """先用 stale seed 候选找锚点，再用码本距离补齐多样 probe 集合，最后用当前反馈确认。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    seed_indices = limited.grid_indices(args.num_codebook_states, budget, offset=slot_idx)
    seed_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        budget,
        salt=47 + slot_idx,
    )
    seed_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=seed_indices,
        error_std=decision_error_std,
        rng=seed_rng,
    )
    ranked_seed_indices = [
        int(candidate["irs_index"])
        for candidate in sorted(seed_candidates, key=limited.candidate_key, reverse=True)
    ]
    anchor_count = max(1, min(len(ranked_seed_indices), budget // 2))
    selected_indices = fill_diverse_codebook_indices(
        ranked_seed_indices[:anchor_count],
        budget,
        args.num_codebook_states,
    )

    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in seed_candidates}
    extra_indices = [index for index in selected_indices if int(index) not in candidate_by_index]
    if extra_indices:
        extra_rng = limited.stable_rng(
            episode_seed,
            decision_error_std,
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            len(extra_indices),
            salt=53 + slot_idx,
        )
        extra_candidates = limited.estimated_preview_candidates(
            env,
            args,
            indices=extra_indices,
            error_std=decision_error_std,
            rng=extra_rng,
        )
        candidate_by_index.update({int(candidate["irs_index"]): candidate for candidate in extra_candidates})

    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=47,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["active_seed_count"] = int(len(seed_indices))
    decision["active_anchor_count"] = int(anchor_count)
    decision["active_extra_preview_count"] = int(len(extra_indices))
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    preview_calls = len(seed_indices) + len(extra_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)


def choose_sparse_topk_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    seed_multiplier=None,
    topk_fraction=None,
):
    """用少量 stale preview 生成 seed pool，从中取 top 候选并混入轮换覆盖项，降低完整预览成本。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if seed_multiplier is None:
        seed_multiplier = getattr(args, "sparse_topk_seed_multiplier", 2.0)
    if topk_fraction is None:
        topk_fraction = getattr(args, "sparse_topk_fraction", 0.75)
    seed_multiplier = float(seed_multiplier)
    topk_fraction = float(topk_fraction)
    seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(seed_multiplier * budget))),
    )
    seed_indices = limited.grid_indices(args.num_codebook_states, seed_budget, offset=slot_idx)
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        seed_budget,
        salt=61 + slot_idx,
    )
    seed_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=seed_indices,
        error_std=decision_error_std,
        rng=error_rng,
    )
    ranked_indices = [
        int(candidate["irs_index"])
        for candidate in sorted(seed_candidates, key=limited.candidate_key, reverse=True)
    ]
    topk_budget = min(budget, max(1, int(np.ceil(topk_fraction * budget))))
    rotating_indices = limited.grid_indices(args.num_codebook_states, budget, offset=slot_idx)
    selected_indices = ordered_unique_prefix(
        ranked_indices[:topk_budget] + rotating_indices + ranked_indices,
        budget,
        args.num_codebook_states,
    )

    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in seed_candidates}
    extra_indices = [index for index in selected_indices if int(index) not in candidate_by_index]
    if extra_indices:
        extra_rng = limited.stable_rng(
            episode_seed,
            decision_error_std,
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            len(extra_indices),
            salt=67 + slot_idx,
        )
        extra_candidates = limited.estimated_preview_candidates(
            env,
            args,
            indices=extra_indices,
            error_std=decision_error_std,
            rng=extra_rng,
        )
        candidate_by_index.update({int(candidate["irs_index"]): candidate for candidate in extra_candidates})

    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=61,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["sparse_seed_multiplier"] = seed_multiplier
    decision["sparse_topk_fraction"] = topk_fraction
    decision["sparse_seed_count"] = int(len(seed_indices))
    decision["sparse_topk_count"] = int(topk_budget)
    decision["sparse_extra_preview_count"] = int(len(extra_indices))
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    preview_calls = len(seed_indices) + len(extra_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)


def choose_random_same_budget_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    seed_multiplier=None,
):
    """Use the same sparse-preview budget as Sparse-TopK, but randomize the probe set."""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if seed_multiplier is None:
        seed_multiplier = getattr(args, "sparse_topk_seed_multiplier", 4.1)
    seed_multiplier = float(seed_multiplier)
    seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(seed_multiplier * budget))),
    )
    random_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_RANDOM_PROBE,
        seed_budget,
        salt=181 + slot_idx,
    )
    seed_indices = [
        int(index)
        for index in random_rng.choice(
            args.num_codebook_states,
            size=seed_budget,
            replace=False,
        )
    ]
    selected_indices = [int(index) for index in seed_indices[:budget]]
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_RANDOM_PROBE,
        seed_budget,
        salt=187 + slot_idx,
    )
    seed_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=seed_indices,
        error_std=decision_error_std,
        rng=error_rng,
    )
    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in seed_candidates}
    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=181,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["random_same_budget_seed_multiplier"] = seed_multiplier
    decision["random_same_budget_seed_count"] = int(len(seed_indices))
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    preview_calls = len(seed_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)


def choose_coverage_sparse_topk_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    seed_multiplier=None,
    topk_fraction=None,
    coverage_weight=None,
    power_weight=None,
):
    """在稀疏 top-k 基础上加入节点覆盖增量，让 probe 集合减少重复覆盖并降低 missed opportunities。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if seed_multiplier is None:
        seed_multiplier = getattr(args, "sparse_topk_seed_multiplier", 3.0)
    if topk_fraction is None:
        topk_fraction = getattr(args, "sparse_topk_fraction", 0.75)
    if coverage_weight is None:
        coverage_weight = getattr(args, "coverage_sparse_weight", 0.5)
    if power_weight is None:
        power_weight = getattr(args, "coverage_sparse_power_weight", 0.0)
    seed_multiplier = float(seed_multiplier)
    topk_fraction = float(topk_fraction)
    coverage_weight = float(coverage_weight)
    power_weight = float(power_weight)
    seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(seed_multiplier * budget))),
    )
    seed_indices = limited.grid_indices(args.num_codebook_states, seed_budget, offset=slot_idx)
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        seed_budget,
        salt=191 + slot_idx,
    )
    seed_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=seed_indices,
        error_std=decision_error_std,
        rng=error_rng,
    )
    selected_indices, anchor_count, marginal_mean, overlap_mean = coverage_aware_sparse_indices(
        seed_candidates,
        args,
        budget,
        topk_fraction,
        coverage_weight,
        power_weight,
    )

    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in seed_candidates}
    if len(selected_indices) < budget:
        ranked_indices = [
            int(candidate["irs_index"])
            for candidate in sorted(seed_candidates, key=limited.candidate_key, reverse=True)
        ]
        selected_indices = ordered_unique_prefix(
            selected_indices + ranked_indices,
            budget,
            args.num_codebook_states,
        )
        marginal_mean, overlap_mean = coverage_increment_stats(
            candidate_by_index,
            selected_indices,
            args.num_nodes,
        )

    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=191,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["sparse_seed_multiplier"] = seed_multiplier
    decision["sparse_topk_fraction"] = topk_fraction
    decision["sparse_seed_count"] = int(len(seed_indices))
    decision["sparse_topk_count"] = int(anchor_count)
    decision["coverage_sparse_weight"] = coverage_weight
    decision["coverage_sparse_power_weight"] = power_weight
    decision["coverage_sparse_selected_marginal_fraction"] = float(marginal_mean)
    decision["coverage_sparse_selected_overlap_fraction"] = float(overlap_mean)
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    preview_calls = len(seed_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)


def choose_diversity_only_fill_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
):
    """Select probes only through the codebook diversity fill rule, then use aggregate feedback."""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    anchor = int(slot_idx) % int(args.num_codebook_states)
    selected_indices = fill_diverse_codebook_indices([anchor], budget, args.num_codebook_states)
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        budget,
        salt=281 + slot_idx,
    )
    candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=selected_indices,
        error_std=decision_error_std,
        rng=error_rng,
    )
    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in candidates}
    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=281,
    )
    decision = candidate_by_index[int(confirmed_index)]
    decision["diversity_only_anchor_index"] = int(anchor)
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    return decision, 2 * len(selected_indices), len(selected_indices)


def choose_coverage_only_fill_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    seed_multiplier=None,
):
    """Greedily fill probes by marginal stale device coverage only."""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if seed_multiplier is None:
        seed_multiplier = getattr(args, "sparse_topk_seed_multiplier", 3.0)
    seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(float(seed_multiplier) * budget))),
    )
    seed_indices = limited.grid_indices(args.num_codebook_states, seed_budget, offset=slot_idx)
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        seed_budget,
        salt=293 + slot_idx,
    )
    seed_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=seed_indices,
        error_std=decision_error_std,
        rng=error_rng,
    )
    selected_indices = []
    covered_mask = np.zeros(int(args.num_nodes), dtype=bool)
    remaining = list(seed_candidates)
    while len(selected_indices) < budget and remaining:
        def coverage_key(candidate):
            candidate_mask = np.asarray(candidate["valid_mask"], dtype=bool)
            marginal_count = int(np.sum(candidate_mask & (~covered_mask)))
            return (
                marginal_count,
                int(candidate["tx_this_slot"]),
                float(candidate["mean_gain_remaining"]),
                -int(candidate["irs_index"]),
            )

        best_candidate = max(remaining, key=coverage_key)
        remaining = [
            candidate
            for candidate in remaining
            if int(candidate["irs_index"]) != int(best_candidate["irs_index"])
        ]
        selected_indices.append(int(best_candidate["irs_index"]))
        covered_mask |= np.asarray(best_candidate["valid_mask"], dtype=bool)
    if len(selected_indices) < budget:
        selected_indices = ordered_unique_prefix(
            selected_indices + list(seed_indices),
            budget,
            args.num_codebook_states,
        )

    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in seed_candidates}
    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=293,
    )
    marginal_mean, overlap_mean = coverage_increment_stats(
        candidate_by_index,
        selected_indices,
        args.num_nodes,
    )
    decision = candidate_by_index[int(confirmed_index)]
    decision["coverage_only_seed_multiplier"] = float(seed_multiplier)
    decision["coverage_only_seed_count"] = int(len(seed_indices))
    decision["coverage_only_selected_marginal_fraction"] = float(marginal_mean)
    decision["coverage_only_selected_overlap_fraction"] = float(overlap_mean)
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    preview_calls = len(seed_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)


def _execution_candidate_on_state(env, args, index, execution_error_std, slot_idx, execution_state):
    snapshot = capture_channel_state(env)
    try:
        if execution_state is not None:
            apply_channel_state(env, execution_state)
        return execution_candidates(
            env,
            args,
            indices=[int(index)],
            execution_error_std=execution_error_std,
            slot_idx=slot_idx,
        )[0]
    finally:
        apply_channel_state(env, snapshot)


def _execution_oracle_on_state(env, args, execution_error_std, slot_idx, execution_state):
    snapshot = capture_channel_state(env)
    try:
        if execution_state is not None:
            apply_channel_state(env, execution_state)
        return execution_oracle_candidate(env, args, execution_error_std, slot_idx)
    finally:
        apply_channel_state(env, snapshot)


def choose_oracle_irs_stale_invitation_decision(
    env,
    args,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
):
    """Use hidden current CSI only to choose the IRS, but keep stale invitations."""
    oracle = _execution_oracle_on_state(
        env,
        args,
        execution_error_std,
        slot_idx,
        execution_state,
    )
    oracle_index = int(oracle["irs_index"])
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_EST_GREEDY,
        args.num_codebook_states,
        salt=307 + slot_idx,
    )
    stale_candidate = limited.estimated_preview_candidates(
        env,
        args,
        indices=[oracle_index],
        error_std=decision_error_std,
        rng=error_rng,
    )[0]
    decision = dict(stale_candidate)
    decision["confirmed_irs_index"] = oracle_index
    decision["oracle_irs_index"] = oracle_index
    decision["stale_invitation_mask"] = np.asarray(stale_candidate["valid_mask"], dtype=bool).copy()
    decision["oracle_diagnostic_type"] = "oracle_irs_with_stale_invitation"
    decision["candidate_irs_indices"] = [oracle_index]
    decision["deployable_candidate_set"] = [oracle_index]
    decision["candidate_stale_predicted_counts"] = [int(stale_candidate["tx_this_slot"])]
    decision["candidate_aggregate_feedback_counts"] = [None]
    decision["selected_feedback_count"] = None
    return decision, args.num_codebook_states, args.num_codebook_states


def choose_deployable_irs_oracle_invitation_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    seed_multiplier=None,
    topk_fraction=None,
    coverage_weight=None,
    power_weight=None,
):
    """Use deployable Coverage-Aware probing, then reveal oracle invitations for that IRS."""
    decision, preview_calls, candidate_count = choose_coverage_sparse_topk_feedback_decision(
        env,
        args,
        budget,
        slot_idx,
        decision_error_std,
        execution_error_std,
        episode_seed,
        execution_state=execution_state,
        seed_multiplier=seed_multiplier,
        topk_fraction=topk_fraction,
        coverage_weight=coverage_weight,
        power_weight=power_weight,
    )
    confirmed_index = int(decision.get("confirmed_irs_index", decision["irs_index"]))
    oracle_same_irs = _execution_candidate_on_state(
        env,
        args,
        confirmed_index,
        execution_error_std,
        slot_idx,
        execution_state,
    )
    stale_mask = np.asarray(decision["valid_mask"], dtype=bool).copy()
    for key in (
        "valid_mask",
        "tx_this_slot",
        "required_power",
        "h_gain",
        "success_reliability",
        "success_margin",
        "power_avg",
        "mean_gain_remaining",
    ):
        decision[key] = oracle_same_irs[key]
    decision["stale_invitation_mask"] = stale_mask
    decision["deployable_oracle_invitation_mask"] = np.asarray(
        oracle_same_irs["valid_mask"],
        dtype=bool,
    ).copy()
    decision["oracle_diagnostic_type"] = "deployable_irs_with_oracle_invitation"
    return decision, preview_calls, candidate_count


def _posterior_viability_matrix_for_slot(env, args, channel_rho, csi_delay_slots, episode_seed, slot_idx):
    return posterior_viability_matrix(
        stale_state=capture_channel_state(env),
        codebook=env.codebook,
        args=args,
        p_max=env.P_max,
        channel_rho=channel_rho,
        csi_delay_slots=csi_delay_slots,
        posterior_mode=getattr(args, "posterior_mode", "analytic"),
        posterior_num_samples=getattr(args, "posterior_num_samples", 256),
        posterior_clip_eps=getattr(args, "posterior_clip_eps", 1e-6),
        posterior_seed_offset=getattr(args, "posterior_seed_offset", 0),
        episode_seed=episode_seed,
        slot_idx=slot_idx,
    )


def _apply_count_conditioned_invitation_feedback(
    decision,
    args,
    env,
    posterior_matrix,
    episode_seed,
    slot_idx,
    channel_rho,
    csi_delay_slots,
):
    if posterior_matrix is None:
        posterior_matrix = _posterior_viability_matrix_for_slot(
            env,
            args,
            channel_rho,
            csi_delay_slots,
            episode_seed,
            slot_idx,
        )
    confirmed_index = int(decision.get("confirmed_irs_index", decision["irs_index"]))
    probabilities = posterior_matrix[:, confirmed_index]
    feedback_count = float(decision.get("selected_feedback_count", 0.0) or 0.0)
    decision["stale_invitation_mask"] = np.asarray(decision["valid_mask"], dtype=bool).copy()
    decision = apply_count_conditioned_invitation_refinement(
        decision,
        args,
        env,
        {"observed_tx_fraction": feedback_count / max(float(args.num_nodes), 1.0)},
        probabilities,
        1.0,
        getattr(args, "posterior_count_noise_std_scale", 1.0),
        POSTERIOR_INVITATION_RULE_TOP_Y,
        getattr(args, "posterior_invitation_threshold", 0.5),
        cardinality_policy=getattr(
            args,
            "posterior_cardinality_policy",
            POSTERIOR_TARGET_POLICY_FIXED_Y,
        ),
        cumulative_probability_target=getattr(
            args,
            "posterior_cumulative_probability_target",
            1.0,
        ),
        lambda_fail=getattr(args, "posterior_lambda_fail", 1.0),
        lambda_miss=getattr(args, "posterior_lambda_miss", 1.0),
    )
    decision["posterior_invitation_mask"] = np.asarray(decision["valid_mask"], dtype=bool).copy()
    decision["posterior_invitation_method"] = "count_conditioned_invitation"
    return decision


def choose_posterior_greedy_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    channel_rho=0.0,
    csi_delay_slots=0,
):
    """Posterior-guided IRS probing under the same aggregate-feedback budget.

    For remaining devices R and codebook states C, this policy estimates
    p[k,c]=P(z[k,c]=1 | stale CSI), samples feasible-count tables Y_c, and
    greedily selects B IRS states to maximize a sample estimate of
    E[max_{c in S} Y_c], optionally plus posterior coverage. The invitation
    mask for the confirmed IRS state remains the stale simulator mask.
    """
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    posterior_matrix = _posterior_viability_matrix_for_slot(
        env,
        args,
        channel_rho,
        csi_delay_slots,
        episode_seed,
        slot_idx,
    )
    remaining_mask = ~np.asarray(env.transmitted_flags, dtype=bool)
    details = posterior_greedy_probe_indices(
        posterior_matrix[remaining_mask],
        budget,
        sample_count=getattr(args, "posterior_probe_samples", 128),
        beta=getattr(args, "posterior_probe_beta", 0.0),
        objective=getattr(args, "posterior_probe_objective", "expected_best_count"),
        seed_offset=getattr(args, "posterior_probe_seed_offset", 0),
        episode_seed=episode_seed,
        slot_idx=slot_idx,
        candidate_prefilter_size=getattr(args, "posterior_probe_candidate_prefilter_size", 0),
    )
    selected_indices = [int(index) for index in details["selected_indices"]]
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        len(selected_indices),
        salt=271 + slot_idx,
    )
    decision_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=selected_indices,
        error_std=decision_error_std,
        rng=error_rng,
    )
    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in decision_candidates}
    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=271,
    )

    decision = dict(candidate_by_index[int(confirmed_index)])
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    decision["posterior_probe_selected_indices"] = selected_indices
    decision["posterior_probe_objective_value"] = float(details["objective_value"])
    decision["posterior_probe_expected_best_count"] = float(details["expected_best_count"])
    decision["posterior_probe_coverage_score"] = float(details["coverage_score"])
    decision["posterior_probe_expected_count_per_selected_state"] = list(
        details["expected_count_per_selected_state"]
    )
    decision["posterior_probe_candidate_prefilter_indices"] = list(
        details["candidate_prefilter_indices"]
    )
    decision["posterior_probe_candidate_prefilter_size"] = int(
        details["posterior_probe_candidate_prefilter_size"]
    )
    decision["posterior_probe_samples"] = int(details["posterior_probe_samples"])
    decision["posterior_probe_beta"] = float(details["posterior_probe_beta"])
    decision["posterior_probe_objective"] = details["posterior_probe_objective"]
    decision["posterior_probe_remaining_device_count"] = int(np.sum(remaining_mask))
    decision["posterior_probe_computed_state_count"] = int(posterior_matrix.shape[1])
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    preview_calls = len(selected_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)


def choose_posterior_greedy_invitation_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    channel_rho=0.0,
    csi_delay_slots=0,
):
    """Compose posterior-greedy IRS probing with count-conditioned invitations."""
    decision, preview_calls, candidate_count = choose_posterior_greedy_feedback_decision(
        env,
        args,
        budget,
        slot_idx,
        decision_error_std,
        execution_error_std,
        episode_seed,
        execution_state=execution_state,
        channel_rho=channel_rho,
        csi_delay_slots=csi_delay_slots,
    )
    decision = _apply_count_conditioned_invitation_feedback(
        decision,
        args,
        env,
        None,
        episode_seed,
        slot_idx,
        channel_rho,
        csi_delay_slots,
    )
    return decision, preview_calls, candidate_count


def choose_posterior_guided_count_refine_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    seed_multiplier=None,
    topk_fraction=None,
    coverage_weight=None,
    power_weight=None,
    posterior_sample_count=None,
    posterior_uncertainty_scale=None,
    posterior_probe_uncertainty_weight=None,
    posterior_count_refinement_strength=None,
    posterior_count_noise_std_scale=None,
    posterior_mean_mode=None,
    posterior_invitation_rule=None,
    posterior_invitation_threshold=None,
    channel_rho=0.0,
    csi_delay_slots=0,
):
    """Posterior-guided probing with Bayesian aggregate-count invitation refinement.

    The policy first estimates per-node viability probabilities from stale CSI,
    then selects IRS probes by posterior expected count plus marginal
    probability coverage. After aggregate feedback confirms the IRS index, the
    confirmed candidate's invitation mask is refined by conditioning the
    Bernoulli viability prior on the observed aggregate count.
    """
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if seed_multiplier is None:
        seed_multiplier = getattr(args, "sparse_topk_seed_multiplier", 4.1)
    if topk_fraction is None:
        topk_fraction = getattr(args, "sparse_topk_fraction", 0.75)
    if coverage_weight is None:
        coverage_weight = getattr(args, "coverage_sparse_weight", 0.5)
    if power_weight is None:
        power_weight = getattr(args, "coverage_sparse_power_weight", 0.0)
    if posterior_sample_count is None:
        posterior_sample_count = getattr(args, "posterior_sample_count", 64)
    if posterior_uncertainty_scale is None:
        posterior_uncertainty_scale = getattr(args, "posterior_uncertainty_scale", 1.0)
    if posterior_probe_uncertainty_weight is None:
        posterior_probe_uncertainty_weight = getattr(
            args,
            "posterior_probe_uncertainty_weight",
            0.0,
        )
    if posterior_count_refinement_strength is None:
        posterior_count_refinement_strength = getattr(
            args,
            "posterior_count_refinement_strength",
            1.0,
        )
    if posterior_count_noise_std_scale is None:
        posterior_count_noise_std_scale = getattr(
            args,
            "posterior_count_noise_std_scale",
            1.0,
        )
    if posterior_mean_mode is None:
        posterior_mean_mode = getattr(
            args,
            "posterior_mean_mode",
            POSTERIOR_MEAN_MODE_AR1_PREDICT,
        )
    if posterior_invitation_rule is None:
        posterior_invitation_rule = getattr(
            args,
            "posterior_invitation_rule",
            POSTERIOR_INVITATION_RULE_MEAN_TOPK,
        )
    if posterior_invitation_threshold is None:
        posterior_invitation_threshold = getattr(args, "posterior_invitation_threshold", 0.5)

    seed_multiplier = float(seed_multiplier)
    topk_fraction = float(topk_fraction)
    coverage_weight = float(coverage_weight)
    power_weight = float(power_weight)
    posterior_sample_count = int(posterior_sample_count)
    posterior_uncertainty_scale = float(posterior_uncertainty_scale)
    posterior_probe_uncertainty_weight = float(posterior_probe_uncertainty_weight)
    posterior_count_refinement_strength = float(posterior_count_refinement_strength)
    posterior_count_noise_std_scale = float(posterior_count_noise_std_scale)
    posterior_invitation_threshold = float(posterior_invitation_threshold)

    seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(seed_multiplier * budget))),
    )
    seed_indices = limited.grid_indices(args.num_codebook_states, seed_budget, offset=slot_idx)
    error_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        seed_budget,
        salt=251 + slot_idx,
    )
    seed_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=seed_indices,
        error_std=decision_error_std,
        rng=error_rng,
    )
    posterior_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        seed_budget,
        salt=257 + slot_idx,
    )
    seed_candidates = [
        attach_posterior_viability(
            candidate,
            args,
            env,
            channel_rho,
            csi_delay_slots,
            posterior_sample_count,
            posterior_uncertainty_scale,
            posterior_mean_mode,
            posterior_rng,
        )
        for candidate in seed_candidates
    ]
    selected_indices, anchor_count, marginal_mean, overlap_mean = posterior_guided_sparse_indices(
        seed_candidates,
        args,
        budget,
        topk_fraction,
        coverage_weight,
        power_weight,
        posterior_probe_uncertainty_weight,
    )

    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in seed_candidates}
    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=251,
    )
    feedback_by_index = {int(feedback["irs_index"]): feedback for feedback in feedbacks}

    decision = dict(candidate_by_index[int(confirmed_index)])
    decision["stale_invitation_mask"] = np.asarray(decision["valid_mask"], dtype=bool).copy()
    decision = apply_count_conditioned_invitation_refinement(
        decision,
        args,
        env,
        feedback_by_index[int(confirmed_index)],
        decision["posterior_viability_prob"],
        posterior_count_refinement_strength,
        posterior_count_noise_std_scale,
        posterior_invitation_rule,
        posterior_invitation_threshold,
    )
    decision["sparse_seed_multiplier"] = seed_multiplier
    decision["sparse_topk_fraction"] = topk_fraction
    decision["sparse_seed_count"] = int(len(seed_indices))
    decision["sparse_topk_count"] = int(anchor_count)
    decision["coverage_sparse_weight"] = coverage_weight
    decision["coverage_sparse_power_weight"] = power_weight
    decision["posterior_sample_count"] = int(posterior_sample_count)
    decision["posterior_uncertainty_scale"] = posterior_uncertainty_scale
    decision["posterior_probe_uncertainty_weight"] = posterior_probe_uncertainty_weight
    decision["posterior_count_refinement_strength"] = posterior_count_refinement_strength
    decision["posterior_count_noise_std_scale"] = posterior_count_noise_std_scale
    decision["posterior_mean_mode"] = posterior_mean_mode
    decision["posterior_invitation_rule"] = posterior_invitation_rule
    decision["posterior_invitation_threshold"] = posterior_invitation_threshold
    decision["posterior_selected_marginal_fraction"] = float(marginal_mean)
    decision["posterior_selected_overlap_fraction"] = float(overlap_mean)
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    decision["posterior_invitation_mask"] = np.asarray(decision["valid_mask"], dtype=bool).copy()
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    preview_calls = len(seed_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)


def choose_count_conditioned_invitation_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    seed_multiplier=None,
    topk_fraction=None,
    coverage_weight=None,
    power_weight=None,
    channel_rho=0.0,
    csi_delay_slots=0,
):
    """Coverage-Aware candidate generation plus count-conditioned invitation.

    By default this method reuses Coverage-Aware Sparse-TopK probing and IRS
    confirmation. The optional ``args.probing_policy`` switch composes the same
    invitation refinement with rotating, sparse-topK, or posterior-greedy
    probing without changing the Bayesian invitation rule itself.
    """
    probing_policy = getattr(args, "probing_policy", "coverage_aware")
    if probing_policy == "posterior_greedy":
        decision, preview_calls, candidate_count = choose_posterior_greedy_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            channel_rho=channel_rho,
            csi_delay_slots=csi_delay_slots,
        )
    elif probing_policy == "rotating":
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
    elif probing_policy == "sparse_topk":
        decision, preview_calls, candidate_count = choose_sparse_topk_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            seed_multiplier=seed_multiplier,
            topk_fraction=topk_fraction,
        )
    else:
        decision, preview_calls, candidate_count = choose_coverage_sparse_topk_feedback_decision(
            env,
            args,
            budget,
            slot_idx,
            decision_error_std,
            execution_error_std,
            episode_seed,
            execution_state=execution_state,
            seed_multiplier=seed_multiplier,
            topk_fraction=topk_fraction,
            coverage_weight=coverage_weight,
            power_weight=power_weight,
        )
    decision = _apply_count_conditioned_invitation_feedback(
        decision,
        args,
        env,
        None,
        episode_seed,
        slot_idx,
        channel_rho,
        csi_delay_slots,
    )
    return decision, preview_calls, candidate_count


def choose_count_only_mask_correction_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    seed_multiplier=None,
    topk_fraction=None,
    coverage_weight=None,
    power_weight=None,
):
    """Coverage-Aware probing plus the legacy count-only invitation correction baseline."""
    decision, preview_calls, candidate_count = choose_coverage_sparse_topk_feedback_decision(
        env,
        args,
        budget,
        slot_idx,
        decision_error_std,
        execution_error_std,
        episode_seed,
        execution_state=execution_state,
        seed_multiplier=seed_multiplier,
        topk_fraction=topk_fraction,
        coverage_weight=coverage_weight,
        power_weight=power_weight,
    )
    feedback_count = float(decision.get("selected_feedback_count", 0.0) or 0.0)
    stale_mask = np.asarray(decision["valid_mask"], dtype=bool).copy()
    corrected = apply_invitation_mask_correction(
        decision,
        args,
        env,
        {"observed_tx_fraction": feedback_count / max(float(args.num_nodes), 1.0)},
        strength=1.0,
        deadband_z=0.0,
        max_delta=-1.0,
        rerank_mode=MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN,
    )
    corrected["stale_invitation_mask"] = stale_mask
    corrected["corrected_invitation_mask"] = np.asarray(corrected["valid_mask"], dtype=bool).copy()
    corrected["count_only_mask_correction_method"] = "global_stale_gain"
    return corrected, preview_calls, candidate_count


def neighbor_pool_indices(center_indices, num_codebook_states, radius, max_count, exclude_indices):
    """处理邻域、pool、索引集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if max_count <= 0 or radius <= 0:
        return []
    selected = []
    seen = {int(index) for index in exclude_indices}
    for center in center_indices:
        center = int(center) % int(num_codebook_states)
        for delta in range(1, int(radius) + 1):
            for index in (
                (center - delta) % int(num_codebook_states),
                (center + delta) % int(num_codebook_states),
            ):
                if int(index) in seen:
                    continue
                selected.append(int(index))
                seen.add(int(index))
                if len(selected) >= int(max_count):
                    return selected
    return selected


def fill_unpreviewed_indices(prefix_indices, num_codebook_states, max_count, exclude_indices):
    """处理补齐、unpreviewed、索引集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    selected = []
    seen = {int(index) for index in exclude_indices}
    for raw_index in list(prefix_indices) + list(range(int(num_codebook_states))):
        index = int(raw_index) % int(num_codebook_states)
        if index in seen:
            continue
        selected.append(index)
        seen.add(index)
        if len(selected) >= int(max_count):
            break
    return selected


def choose_neighbor_coverage_sparse_topk_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    seed_multiplier=None,
    topk_fraction=None,
    coverage_weight=None,
    power_weight=None,
    neighbor_radius=None,
    neighbor_count=None,
):
    """按照邻域、覆盖感知、稀疏、TopK、聚合反馈、决策规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if seed_multiplier is None:
        seed_multiplier = getattr(args, "sparse_topk_seed_multiplier", 4.1)
    if topk_fraction is None:
        topk_fraction = getattr(args, "sparse_topk_fraction", 0.75)
    if coverage_weight is None:
        coverage_weight = getattr(args, "coverage_sparse_weight", 0.5)
    if power_weight is None:
        power_weight = getattr(args, "coverage_sparse_power_weight", 0.0)
    if neighbor_radius is None:
        neighbor_radius = getattr(args, "adaptive_sparse_v3_neighbor_radius", 1)
    if neighbor_count is None:
        neighbor_count = getattr(args, "adaptive_sparse_v3_neighbor_count", 3)

    seed_multiplier = float(seed_multiplier)
    topk_fraction = float(topk_fraction)
    coverage_weight = float(coverage_weight)
    power_weight = float(power_weight)
    neighbor_radius = max(int(neighbor_radius), 0)
    neighbor_count = max(int(neighbor_count), 0)
    total_seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(seed_multiplier * budget))),
    )
    neighbor_budget = min(neighbor_count, max(total_seed_budget - budget, 0))
    base_seed_budget = max(budget, total_seed_budget - neighbor_budget)
    base_seed_indices = limited.grid_indices(
        args.num_codebook_states,
        base_seed_budget,
        offset=slot_idx,
    )
    base_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        base_seed_budget,
        salt=211 + slot_idx,
    )
    base_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=base_seed_indices,
        error_std=decision_error_std,
        rng=base_rng,
    )
    ranked_indices = [
        int(candidate["irs_index"])
        for candidate in sorted(base_candidates, key=limited.candidate_key, reverse=True)
    ]
    topk_budget = min(budget, max(1, int(np.ceil(topk_fraction * budget))))
    neighbor_centers = ranked_indices[: max(1, min(topk_budget, 2))]
    neighbor_indices = neighbor_pool_indices(
        neighbor_centers,
        args.num_codebook_states,
        neighbor_radius,
        neighbor_budget,
        base_seed_indices,
    )
    if len(neighbor_indices) < neighbor_budget:
        neighbor_indices += fill_unpreviewed_indices(
            ranked_indices,
            args.num_codebook_states,
            neighbor_budget - len(neighbor_indices),
            list(base_seed_indices) + neighbor_indices,
        )

    neighbor_candidates = []
    if neighbor_indices:
        neighbor_rng = limited.stable_rng(
            episode_seed,
            decision_error_std,
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            len(neighbor_indices),
            salt=223 + slot_idx,
        )
        neighbor_candidates = limited.estimated_preview_candidates(
            env,
            args,
            indices=neighbor_indices,
            error_std=decision_error_std,
            rng=neighbor_rng,
        )

    seed_candidates = base_candidates + neighbor_candidates
    selected_indices, anchor_count, marginal_mean, overlap_mean = coverage_aware_sparse_indices(
        seed_candidates,
        args,
        budget,
        topk_fraction,
        coverage_weight,
        power_weight,
    )
    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in seed_candidates}
    if len(selected_indices) < budget:
        combined_ranked_indices = [
            int(candidate["irs_index"])
            for candidate in sorted(seed_candidates, key=limited.candidate_key, reverse=True)
        ]
        selected_indices = ordered_unique_prefix(
            selected_indices + combined_ranked_indices,
            budget,
            args.num_codebook_states,
        )
        marginal_mean, overlap_mean = coverage_increment_stats(
            candidate_by_index,
            selected_indices,
            args.num_nodes,
        )

    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=211,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["sparse_seed_multiplier"] = seed_multiplier
    decision["sparse_topk_fraction"] = topk_fraction
    decision["sparse_seed_count"] = int(len(seed_candidates))
    decision["sparse_topk_count"] = int(anchor_count)
    decision["coverage_sparse_weight"] = coverage_weight
    decision["coverage_sparse_power_weight"] = power_weight
    decision["coverage_sparse_selected_marginal_fraction"] = float(marginal_mean)
    decision["coverage_sparse_selected_overlap_fraction"] = float(overlap_mean)
    decision["adaptive_sparse_v3_neighbor_extra_preview_count"] = float(len(neighbor_indices))
    decision["adaptive_sparse_v3_selected_extra_preview_count"] = float(
        len([index for index in selected_indices if int(index) in set(neighbor_indices)])
    )
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    _attach_feedback_logging(decision, args, selected_indices, candidate_by_index, feedbacks)
    preview_calls = len(seed_candidates) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)
