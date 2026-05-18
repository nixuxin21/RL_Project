"""实现 Adaptive Sparse-TopK 系列策略，结合 margin、历史反馈、deadline 和邻域候选调节 preview 成本。"""

import numpy as np

import ms_aircomp.limited_csi as limited
from ms_aircomp.confirmation import confirm_index_with_current_feedback
from ms_aircomp.execution_policies import choose_no_irs_fallback
from ms_aircomp.probe_sets import ordered_unique_prefix

__all__ = [
    "adaptive_sparse_deadline_urgency",
    "adaptive_sparse_history_indices",
    "adaptive_sparse_history_signal",
    "adaptive_sparse_stale_uncertainty",
    "choose_adaptive_sparse_topk_feedback_decision",
    "choose_adaptive_sparse_topk_v2_feedback_decision",
    "choose_adaptive_sparse_topk_v3_feedback_decision",
    "local_codebook_neighbor_indices",
    "sparse_topk_count_margin",
]


def sparse_topk_count_margin(seed_candidates, topk_budget, num_nodes):
    """处理稀疏、TopK、count、margin相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if not seed_candidates:
        return 0.0
    ranked = sorted(seed_candidates, key=limited.candidate_key, reverse=True)
    kth_index = min(max(int(topk_budget), 1), len(ranked)) - 1
    top_count = float(ranked[0]["tx_this_slot"])
    kth_count = float(ranked[kth_index]["tx_this_slot"])
    return max(0.0, (top_count - kth_count) / max(float(num_nodes), 1.0))


def choose_adaptive_sparse_topk_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    base_multiplier=None,
    expanded_multiplier=None,
    topk_fraction=None,
    margin_threshold=None,
):
    """按照adaptive、稀疏、TopK、聚合反馈、决策规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if base_multiplier is None:
        base_multiplier = getattr(args, "adaptive_sparse_base_multiplier", 2.0)
    if expanded_multiplier is None:
        expanded_multiplier = getattr(args, "adaptive_sparse_expanded_multiplier", 3.0)
    if topk_fraction is None:
        topk_fraction = getattr(args, "sparse_topk_fraction", 0.75)
    if margin_threshold is None:
        margin_threshold = getattr(args, "adaptive_sparse_margin_threshold", 0.05)

    base_multiplier = float(base_multiplier)
    expanded_multiplier = max(float(expanded_multiplier), base_multiplier)
    topk_fraction = float(topk_fraction)
    margin_threshold = float(margin_threshold)
    topk_budget = min(budget, max(1, int(np.ceil(topk_fraction * budget))))

    base_seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(base_multiplier * budget))),
    )
    base_seed_indices = limited.grid_indices(args.num_codebook_states, base_seed_budget, offset=slot_idx)
    base_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        base_seed_budget,
        salt=83 + slot_idx,
    )
    base_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=base_seed_indices,
        error_std=decision_error_std,
        rng=base_rng,
    )
    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in base_candidates}
    stale_margin = sparse_topk_count_margin(base_candidates, topk_budget, args.num_nodes)
    should_expand = stale_margin < margin_threshold

    expansion_indices = []
    applied_multiplier = base_multiplier
    if should_expand:
        expanded_seed_budget = min(
            args.num_codebook_states,
            max(budget, int(np.ceil(expanded_multiplier * budget))),
        )
        expanded_seed_indices = limited.grid_indices(
            args.num_codebook_states,
            expanded_seed_budget,
            offset=slot_idx,
        )
        expansion_indices = [
            index for index in expanded_seed_indices if int(index) not in candidate_by_index
        ]
        if expansion_indices:
            expansion_rng = limited.stable_rng(
                episode_seed,
                decision_error_std,
                limited.POLICY_RISK_AWARE_ROTATING_GRID,
                len(expansion_indices),
                salt=89 + slot_idx,
            )
            expansion_candidates = limited.estimated_preview_candidates(
                env,
                args,
                indices=expansion_indices,
                error_std=decision_error_std,
                rng=expansion_rng,
            )
            candidate_by_index.update(
                {int(candidate["irs_index"]): candidate for candidate in expansion_candidates}
            )
        applied_multiplier = expanded_multiplier

    ranked_indices = [
        int(candidate["irs_index"])
        for candidate in sorted(candidate_by_index.values(), key=limited.candidate_key, reverse=True)
    ]
    rotating_indices = limited.grid_indices(args.num_codebook_states, budget, offset=slot_idx)
    selected_indices = ordered_unique_prefix(
        ranked_indices[:topk_budget] + rotating_indices + ranked_indices,
        budget,
        args.num_codebook_states,
    )

    extra_indices = [index for index in selected_indices if int(index) not in candidate_by_index]
    if extra_indices:
        extra_rng = limited.stable_rng(
            episode_seed,
            decision_error_std,
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            len(extra_indices),
            salt=97 + slot_idx,
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
        feedback_salt=83,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["sparse_seed_multiplier"] = applied_multiplier
    decision["sparse_topk_fraction"] = topk_fraction
    decision["sparse_seed_count"] = int(len(candidate_by_index) - len(extra_indices))
    decision["sparse_topk_count"] = int(topk_budget)
    decision["sparse_extra_preview_count"] = int(len(extra_indices))
    decision["adaptive_sparse_base_multiplier"] = base_multiplier
    decision["adaptive_sparse_expanded_multiplier"] = expanded_multiplier
    decision["adaptive_sparse_margin_threshold"] = margin_threshold
    decision["adaptive_sparse_margin"] = float(stale_margin)
    decision["adaptive_sparse_expanded"] = float(should_expand)
    decision["adaptive_sparse_expansion_preview_count"] = int(len(expansion_indices))
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    preview_calls = len(base_seed_indices) + len(expansion_indices) + len(extra_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)


def adaptive_sparse_history_signal(confirmed_history, window):
    """处理adaptive、稀疏、history、signal相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if not confirmed_history:
        return None, 0.0
    tail = [int(index) for index in confirmed_history[-max(int(window), 1) :]]
    if not tail:
        return None, 0.0
    counts = {}
    for index in tail:
        counts[index] = counts.get(index, 0) + 1
    best_index, best_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    return int(best_index), float(best_count) / float(len(tail))


def adaptive_sparse_deadline_urgency(env, args, slot_idx, base_candidates):
    """处理adaptive、稀疏、deadline、urgency相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    remaining_nodes = int(args.num_nodes) - int(np.sum(env.transmitted_flags))
    remaining_slots = max(int(args.num_slots) - int(slot_idx), 1)
    required_per_slot = float(remaining_nodes) / float(remaining_slots)
    best_base_count = max((float(candidate["tx_this_slot"]) for candidate in base_candidates), default=0.0)
    return max(0.0, required_per_slot - best_base_count) / max(float(args.num_nodes), 1.0)


def adaptive_sparse_stale_uncertainty(channel_rho, csi_delay_slots):
    """处理adaptive、稀疏、过时、uncertainty相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    delay = max(int(csi_delay_slots), 0)
    if delay <= 0:
        return 0.0
    rho = min(max(float(channel_rho), 0.0), 1.0)
    return 1.0 - float(rho**delay)


def choose_adaptive_sparse_topk_v2_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    confirmed_history=None,
    channel_rho=1.0,
    csi_delay_slots=0,
    base_multiplier=None,
    expanded_multiplier=None,
    topk_fraction=None,
    margin_threshold=None,
    preview_cost=None,
    uncertainty_weight=None,
    urgency_weight=None,
    history_weight=None,
    history_window=None,
    history_prior_threshold=None,
):
    """按照adaptive、稀疏、TopK、v2、聚合反馈、决策规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if base_multiplier is None:
        base_multiplier = getattr(args, "adaptive_sparse_base_multiplier", 2.0)
    if expanded_multiplier is None:
        expanded_multiplier = getattr(args, "adaptive_sparse_expanded_multiplier", 3.0)
    if topk_fraction is None:
        topk_fraction = getattr(args, "sparse_topk_fraction", 0.75)
    if margin_threshold is None:
        margin_threshold = getattr(args, "adaptive_sparse_margin_threshold", 0.05)
    if preview_cost is None:
        preview_cost = getattr(args, "adaptive_sparse_v2_preview_cost", 0.002)
    if uncertainty_weight is None:
        uncertainty_weight = getattr(args, "adaptive_sparse_v2_uncertainty_weight", 0.02)
    if urgency_weight is None:
        urgency_weight = getattr(args, "adaptive_sparse_v2_urgency_weight", 0.5)
    if history_weight is None:
        history_weight = getattr(args, "adaptive_sparse_v2_history_weight", 0.02)
    if history_window is None:
        history_window = getattr(args, "adaptive_sparse_history_window", 3)
    if history_prior_threshold is None:
        history_prior_threshold = getattr(args, "adaptive_sparse_history_prior_threshold", 0.67)

    base_multiplier = float(base_multiplier)
    expanded_multiplier = max(float(expanded_multiplier), base_multiplier)
    topk_fraction = float(topk_fraction)
    margin_threshold = float(margin_threshold)
    preview_cost = float(preview_cost)
    uncertainty_weight = float(uncertainty_weight)
    urgency_weight = float(urgency_weight)
    history_weight = float(history_weight)
    history_window = max(int(history_window), 1)
    history_prior_threshold = float(history_prior_threshold)
    topk_budget = min(budget, max(1, int(np.ceil(topk_fraction * budget))))

    base_seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(base_multiplier * budget))),
    )
    base_seed_indices = limited.grid_indices(args.num_codebook_states, base_seed_budget, offset=slot_idx)
    base_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        base_seed_budget,
        salt=109 + slot_idx,
    )
    base_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=base_seed_indices,
        error_std=decision_error_std,
        rng=base_rng,
    )
    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in base_candidates}
    stale_margin = sparse_topk_count_margin(base_candidates, topk_budget, args.num_nodes)

    expanded_seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(expanded_multiplier * budget))),
    )
    expanded_seed_indices = limited.grid_indices(
        args.num_codebook_states,
        expanded_seed_budget,
        offset=slot_idx,
    )
    expansion_indices = [
        int(index) for index in expanded_seed_indices if int(index) not in candidate_by_index
    ]

    history_best_index, history_stability = adaptive_sparse_history_signal(
        confirmed_history or [],
        history_window,
    )
    urgency = adaptive_sparse_deadline_urgency(env, args, slot_idx, base_candidates)
    stale_uncertainty = adaptive_sparse_stale_uncertainty(channel_rho, csi_delay_slots)
    effective_threshold = (
        margin_threshold
        + uncertainty_weight * stale_uncertainty
        + urgency_weight * urgency
        - history_weight * history_stability
    )
    effective_threshold = max(0.0, float(effective_threshold))
    cost_penalty = float(preview_cost) * float(len(expansion_indices))
    expand_score = effective_threshold - float(stale_margin) - cost_penalty
    should_expand = expand_score > 0.0 and bool(expansion_indices)

    applied_multiplier = base_multiplier
    if should_expand:
        expansion_rng = limited.stable_rng(
            episode_seed,
            decision_error_std,
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            len(expansion_indices),
            salt=113 + slot_idx,
        )
        expansion_candidates = limited.estimated_preview_candidates(
            env,
            args,
            indices=expansion_indices,
            error_std=decision_error_std,
            rng=expansion_rng,
        )
        candidate_by_index.update(
            {int(candidate["irs_index"]): candidate for candidate in expansion_candidates}
        )
        applied_multiplier = expanded_multiplier

    ranked_indices = [
        int(candidate["irs_index"])
        for candidate in sorted(candidate_by_index.values(), key=limited.candidate_key, reverse=True)
    ]
    history_indices = []
    if (
        history_best_index is not None
        and history_stability >= history_prior_threshold
        and int(history_best_index) in candidate_by_index
    ):
        history_indices = [int(history_best_index)]
    rotating_indices = limited.grid_indices(args.num_codebook_states, budget, offset=slot_idx)
    selected_indices = ordered_unique_prefix(
        ranked_indices[:topk_budget] + history_indices + rotating_indices + ranked_indices,
        budget,
        args.num_codebook_states,
    )

    extra_indices = [index for index in selected_indices if int(index) not in candidate_by_index]
    if extra_indices:
        extra_rng = limited.stable_rng(
            episode_seed,
            decision_error_std,
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            len(extra_indices),
            salt=127 + slot_idx,
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
        feedback_salt=109,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["sparse_seed_multiplier"] = applied_multiplier
    decision["sparse_topk_fraction"] = topk_fraction
    decision["sparse_seed_count"] = int(len(candidate_by_index) - len(extra_indices))
    decision["sparse_topk_count"] = int(topk_budget)
    decision["sparse_extra_preview_count"] = int(len(extra_indices))
    decision["adaptive_sparse_base_multiplier"] = base_multiplier
    decision["adaptive_sparse_expanded_multiplier"] = expanded_multiplier
    decision["adaptive_sparse_margin_threshold"] = margin_threshold
    decision["adaptive_sparse_margin"] = float(stale_margin)
    decision["adaptive_sparse_expanded"] = float(should_expand)
    decision["adaptive_sparse_expansion_preview_count"] = int(len(expansion_indices) if should_expand else 0)
    decision["adaptive_sparse_effective_margin_threshold"] = float(effective_threshold)
    decision["adaptive_sparse_expand_score"] = float(expand_score)
    decision["adaptive_sparse_history_stability"] = float(history_stability)
    decision["adaptive_sparse_urgency"] = float(urgency)
    decision["adaptive_sparse_cost_penalty"] = float(cost_penalty)
    decision["adaptive_sparse_v2_preview_cost"] = float(preview_cost)
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    preview_calls = (
        len(base_seed_indices)
        + (len(expansion_indices) if should_expand else 0)
        + len(extra_indices)
        + len(selected_indices)
    )
    return decision, preview_calls, len(selected_indices)


def adaptive_sparse_history_indices(confirmed_history, window, threshold, max_count):
    """处理adaptive、稀疏、history、索引集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if max_count <= 0 or not confirmed_history:
        return [], 0.0
    tail = [int(index) for index in confirmed_history[-max(int(window), 1) :]]
    if not tail:
        return [], 0.0
    counts = {}
    for index in tail:
        counts[index] = counts.get(index, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    selected = [
        int(index)
        for index, count in ranked
        if float(count) / float(len(tail)) >= float(threshold)
    ][: max(int(max_count), 0)]
    best_stability = float(ranked[0][1]) / float(len(tail)) if ranked else 0.0
    return selected, best_stability


def local_codebook_neighbor_indices(center_indices, num_codebook_states, radius=1, max_count=2):
    """处理local、码本、邻域、索引集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if max_count <= 0 or radius <= 0:
        return []
    indices = []
    for center in center_indices:
        center = int(center) % int(num_codebook_states)
        for delta in range(1, int(radius) + 1):
            indices.append((center - delta) % int(num_codebook_states))
            indices.append((center + delta) % int(num_codebook_states))
    return ordered_unique_prefix(indices, int(max_count), int(num_codebook_states))


def choose_adaptive_sparse_topk_v3_feedback_decision(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    execution_state=None,
    confirmed_history=None,
    base_multiplier=None,
    topk_fraction=None,
    history_window=None,
    history_prior_threshold=None,
    history_count=None,
    neighbor_radius=None,
    neighbor_count=None,
):
    """按照adaptive、稀疏、TopK、v3、聚合反馈、决策规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if base_multiplier is None:
        base_multiplier = getattr(args, "adaptive_sparse_base_multiplier", 2.0)
    if topk_fraction is None:
        topk_fraction = getattr(args, "sparse_topk_fraction", 0.75)
    if history_window is None:
        history_window = getattr(args, "adaptive_sparse_history_window", 3)
    if history_prior_threshold is None:
        history_prior_threshold = getattr(args, "adaptive_sparse_history_prior_threshold", 0.67)
    if history_count is None:
        history_count = getattr(args, "adaptive_sparse_v3_history_count", 1)
    if neighbor_radius is None:
        neighbor_radius = getattr(args, "adaptive_sparse_v3_neighbor_radius", 1)
    if neighbor_count is None:
        neighbor_count = getattr(args, "adaptive_sparse_v3_neighbor_count", 2)

    base_multiplier = float(base_multiplier)
    topk_fraction = float(topk_fraction)
    history_window = max(int(history_window), 1)
    history_prior_threshold = float(history_prior_threshold)
    history_count = max(int(history_count), 0)
    neighbor_radius = max(int(neighbor_radius), 0)
    neighbor_count = max(int(neighbor_count), 0)
    topk_budget = min(budget, max(1, int(np.ceil(topk_fraction * budget))))

    base_seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(base_multiplier * budget))),
    )
    base_seed_indices = limited.grid_indices(args.num_codebook_states, base_seed_budget, offset=slot_idx)
    base_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        base_seed_budget,
        salt=137 + slot_idx,
    )
    base_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=base_seed_indices,
        error_std=decision_error_std,
        rng=base_rng,
    )
    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in base_candidates}
    ranked_indices = [
        int(candidate["irs_index"])
        for candidate in sorted(base_candidates, key=limited.candidate_key, reverse=True)
    ]
    stale_margin = sparse_topk_count_margin(base_candidates, topk_budget, args.num_nodes)

    history_indices, history_stability = adaptive_sparse_history_indices(
        confirmed_history or [],
        history_window,
        history_prior_threshold,
        history_count,
    )
    neighbor_indices = local_codebook_neighbor_indices(
        ranked_indices[: max(1, min(topk_budget, 2))],
        args.num_codebook_states,
        radius=neighbor_radius,
        max_count=neighbor_count,
    )
    rotating_indices = limited.grid_indices(args.num_codebook_states, budget, offset=slot_idx)
    top_head_count = max(1, min(topk_budget, budget - min(1, len(history_indices) + len(neighbor_indices))))
    selected_indices = ordered_unique_prefix(
        (
            ranked_indices[:top_head_count]
            + history_indices
            + neighbor_indices
            + ranked_indices[top_head_count:topk_budget]
            + rotating_indices
            + ranked_indices
        ),
        budget,
        args.num_codebook_states,
    )

    extra_indices = [index for index in selected_indices if int(index) not in candidate_by_index]
    neighbor_extra_count = len([index for index in extra_indices if int(index) in set(neighbor_indices)])
    if extra_indices:
        extra_rng = limited.stable_rng(
            episode_seed,
            decision_error_std,
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            len(extra_indices),
            salt=149 + slot_idx,
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
        feedback_salt=137,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["sparse_seed_multiplier"] = base_multiplier
    decision["sparse_topk_fraction"] = topk_fraction
    decision["sparse_seed_count"] = int(len(base_seed_indices))
    decision["sparse_topk_count"] = int(topk_budget)
    decision["sparse_extra_preview_count"] = int(len(extra_indices))
    decision["adaptive_sparse_base_multiplier"] = base_multiplier
    decision["adaptive_sparse_expanded_multiplier"] = base_multiplier
    decision["adaptive_sparse_margin_threshold"] = 0.0
    decision["adaptive_sparse_margin"] = float(stale_margin)
    decision["adaptive_sparse_expanded"] = 0.0
    decision["adaptive_sparse_effective_margin_threshold"] = 0.0
    decision["adaptive_sparse_expand_score"] = 0.0
    decision["adaptive_sparse_history_stability"] = float(history_stability)
    decision["adaptive_sparse_urgency"] = adaptive_sparse_deadline_urgency(env, args, slot_idx, base_candidates)
    decision["adaptive_sparse_cost_penalty"] = 0.0
    decision["adaptive_sparse_v3_history_prior_used"] = float(bool(history_indices))
    decision["adaptive_sparse_v3_neighbor_extra_preview_count"] = float(neighbor_extra_count)
    decision["adaptive_sparse_v3_selected_extra_preview_count"] = float(len(extra_indices))
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    preview_calls = len(base_seed_indices) + len(extra_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)
