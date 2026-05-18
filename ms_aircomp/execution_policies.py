"""Execution-mismatch policy decision functions for active probe sets."""

import numpy as np

import ms_aircomp.limited_csi as limited
from ms_aircomp.confirmation import confirm_index_with_current_feedback
from ms_aircomp.probe_sets import (
    coverage_aware_sparse_indices,
    coverage_increment_stats,
    fill_diverse_codebook_indices,
    ordered_unique_prefix,
)

__all__ = [
    "choose_active_diverse_feedback_decision",
    "choose_coverage_sparse_topk_feedback_decision",
    "choose_neighbor_coverage_sparse_topk_feedback_decision",
    "choose_no_irs_fallback",
    "choose_rotating_feedback_confirm_decision",
    "choose_sparse_topk_feedback_decision",
    "choose_stale_topk_feedback_decision",
    "fill_unpreviewed_indices",
    "neighbor_pool_indices",
]


def choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed):
    """Return the no-IRS fallback used when a probe budget is empty."""
    return limited.choose_policy_candidate(
        env,
        args,
        limited.POLICY_NO_IRS,
        0,
        slot_idx,
        decision_error_std,
        episode_seed,
    )


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
    """
    Confirm the standard rotating probe set using current aggregate feedback.

    This is the low-cost counterpart to stale-topK confirmation: it keeps the
    same rotating B indices, builds invitation masks from stale CSI, and uses
    current aggregate feedback only to choose among those B candidates.
    """
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
    """
    Pick an active stale-CSI probe set, then confirm it with current feedback.

    The top half of the budget comes from a full stale-codebook ranking and the
    rest preserves rotating-grid coverage. The final IRS index is selected from
    B aggregate current-channel feedback probes, not from full execution CSI.
    """
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
    """
    Build a low-cost active probe set from sparse stale CSI and diversity.

    The policy first previews a rotating seed set under stale/estimated CSI,
    uses the best seed candidates as anchors, fills the final set with
    codebook-diverse indices, and then selects among the final set using current
    aggregate feedback. It never ranks all C codebooks, so its preview cost is
    bounded by the seed previews, new final-set previews, and B confirmations.
    """
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
    """
    Rank a sparse stale-CSI probe pool, then confirm the final B candidates.

    This is a lower-cost approximation to stale-topK feedback. Instead of
    ranking all C codebooks, it previews a configurable sparse pool of evenly
    spaced stale candidates, keeps the best stale candidates plus a rotating
    coverage set, and selects the final IRS index using current aggregate
    feedback.
    """
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
    preview_calls = len(seed_indices) + len(extra_indices) + len(selected_indices)
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
    """
    Sparse-TopK variant that fills non-anchor probes by device coverage gain.

    It uses the same sparse stale preview pool as Sparse-TopK, but replaces the
    rotating fill step with marginal device-coverage selection within that pool.
    Final IRS selection still uses current aggregate feedback over B candidates.
    """
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
    preview_calls = len(seed_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)


def neighbor_pool_indices(center_indices, num_codebook_states, radius, max_count, exclude_indices):
    """Return wrapped local neighbors excluding already previewed indices."""
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
    """Fill a preview list with unpreviewed codebook indices."""
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
    """
    Coverage-Aware Sparse-TopK with nonuniform stale neighbor previews.

    The total stale preview budget follows ``seed_multiplier * B`` as in the
    main Coverage-Aware setting. A base evenly spaced stale pool is previewed
    first, then a small number of stale previews is reallocated to local
    neighbors around the best base stale candidates. Final B-candidate
    selection still uses the same coverage-aware fill and current aggregate
    feedback confirmation.
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
    preview_calls = len(seed_candidates) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)
