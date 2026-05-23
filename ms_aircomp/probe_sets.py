"""提供 probe-set 构造和 coverage-aware candidate selection 工具，控制候选多样性和预算。"""

import numpy as np

import ms_aircomp.limited_csi as limited

__all__ = [
    "POSTERIOR_PROBE_OBJECTIVE_EXPECTED_BEST",
    "POSTERIOR_PROBE_OBJECTIVE_EXPECTED_BEST_PLUS_COVERAGE",
    "POSTERIOR_PROBE_OBJECTIVE_PLUS_COVERAGE",
    "VALID_POSTERIOR_PROBE_OBJECTIVES",
    "circular_codebook_distance",
    "coverage_aware_sparse_indices",
    "coverage_increment_stats",
    "fill_diverse_codebook_indices",
    "ordered_unique_prefix",
    "posterior_greedy_probe_indices",
    "posterior_probe_objective",
    "validate_posterior_probe_objective",
]

POSTERIOR_PROBE_OBJECTIVE_EXPECTED_BEST = "expected_best_count"
POSTERIOR_PROBE_OBJECTIVE_PLUS_COVERAGE = "expected_best_count_plus_coverage"
POSTERIOR_PROBE_OBJECTIVE_EXPECTED_BEST_PLUS_COVERAGE = POSTERIOR_PROBE_OBJECTIVE_PLUS_COVERAGE
VALID_POSTERIOR_PROBE_OBJECTIVES = (
    POSTERIOR_PROBE_OBJECTIVE_EXPECTED_BEST,
    POSTERIOR_PROBE_OBJECTIVE_PLUS_COVERAGE,
)


def validate_posterior_probe_objective(objective):
    """Normalize and validate the posterior-greedy probe objective."""
    objective = str(objective).strip()
    if objective not in VALID_POSTERIOR_PROBE_OBJECTIVES:
        valid = ", ".join(VALID_POSTERIOR_PROBE_OBJECTIVES)
        raise ValueError(f"unknown posterior probe objective {objective!r}; expected one of: {valid}")
    return objective


def ordered_unique_prefix(indices, budget, num_codebook_states):
    """处理ordered、去重、前缀相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
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


def circular_codebook_distance(index, other, num_codebook_states):
    """处理circular、码本、distance相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if num_codebook_states <= 1:
        return 0
    diff = abs(int(index) - int(other)) % int(num_codebook_states)
    return min(diff, int(num_codebook_states) - diff)


def fill_diverse_codebook_indices(priority_indices, budget, num_codebook_states):
    """处理补齐、多样性、码本、索引集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    selected = ordered_unique_prefix(priority_indices, budget, num_codebook_states)
    if not selected and num_codebook_states > 0:
        selected.append(0)

    seen = set(selected)
    while len(selected) < min(int(budget), int(num_codebook_states)):
        best_index = max(
            (index for index in range(num_codebook_states) if index not in seen),
            key=lambda index: (
                min(circular_codebook_distance(index, chosen, num_codebook_states) for chosen in selected),
                -index,
            ),
        )
        selected.append(int(best_index))
        seen.add(int(best_index))
    return selected


def coverage_increment_stats(candidates_by_index, selected_indices, num_nodes):
    """处理覆盖感知、increment、stats相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    covered_mask = np.zeros(int(num_nodes), dtype=bool)
    marginal_fractions = []
    overlap_fractions = []
    for index in selected_indices:
        candidate = candidates_by_index[int(index)]
        candidate_mask = np.asarray(candidate["valid_mask"], dtype=bool)
        marginal_count = int(np.sum(candidate_mask & (~covered_mask)))
        overlap_count = int(np.sum(candidate_mask & covered_mask))
        tx_count = max(int(candidate["tx_this_slot"]), 1)
        marginal_fractions.append(marginal_count / max(float(num_nodes), 1.0))
        overlap_fractions.append(overlap_count / float(tx_count))
        covered_mask |= candidate_mask
    return (
        float(np.mean(marginal_fractions)) if marginal_fractions else 0.0,
        float(np.mean(overlap_fractions)) if overlap_fractions else 0.0,
    )


def coverage_aware_sparse_indices(
    seed_candidates,
    args,
    budget,
    topk_fraction,
    coverage_weight,
    power_weight,
):
    """处理覆盖感知、aware、稀疏、索引集合相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    ranked_candidates = sorted(seed_candidates, key=limited.candidate_key, reverse=True)
    if not ranked_candidates:
        return [], 0, 0.0, 0.0

    budget = min(int(budget), len(ranked_candidates), int(args.num_codebook_states))
    anchor_count = min(budget, max(1, int(np.ceil(float(topk_fraction) * budget))))
    selected = list(ranked_candidates[:anchor_count])
    selected_indices = [int(candidate["irs_index"]) for candidate in selected]
    selected_set = set(selected_indices)
    covered_mask = np.zeros(int(args.num_nodes), dtype=bool)
    for candidate in selected:
        covered_mask |= np.asarray(candidate["valid_mask"], dtype=bool)

    remaining = [
        candidate
        for candidate in ranked_candidates[anchor_count:]
        if int(candidate["irs_index"]) not in selected_set
    ]
    while len(selected_indices) < budget and remaining:
        def coverage_key(candidate):
            candidate_mask = np.asarray(candidate["valid_mask"], dtype=bool)
            marginal_count = int(np.sum(candidate_mask & (~covered_mask)))
            overlap_count = int(np.sum(candidate_mask & covered_mask))
            tx_fraction = float(candidate["tx_this_slot"]) / max(float(args.num_nodes), 1.0)
            marginal_fraction = marginal_count / max(float(args.num_nodes), 1.0)
            overlap_fraction = overlap_count / max(float(args.num_nodes), 1.0)
            power_penalty = float(power_weight) * float(candidate["power_avg"])
            score = tx_fraction + float(coverage_weight) * marginal_fraction - power_penalty
            return (
                score,
                marginal_fraction,
                tx_fraction,
                -overlap_fraction,
                -float(candidate["power_avg"]) if int(candidate["tx_this_slot"]) > 0 else 0.0,
                float(candidate["mean_gain_remaining"]),
                -int(candidate["irs_index"]),
            )

        best_candidate = max(remaining, key=coverage_key)
        remaining = [
            candidate
            for candidate in remaining
            if int(candidate["irs_index"]) != int(best_candidate["irs_index"])
        ]
        selected.append(best_candidate)
        selected_indices.append(int(best_candidate["irs_index"]))
        selected_set.add(int(best_candidate["irs_index"]))
        covered_mask |= np.asarray(best_candidate["valid_mask"], dtype=bool)

    candidates_by_index = {int(candidate["irs_index"]): candidate for candidate in seed_candidates}
    marginal_mean, overlap_mean = coverage_increment_stats(
        candidates_by_index,
        selected_indices,
        args.num_nodes,
    )
    return selected_indices, anchor_count, marginal_mean, overlap_mean


def _posterior_probe_rng(episode_seed, slot_idx, seed_offset):
    if episode_seed is None:
        return np.random.default_rng()
    seed = (
        int(episode_seed)
        + int(seed_offset)
        + 0xB7E15162
        + int(slot_idx) * 0x9E3779B1
    ) % (2**32)
    return np.random.default_rng(seed)


def _selected_coverage(probabilities, selected_indices):
    probabilities = np.asarray(probabilities, dtype=float)
    if len(selected_indices) == 0 or probabilities.size == 0:
        return 0.0
    miss_probability = np.prod(1.0 - probabilities[:, selected_indices], axis=1)
    return float(np.sum(1.0 - miss_probability))


def posterior_probe_objective(probabilities, selected_indices, sample_counts=None, beta=0.0):
    """Evaluate E[max_c Y_c] + beta * coverage for a selected IRS set."""
    probabilities = np.asarray(probabilities, dtype=float)
    selected_indices = [int(index) for index in selected_indices]
    if not selected_indices:
        return {
            "objective_value": 0.0,
            "expected_best_count": 0.0,
            "coverage_score": 0.0,
            "expected_count_per_selected_state": [],
        }
    expected_counts = np.sum(probabilities, axis=0)
    if sample_counts is None:
        expected_best_count = float(np.max(expected_counts[selected_indices]))
    else:
        counts = np.asarray(sample_counts, dtype=float)
        expected_best_count = float(np.mean(np.max(counts[:, selected_indices], axis=1)))
    coverage_score = _selected_coverage(probabilities, selected_indices)
    return {
        "objective_value": float(expected_best_count + float(beta) * coverage_score),
        "expected_best_count": expected_best_count,
        "coverage_score": coverage_score,
        "expected_count_per_selected_state": [
            float(expected_counts[index]) for index in selected_indices
        ],
    }


def posterior_greedy_probe_indices(
    probabilities,
    budget,
    *,
    sample_count=128,
    beta=0.0,
    objective=POSTERIOR_PROBE_OBJECTIVE_EXPECTED_BEST,
    seed_offset=0,
    episode_seed=None,
    slot_idx=0,
    candidate_prefilter_size=0,
):
    """Greedily select IRS probes using posterior samples of feasible counts.

    The sample matrix stores Y_c for each posterior draw, so each greedy step
    only updates the sampled best-count vector. When `candidate_prefilter_size`
    is positive, selection is restricted to the top expected-count states.
    """
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape K x C")
    num_states = int(probabilities.shape[1])
    budget = min(max(int(budget), 0), num_states)
    if budget == 0:
        return {
            "selected_indices": [],
            "candidate_prefilter_indices": [],
            "objective_value": 0.0,
            "expected_best_count": 0.0,
            "coverage_score": 0.0,
            "expected_count_per_selected_state": [],
        }
    objective = validate_posterior_probe_objective(objective)

    expected_counts = np.sum(probabilities, axis=0)
    prefilter_size = int(candidate_prefilter_size)
    if prefilter_size <= 0 or prefilter_size > num_states:
        prefilter_size = num_states
    prefilter_size = max(prefilter_size, budget)
    ranked_prefilter = sorted(
        range(num_states),
        key=lambda index: (float(expected_counts[index]), -int(index)),
        reverse=True,
    )[:prefilter_size]
    prefilter_probs = probabilities[:, ranked_prefilter]
    rng = _posterior_probe_rng(episode_seed, slot_idx, seed_offset)
    draws = rng.random((max(int(sample_count), 1), probabilities.shape[0], prefilter_size))
    sample_counts = np.sum(draws < prefilter_probs[np.newaxis, :, :], axis=1)

    selected_positions = []
    selected_indices = []
    current_best = np.zeros(sample_counts.shape[0], dtype=float)
    miss_probability = np.ones(probabilities.shape[0], dtype=float)
    remaining_positions = set(range(prefilter_size))
    use_coverage = objective == POSTERIOR_PROBE_OBJECTIVE_PLUS_COVERAGE
    while len(selected_indices) < budget and remaining_positions:
        def greedy_key(position):
            candidate_best = np.maximum(current_best, sample_counts[:, position])
            expected_best = float(np.mean(candidate_best))
            if use_coverage:
                coverage = float(
                    np.sum(1.0 - miss_probability * (1.0 - prefilter_probs[:, position]))
                )
            else:
                coverage = 0.0
            original_index = int(ranked_prefilter[position])
            return (
                expected_best + float(beta) * coverage,
                expected_best,
                coverage,
                float(expected_counts[original_index]),
                -original_index,
            )

        best_position = max(remaining_positions, key=greedy_key)
        remaining_positions.remove(best_position)
        selected_positions.append(int(best_position))
        selected_indices.append(int(ranked_prefilter[best_position]))
        current_best = np.maximum(current_best, sample_counts[:, best_position])
        miss_probability *= 1.0 - prefilter_probs[:, best_position]

    full_sample_counts = np.zeros((sample_counts.shape[0], num_states), dtype=float)
    full_sample_counts[:, ranked_prefilter] = sample_counts
    details = posterior_probe_objective(
        probabilities,
        selected_indices,
        sample_counts=full_sample_counts,
        beta=float(beta) if use_coverage else 0.0,
    )
    details.update(
        {
            "selected_indices": selected_indices,
            "candidate_prefilter_indices": [int(index) for index in ranked_prefilter],
            "posterior_probe_candidate_prefilter_size": int(prefilter_size),
            "posterior_probe_samples": int(sample_counts.shape[0]),
            "posterior_probe_beta": float(beta),
            "posterior_probe_objective": objective,
        }
    )
    return details
