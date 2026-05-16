"""Probe-set and coverage-aware candidate selection helpers."""

import numpy as np

import evaluate_limited_csi_ms_aircomp as limited

__all__ = [
    "circular_codebook_distance",
    "coverage_aware_sparse_indices",
    "coverage_increment_stats",
    "fill_diverse_codebook_indices",
    "ordered_unique_prefix",
]


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


def circular_codebook_distance(index, other, num_codebook_states):
    """Circular distance between two DFT codebook indices."""
    if num_codebook_states <= 1:
        return 0
    diff = abs(int(index) - int(other)) % int(num_codebook_states)
    return min(diff, int(num_codebook_states) - diff)


def fill_diverse_codebook_indices(priority_indices, budget, num_codebook_states):
    """
    Fill a probe set with codebook-diverse indices after priority candidates.

    Diversity uses only codebook geometry, not hidden execution CSI. This keeps
    the candidate-generation step cheap while avoiding probe sets that cluster
    around one stale-CSI winner.
    """
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
    """Return average marginal coverage and overlap for selected stale candidates."""
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
    """
    Select sparse stale candidates with marginal device-coverage awareness.

    The highest stale-score candidates are kept as anchors. Remaining probe
    slots are filled greedily from the same sparse stale pool by combining stale
    tx fraction, marginal device coverage over the already selected anchors,
    and a small stale power penalty.
    """
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
