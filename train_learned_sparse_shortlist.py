"""
Train a learned sparse-shortlist ranker for temporal execution mismatch.

Training labels use hidden current-channel outcomes, but the saved model only
scores features available to the deployable shortlist policy: sparse stale
preview summaries, codebook geometry, recent confirmed IRS history, rho/delay,
and slot/deadline context.
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import numpy as np

import ms_aircomp.limited_csi as limited
from evaluate_policy_comparison import ensure_parent_dir, format_float_for_suffix
from ms_aircomp.channel_models import (
    apply_channel_state,
    build_temporal_channel_trace,
    capture_channel_state,
    delayed_channel_state,
)
from ms_aircomp.confirmation import confirm_index_with_current_feedback
from ms_aircomp.execution_candidates import (
    execution_candidate_for_decision,
    execution_candidates,
)
from ms_aircomp.execution_policies import choose_sparse_topk_feedback_decision
from ms_aircomp.learned_shortlist import (
    LEARNED_SET_SHORTLIST_FEATURE_NAMES,
    LEARNED_SHORTLIST_FEATURE_NAMES,
    learned_set_shortlist_feature_matrix,
    learned_set_shortlist_variants,
    learned_shortlist_context,
    learned_shortlist_feature_matrix,
    score_learned_shortlist_candidates,
)
from ms_aircomp.probe_sets import ordered_unique_prefix


DIAGNOSTIC_METADATA = {
    "result_role": "diagnostic",
    "uses_hidden_training_labels": "true",
    "inference_uses_hidden_current_csi": "false",
    "supervision_signal": "hidden_current_channel_supervised_targets",
}


def parse_args():
    """Parse training arguments."""
    parser = argparse.ArgumentParser(
        description="Train a linear learned sparse-shortlist ranker."
    )
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--val-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--probe-budget", type=int, default=4)
    parser.add_argument("--channel-rho-values", default="0.7,0.9,0.98")
    parser.add_argument("--csi-delay-slots", default="1,2,3")
    parser.add_argument("--decision-error-std", type=float, default=0.0)
    parser.add_argument("--execution-error-std", type=float, default=0.0)
    parser.add_argument("--confirmation-feedback-noise-std", type=float, default=0.0)
    parser.add_argument("--confirmation-feedback-power-weight", type=float, default=0.05)
    parser.add_argument("--base-multiplier", type=float, default=2.0)
    parser.add_argument("--topk-fraction", type=float, default=0.75)
    parser.add_argument(
        "--target-mode",
        choices=["absolute", "marginal", "set_value", "execution_value", "pairwise_execution"],
        default="absolute",
        help=(
            "Train on absolute hidden candidate value, marginal value after inserting the candidate, "
            "hidden value of a final B-sized shortlist set, closed-loop execution value, "
            "or pairwise execution utility differences between final-set variants."
        ),
    )
    parser.add_argument(
        "--target-extra-count",
        type=int,
        default=1,
        help="Number of learned extra slots assumed for marginal labels or max extras for set-value labels.",
    )
    parser.add_argument("--label-power-weight", type=float, default=0.05)
    parser.add_argument("--label-failure-weight", type=float, default=0.5)
    parser.add_argument("--label-missed-weight", type=float, default=0.5)
    parser.add_argument(
        "--label-preview-cost",
        type=float,
        default=0.0,
        help="Utility penalty per extra stale preview used by pairwise execution labels.",
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--fixed-irs-index", type=int, default=7)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def validate_args(args):
    """Validate and parse list-like options."""
    for name in (
        "train_episodes",
        "val_episodes",
        "probe_budget",
        "num_nodes",
        "num_slots",
        "num_irs_elements",
        "num_codebook_states",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")
    if args.decision_error_std < 0.0 or args.execution_error_std < 0.0:
        raise ValueError("error std values must be non-negative")
    if args.confirmation_feedback_noise_std < 0.0:
        raise ValueError("--confirmation-feedback-noise-std must be non-negative")
    if args.confirmation_feedback_power_weight < 0.0:
        raise ValueError("--confirmation-feedback-power-weight must be non-negative")
    if args.base_multiplier <= 0.0:
        raise ValueError("--base-multiplier must be positive")
    if args.topk_fraction <= 0.0 or args.topk_fraction > 1.0:
        raise ValueError("--topk-fraction must be in (0, 1]")
    if args.target_extra_count < 0:
        raise ValueError("--target-extra-count must be non-negative")
    if args.label_power_weight < 0.0:
        raise ValueError("--label-power-weight must be non-negative")
    if args.label_failure_weight < 0.0:
        raise ValueError("--label-failure-weight must be non-negative")
    if args.label_missed_weight < 0.0:
        raise ValueError("--label-missed-weight must be non-negative")
    if args.label_preview_cost < 0.0:
        raise ValueError("--label-preview-cost must be non-negative")
    if args.ridge < 0.0:
        raise ValueError("--ridge must be non-negative")
    if args.g_th <= 0.0 or args.alpha_th <= 0.0:
        raise ValueError("--g-th and --alpha-th must be positive")

    args.channel_rho_values = limited.parse_float_list(args.channel_rho_values)
    args.csi_delay_slots = limited.parse_int_list(args.csi_delay_slots)
    if not args.channel_rho_values or any(value < 0.0 or value > 1.0 for value in args.channel_rho_values):
        raise ValueError("--channel-rho-values must be non-empty and in [0, 1]")
    if not args.csi_delay_slots or any(value < 0 for value in args.csi_delay_slots):
        raise ValueError("--csi-delay-slots must be non-empty and non-negative")
    args.probe_budget = min(int(args.probe_budget), int(args.num_codebook_states))
    args.target_extra_count = min(int(args.target_extra_count), max(args.probe_budget - 1, 0))
    args.fixed_irs_index = int(np.clip(args.fixed_irs_index, 0, args.num_codebook_states - 1))


def resolve_output_prefix(args):
    """Resolve output prefix for model and diagnostics."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    rho_label = "-".join(format_float_for_suffix(value) for value in args.channel_rho_values)
    delay_label = "-".join(str(value) for value in args.csi_delay_slots)
    suffix = (
        f"train{args.train_episodes}_val{args.val_episodes}_seed{args.seed}_"
        f"rho{rho_label}_delay{delay_label}_b{args.probe_budget}_"
        f"bm{format_float_for_suffix(args.base_multiplier)}_tf{format_float_for_suffix(args.topk_fraction)}"
        f"_{args.target_mode}_tex{args.target_extra_count}"
    )
    output_prefix = os.path.join(
        "results",
        "execution_mismatch",
        f"learned_sparse_shortlist_{suffix}",
    )
    ensure_parent_dir(output_prefix)
    return output_prefix


def split_episode_specs(args, episodes, seed_offset):
    """Return deterministic episode seeds and scenario choices."""
    rng = np.random.default_rng(int(args.seed) + int(seed_offset))
    specs = []
    for _ in range(int(episodes)):
        specs.append(
            {
                "episode_seed": int(rng.integers(0, 2**31 - 1)),
                "rho": float(rng.choice(args.channel_rho_values)),
                "delay": int(rng.choice(args.csi_delay_slots)),
            }
        )
    return specs


def hidden_scores_for_indices(env, args, indices, execution_state, slot_idx):
    """Compute hidden current-channel labels for candidate indices."""
    snapshot = capture_channel_state(env)
    try:
        apply_channel_state(env, execution_state)
        candidates = execution_candidates(
            env,
            args,
            indices=indices,
            execution_error_std=args.execution_error_std,
            slot_idx=slot_idx,
        )
    finally:
        apply_channel_state(env, snapshot)
    return np.asarray(
        [
            float(candidate["tx_this_slot"]) / max(float(args.num_nodes), 1.0)
            - float(args.label_power_weight) * float(candidate["power_avg"])
            for candidate in candidates
        ],
        dtype=float,
    )


def hidden_score_by_index(env, args, execution_state, slot_idx):
    """Return hidden current-channel score for every codebook index."""
    indices = list(range(args.num_codebook_states))
    scores = hidden_scores_for_indices(env, args, indices, execution_state, slot_idx)
    return {int(index): float(score) for index, score in zip(indices, scores)}


def selected_hidden_value(indices, score_by_index):
    """Return the hidden best-confirmation value for a candidate set."""
    values = [float(score_by_index[int(index)]) for index in indices if int(index) in score_by_index]
    return max(values) if values else 0.0


def marginal_labels(args, context, candidate_indices, score_by_index):
    """Label each extra candidate by marginal hidden value after forced insertion."""
    budget = int(context["budget"])
    topk_budget = int(context["topk_budget"])
    ranked_indices = list(context["ranked_indices"])
    rotating_indices = list(context["rotating_indices"])
    baseline_indices = ordered_unique_prefix(
        ranked_indices[:topk_budget] + rotating_indices + ranked_indices,
        budget,
        args.num_codebook_states,
    )
    baseline_value = selected_hidden_value(baseline_indices, score_by_index)
    assumed_extra_count = min(max(int(args.target_extra_count), 1), max(budget - 1, 1))
    base_head_count = max(1, min(topk_budget, budget - assumed_extra_count))
    labels = []
    for candidate_index in candidate_indices:
        selected_indices = ordered_unique_prefix(
            (
                ranked_indices[:base_head_count]
                + [int(candidate_index)]
                + rotating_indices
                + ranked_indices[base_head_count:]
            ),
            budget,
            args.num_codebook_states,
        )
        labels.append(selected_hidden_value(selected_indices, score_by_index) - baseline_value)
    return np.asarray(labels, dtype=float)


def set_value_labels(variants, score_by_index):
    """Label each final-set variant by hidden best-confirmation value."""
    return np.asarray(
        [
            selected_hidden_value(variant["selected_indices"], score_by_index)
            for variant in variants
        ],
        dtype=float,
    )


def decision_candidates_for_selected_indices(env, args, context, selected_indices, slot_idx, episode_seed):
    """Return stale/estimated decision candidates for all selected indices."""
    candidate_by_index = dict(context["candidate_by_index"])
    missing_indices = [
        int(index)
        for index in selected_indices
        if int(index) not in candidate_by_index
    ]
    if missing_indices:
        extra_rng = limited.stable_rng(
            episode_seed,
            args.decision_error_std,
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            len(missing_indices),
            salt=191 + int(slot_idx),
        )
        extra_candidates = limited.estimated_preview_candidates(
            env,
            args,
            indices=missing_indices,
            error_std=args.decision_error_std,
            rng=extra_rng,
        )
        candidate_by_index.update({int(candidate["irs_index"]): candidate for candidate in extra_candidates})
    return candidate_by_index


def execution_candidate_for_index(env, args, index, execution_state, slot_idx):
    """Return hidden execution candidate for one IRS index without changing decision CSI."""
    snapshot = capture_channel_state(env)
    try:
        apply_channel_state(env, execution_state)
        return execution_candidates(
            env,
            args,
            indices=[int(index)],
            execution_error_std=args.execution_error_std,
            slot_idx=slot_idx,
        )[0]
    finally:
        apply_channel_state(env, snapshot)


def execution_value_for_variant(env, args, context, variant, execution_state, slot_idx, episode_seed):
    """Label one final set by closed-loop confirmation and execution outcome."""
    selected_indices = [int(index) for index in variant["selected_indices"]]
    if not selected_indices:
        return 0.0
    decision_by_index = decision_candidates_for_selected_indices(
        env,
        args,
        context,
        selected_indices,
        slot_idx,
        episode_seed,
    )
    confirmed_index, _feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        args.execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=191,
    )
    decision_candidate = decision_by_index[int(confirmed_index)]
    execution_candidate = execution_candidate_for_index(
        env,
        args,
        confirmed_index,
        execution_state,
        slot_idx,
    )

    remaining = ~env.transmitted_flags
    scheduled_mask = np.asarray(decision_candidate["valid_mask"], dtype=bool) & remaining
    true_valid_mask = np.asarray(execution_candidate["valid_mask"], dtype=bool) & remaining
    success_count = float(np.sum(scheduled_mask & true_valid_mask))
    failed_count = float(np.sum(scheduled_mask & (~true_valid_mask)))
    missed_count = float(np.sum(true_valid_mask & (~scheduled_mask)))
    scheduled_power = np.asarray(decision_candidate["required_power"], dtype=float)[scheduled_mask]
    power_avg = float(np.mean(scheduled_power)) if scheduled_power.size else 0.0
    normalizer = max(float(args.num_nodes), 1.0)
    return (
        success_count / normalizer
        - float(args.label_failure_weight) * failed_count / normalizer
        - float(args.label_missed_weight) * missed_count / normalizer
        - float(args.label_power_weight) * power_avg
    )


def execution_value_labels(env, args, context, variants, execution_state, slot_idx, episode_seed):
    """Label final-set variants by closed-loop execution value."""
    return np.asarray(
        [
            execution_value_for_variant(
                env,
                args,
                context,
                variant,
                execution_state,
                slot_idx,
                episode_seed,
            )
            for variant in variants
        ],
        dtype=float,
    )


def variant_extra_preview_counts(context, variants):
    """Return deploy-time extra stale preview counts for final-set variants."""
    candidate_indices = set(int(index) for index in context["candidate_by_index"])
    return np.asarray(
        [
            len(
                [
                    int(index)
                    for index in variant["selected_indices"]
                    if int(index) not in candidate_indices
                ]
            )
            for variant in variants
        ],
        dtype=float,
    )


def cost_aware_scores(args, context, variants, scores):
    """Apply optional per-extra-preview cost to variant utility labels."""
    if float(args.label_preview_cost) <= 0.0:
        return np.asarray(scores, dtype=float)
    return (
        np.asarray(scores, dtype=float)
        - float(args.label_preview_cost) * variant_extra_preview_counts(context, variants)
    )


def append_episode_rows(env, args, spec, feature_rows, labels, group_ids):
    """Collect training rows from one temporally correlated episode."""
    episode_seed = int(spec["episode_seed"])
    channel_rho = float(spec["rho"])
    csi_delay_slots = int(spec["delay"])
    env.reset(seed=episode_seed)
    env._last_seed = episode_seed
    temporal_history, temporal_states = build_temporal_channel_trace(
        env,
        args,
        episode_seed,
        channel_rho,
        prehistory_slots=csi_delay_slots,
    )
    confirmed_history = []

    for slot_idx in range(args.num_slots):
        execution_state = temporal_states[min(slot_idx, len(temporal_states) - 1)]
        decision_state = delayed_channel_state(
            temporal_states,
            slot_idx,
            csi_delay_slots,
            history_states=temporal_history,
        )
        apply_channel_state(env, decision_state)
        context = learned_shortlist_context(
            env,
            args,
            args.probe_budget,
            slot_idx,
            args.decision_error_std,
            episode_seed,
            base_multiplier=args.base_multiplier,
            topk_fraction=args.topk_fraction,
            confirmed_history=confirmed_history,
            channel_rho=channel_rho,
            csi_delay_slots=csi_delay_slots,
        )
        if args.target_mode in {"set_value", "execution_value", "pairwise_execution"}:
            variants = learned_set_shortlist_variants(
                args,
                context,
                args.target_extra_count,
            )
            features = learned_set_shortlist_feature_matrix(
                args,
                variants,
                slot_idx,
                context,
            )
            if args.target_mode in {"execution_value", "pairwise_execution"}:
                scores = execution_value_labels(
                    env,
                    args,
                    context,
                    variants,
                    execution_state,
                    slot_idx,
                    episode_seed,
                )
                if args.target_mode == "pairwise_execution":
                    scores = cost_aware_scores(args, context, variants, scores)
            else:
                scores = set_value_labels(
                    variants,
                    hidden_score_by_index(env, args, execution_state, slot_idx),
                )
        else:
            candidate_indices = [
                index
                for index in range(args.num_codebook_states)
                if index not in set(context["base_seed_indices"])
            ]
            features = learned_shortlist_feature_matrix(
                args,
                candidate_indices,
                slot_idx,
                context,
            )
            if args.target_mode == "marginal":
                scores = marginal_labels(
                    args,
                    context,
                    candidate_indices,
                    hidden_score_by_index(env, args, execution_state, slot_idx),
                )
            else:
                scores = hidden_scores_for_indices(env, args, candidate_indices, execution_state, slot_idx)
        feature_rows.append(features)
        labels.append(scores)
        group_ids.extend([len(group_ids)] * len(scores))

        decision, _preview_calls, _candidate_count = choose_sparse_topk_feedback_decision(
            env,
            args,
            args.probe_budget,
            slot_idx,
            args.decision_error_std,
            args.execution_error_std,
            episode_seed,
            execution_state=execution_state,
            seed_multiplier=args.base_multiplier,
            topk_fraction=args.topk_fraction,
        )
        apply_channel_state(env, execution_state)
        true_selected = execution_candidate_for_decision(
            env,
            args,
            decision,
            args.execution_error_std,
            slot_idx,
        )
        _info, done = limited.execute_limited_csi_slot(env, args, decision, true_selected)
        if int(decision.get("confirmed_irs_index", decision.get("irs_index", -1))) >= 0:
            confirmed_history.append(
                int(decision.get("confirmed_irs_index", decision.get("irs_index", -1)))
            )
        if done:
            break


def build_dataset(args, episodes, seed_offset):
    """Build feature matrix, labels, and group ids for one split."""
    env = limited.make_env(args)
    feature_rows = []
    labels = []
    group_ids = []
    for spec in split_episode_specs(args, episodes, seed_offset):
        append_episode_rows(env, args, spec, feature_rows, labels, group_ids)
    if not feature_rows:
        raise RuntimeError("No dataset rows were generated")
    return (
        np.vstack(feature_rows).astype(float),
        np.concatenate(labels).astype(float),
        np.asarray(group_ids, dtype=int),
    )


def fit_ridge_linear(features, labels, ridge):
    """Fit a standardized linear ridge regressor."""
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.where(feature_scale < 1e-12, 1.0, feature_scale)
    normalized = (features - feature_mean) / feature_scale
    design = np.column_stack([normalized, np.ones(len(normalized))])
    penalty = np.eye(design.shape[1], dtype=float) * float(ridge)
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ labels)
    return {
        "weights": solution[:-1],
        "bias": float(solution[-1]),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "fit_objective": "scalar",
    }


def pairwise_difference_dataset(features, labels, group_ids, min_gap=1e-12):
    """Build best-vs-rest feature differences for pairwise utility training."""
    diff_rows = []
    diff_labels = []
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        group_indices = np.flatnonzero(mask)
        if len(group_indices) < 2:
            continue
        group_labels = labels[group_indices]
        best_local_idx = int(np.argmax(group_labels))
        best_index = int(group_indices[best_local_idx])
        best_label = float(labels[best_index])
        for index in group_indices:
            index = int(index)
            if index == best_index:
                continue
            gap = best_label - float(labels[index])
            if gap <= min_gap:
                continue
            diff = features[best_index] - features[index]
            diff_rows.append(diff)
            diff_labels.append(gap)
            diff_rows.append(-diff)
            diff_labels.append(-gap)
    if not diff_rows:
        return (
            np.zeros((0, features.shape[1]), dtype=float),
            np.zeros((0,), dtype=float),
        )
    return np.vstack(diff_rows).astype(float), np.asarray(diff_labels, dtype=float)


def fit_pairwise_ridge_linear(features, labels, group_ids, ridge):
    """Fit a linear ranker from pairwise utility differences."""
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.where(feature_scale < 1e-12, 1.0, feature_scale)
    diff_features, diff_labels = pairwise_difference_dataset(features, labels, group_ids)
    if len(diff_labels) == 0:
        return fit_ridge_linear(features, labels, ridge)
    normalized_diff = diff_features / feature_scale
    penalty = np.eye(normalized_diff.shape[1], dtype=float) * float(ridge)
    weights = np.linalg.solve(
        normalized_diff.T @ normalized_diff + penalty,
        normalized_diff.T @ diff_labels,
    )
    normalized_features = (features - feature_mean) / feature_scale
    bias = float(np.mean(labels - normalized_features @ weights))
    return {
        "weights": weights,
        "bias": bias,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "fit_objective": "pairwise",
        "pairwise_rows": int(len(diff_labels)),
    }


def predict(model, features):
    """Predict labels from model and features."""
    return score_learned_shortlist_candidates(model, features)


def grouped_top1_regret(labels, predictions, group_ids):
    """Mean per-state regret from picking the predicted top extra candidate."""
    regrets = []
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        group_labels = labels[mask]
        group_preds = predictions[mask]
        best_label = float(np.max(group_labels))
        chosen_label = float(group_labels[int(np.argmax(group_preds))])
        regrets.append(best_label - chosen_label)
    return float(np.mean(regrets)) if regrets else 0.0


def metrics(model, features, labels, group_ids):
    """Compute split diagnostics."""
    predictions = predict(model, features)
    return {
        "rows": int(len(labels)),
        "mse": float(np.mean((predictions - labels) ** 2)),
        "corr": float(np.corrcoef(predictions, labels)[0, 1]) if len(labels) > 1 else 0.0,
        "top1_regret": grouped_top1_regret(labels, predictions, group_ids),
    }


def write_diagnostics(path, train_metrics, val_metrics):
    """Write train/validation diagnostics."""
    rows = []
    for split, values in (("train", train_metrics), ("val", val_metrics)):
        row = {"split": split, **DIAGNOSTIC_METADATA}
        row.update(values)
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "split",
                "result_role",
                "uses_hidden_training_labels",
                "inference_uses_hidden_current_csi",
                "supervision_signal",
                "rows",
                "mse",
                "corr",
                "top1_regret",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_model(path, args, model):
    """Save model arrays and metadata."""
    feature_names = (
        LEARNED_SET_SHORTLIST_FEATURE_NAMES
        if args.target_mode in {"set_value", "execution_value", "pairwise_execution"}
        else LEARNED_SHORTLIST_FEATURE_NAMES
    )
    np.savez(
        path,
        weights=np.asarray(model["weights"], dtype=float),
        bias=np.asarray(model["bias"], dtype=float),
        feature_mean=np.asarray(model["feature_mean"], dtype=float),
        feature_scale=np.asarray(model["feature_scale"], dtype=float),
        feature_names=np.asarray(feature_names),
        probe_budget=np.asarray(args.probe_budget),
        base_multiplier=np.asarray(args.base_multiplier),
        topk_fraction=np.asarray(args.topk_fraction),
        target_mode=np.asarray(args.target_mode),
        target_extra_count=np.asarray(args.target_extra_count),
        label_power_weight=np.asarray(args.label_power_weight),
        label_failure_weight=np.asarray(args.label_failure_weight),
        label_missed_weight=np.asarray(args.label_missed_weight),
        label_preview_cost=np.asarray(args.label_preview_cost),
        fit_objective=np.asarray(model.get("fit_objective", "scalar")),
        pairwise_rows=np.asarray(model.get("pairwise_rows", 0)),
        result_role=np.asarray(DIAGNOSTIC_METADATA["result_role"]),
        uses_hidden_training_labels=np.asarray(DIAGNOSTIC_METADATA["uses_hidden_training_labels"]),
        inference_uses_hidden_current_csi=np.asarray(
            DIAGNOSTIC_METADATA["inference_uses_hidden_current_csi"]
        ),
        supervision_signal=np.asarray(DIAGNOSTIC_METADATA["supervision_signal"]),
    )
    print(f"Saved: {path}")


def main():
    """Train and save the learned sparse shortlist ranker."""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)

    print("=" * 96)
    print(
        f"Learned sparse shortlist training: train={args.train_episodes}, "
        f"val={args.val_episodes}, rhos={args.channel_rho_values}, "
        f"delays={args.csi_delay_slots}, budget={args.probe_budget}, "
        f"base_multiplier={args.base_multiplier:g}, topk_fraction={args.topk_fraction:g}, "
        f"target_mode={args.target_mode}, target_extra_count={args.target_extra_count}"
    )
    print(f"Output prefix: {output_prefix}")
    print("=" * 96)

    train_x, train_y, train_groups = build_dataset(args, args.train_episodes, seed_offset=0)
    val_x, val_y, val_groups = build_dataset(args, args.val_episodes, seed_offset=10_000)
    if args.target_mode == "pairwise_execution":
        model = fit_pairwise_ridge_linear(train_x, train_y, train_groups, args.ridge)
    else:
        model = fit_ridge_linear(train_x, train_y, args.ridge)
    train_metrics = metrics(model, train_x, train_y, train_groups)
    val_metrics = metrics(model, val_x, val_y, val_groups)

    if model.get("fit_objective") == "pairwise":
        print(f"Pairwise training rows: {int(model.get('pairwise_rows', 0))}")
    print(
        "Diagnostics: "
        f"train_mse={train_metrics['mse']:.6f}, val_mse={val_metrics['mse']:.6f}, "
        f"val_corr={val_metrics['corr']:.3f}, val_top1_regret={val_metrics['top1_regret']:.4f}"
    )
    save_model(f"{output_prefix}_model.npz", args, model)
    write_diagnostics(f"{output_prefix}_diagnostics.csv", train_metrics, val_metrics)


if __name__ == "__main__":
    main()
