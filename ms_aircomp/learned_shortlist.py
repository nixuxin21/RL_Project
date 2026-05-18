"""实现 learned sparse/set shortlist 的特征构造、模型加载和闭环策略选择。"""

import itertools

import numpy as np

import ms_aircomp.limited_csi as limited
from ms_aircomp.adaptive_sparse_policies import (
    adaptive_sparse_history_signal,
    sparse_topk_count_margin,
)
from ms_aircomp.confirmation import confirm_index_with_current_feedback
from ms_aircomp.execution_policies import choose_no_irs_fallback
from ms_aircomp.probe_sets import circular_codebook_distance, ordered_unique_prefix

LEARNED_SHORTLIST_FEATURE_NAMES = (
    "index_sin",
    "index_cos",
    "slot_fraction",
    "remaining_slot_fraction",
    "rho_delay",
    "stale_uncertainty",
    "top_tx_fraction",
    "kth_tx_fraction",
    "margin_fraction",
    "top_power_avg",
    "top_mean_gain_remaining",
    "distance_to_top1",
    "distance_to_top2",
    "distance_to_top3",
    "distance_to_seed_set",
    "distance_to_rotating_set",
    "distance_to_history_winner",
    "is_history_winner",
    "history_stability",
    "is_rotating_candidate",
)

LEARNED_SET_SHORTLIST_FEATURE_NAMES = tuple(
    f"mean_{name}" for name in LEARNED_SHORTLIST_FEATURE_NAMES
) + tuple(
    f"max_{name}" for name in LEARNED_SHORTLIST_FEATURE_NAMES
) + (
    "selected_extra_fraction",
    "selected_seed_fraction",
    "selected_rotating_fraction",
    "selected_sparse_topk_fraction",
    "stale_rank_score_mean",
    "stale_rank_score_max",
    "selected_stale_tx_fraction_mean",
    "selected_stale_tx_fraction_max",
    "pairwise_distance_mean",
    "pairwise_distance_min",
    "history_winner_in_set",
    "set_size_fraction",
)

__all__ = [
    "LEARNED_SET_SHORTLIST_FEATURE_NAMES",
    "LEARNED_SHORTLIST_FEATURE_NAMES",
    "choose_learned_set_shortlist_feedback_decision",
    "choose_learned_sparse_shortlist_feedback_decision",
    "learned_set_pairwise_distance_features",
    "learned_set_shortlist_feature_matrix",
    "learned_set_shortlist_feature_vector",
    "learned_set_shortlist_variants",
    "learned_shortlist_context",
    "learned_shortlist_feature_matrix",
    "learned_shortlist_feature_vector",
    "load_learned_set_shortlist_model",
    "load_learned_shortlist_model",
    "normalized_codebook_distance",
    "score_learned_shortlist_candidates",
]

def load_learned_shortlist_model(path):
    """读取learned、候选短名单、模型输入数据，并转换成脚本内部统一使用的行、字典或数组结构。"""
    data = np.load(path, allow_pickle=False)
    model = {
        "weights": np.asarray(data["weights"], dtype=float),
        "bias": float(np.asarray(data["bias"], dtype=float)),
        "feature_mean": np.asarray(data["feature_mean"], dtype=float),
        "feature_scale": np.asarray(data["feature_scale"], dtype=float),
    }
    if "feature_names" in data:
        model["feature_names"] = tuple(str(name) for name in data["feature_names"])
    else:
        model["feature_names"] = LEARNED_SHORTLIST_FEATURE_NAMES
    if tuple(model["feature_names"]) != LEARNED_SHORTLIST_FEATURE_NAMES:
        raise ValueError(
            "Learned shortlist model feature names do not match this evaluator. "
            f"Expected {LEARNED_SHORTLIST_FEATURE_NAMES}, got {model['feature_names']}"
        )
    expected_dim = len(LEARNED_SHORTLIST_FEATURE_NAMES)
    for key in ("weights", "feature_mean", "feature_scale"):
        if model[key].shape != (expected_dim,):
            raise ValueError(f"Learned shortlist model field {key} must have shape {(expected_dim,)}")
    return model


def load_learned_set_shortlist_model(path):
    """读取learned、set、候选短名单、模型输入数据，并转换成脚本内部统一使用的行、字典或数组结构。"""
    data = np.load(path, allow_pickle=False)
    model = {
        "weights": np.asarray(data["weights"], dtype=float),
        "bias": float(np.asarray(data["bias"], dtype=float)),
        "feature_mean": np.asarray(data["feature_mean"], dtype=float),
        "feature_scale": np.asarray(data["feature_scale"], dtype=float),
    }
    if "feature_names" in data:
        model["feature_names"] = tuple(str(name) for name in data["feature_names"])
    else:
        model["feature_names"] = LEARNED_SET_SHORTLIST_FEATURE_NAMES
    if tuple(model["feature_names"]) != LEARNED_SET_SHORTLIST_FEATURE_NAMES:
        raise ValueError(
            "Learned set shortlist model feature names do not match this evaluator. "
            f"Expected {LEARNED_SET_SHORTLIST_FEATURE_NAMES}, got {model['feature_names']}"
        )
    expected_dim = len(LEARNED_SET_SHORTLIST_FEATURE_NAMES)
    for key in ("weights", "feature_mean", "feature_scale"):
        if model[key].shape != (expected_dim,):
            raise ValueError(f"Learned set shortlist model field {key} must have shape {(expected_dim,)}")
    return model


def learned_shortlist_context(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    episode_seed,
    base_multiplier=2.0,
    topk_fraction=0.75,
    confirmed_history=None,
    channel_rho=1.0,
    csi_delay_slots=0,
):
    """处理learned、候选短名单、context相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    budget = min(int(budget), args.num_codebook_states)
    base_seed_budget = min(
        args.num_codebook_states,
        max(budget, int(np.ceil(float(base_multiplier) * budget))),
    )
    base_seed_indices = limited.grid_indices(args.num_codebook_states, base_seed_budget, offset=slot_idx)
    base_rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_ROTATING_GRID,
        base_seed_budget,
        salt=163 + slot_idx,
    )
    base_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=base_seed_indices,
        error_std=decision_error_std,
        rng=base_rng,
    )
    ranked_candidates = sorted(base_candidates, key=limited.candidate_key, reverse=True)
    ranked_indices = [int(candidate["irs_index"]) for candidate in ranked_candidates]
    topk_budget = min(budget, max(1, int(np.ceil(float(topk_fraction) * budget))))
    stale_margin = sparse_topk_count_margin(base_candidates, topk_budget, args.num_nodes)
    history_best_index, history_stability = adaptive_sparse_history_signal(
        confirmed_history or [],
        getattr(args, "adaptive_sparse_history_window", 3),
    )
    rotating_indices = limited.grid_indices(args.num_codebook_states, budget, offset=slot_idx)
    return {
        "budget": int(budget),
        "base_seed_indices": [int(index) for index in base_seed_indices],
        "base_candidates": base_candidates,
        "candidate_by_index": {int(candidate["irs_index"]): candidate for candidate in base_candidates},
        "ranked_candidates": ranked_candidates,
        "ranked_indices": ranked_indices,
        "topk_budget": int(topk_budget),
        "stale_margin": float(stale_margin),
        "history_best_index": history_best_index,
        "history_stability": float(history_stability),
        "rotating_indices": [int(index) for index in rotating_indices],
        "channel_rho": float(channel_rho),
        "csi_delay_slots": int(csi_delay_slots),
    }


def normalized_codebook_distance(index, other_indices, num_codebook_states):
    """处理normalized、码本、distance相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if not other_indices:
        return 1.0
    max_distance = max(float(num_codebook_states) / 2.0, 1.0)
    distance = min(
        circular_codebook_distance(index, other, num_codebook_states)
        for other in other_indices
    )
    return float(distance) / max_distance


def learned_shortlist_feature_vector(args, candidate_index, slot_idx, context):
    """处理learned、候选短名单、特征、vector相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    num_codebook_states = int(args.num_codebook_states)
    index = int(candidate_index) % num_codebook_states
    phase = 2.0 * np.pi * float(index) / max(float(num_codebook_states), 1.0)
    ranked_candidates = context["ranked_candidates"]
    ranked_indices = context["ranked_indices"]
    top_candidate = ranked_candidates[0] if ranked_candidates else None
    kth_pos = min(max(int(context["topk_budget"]), 1), max(len(ranked_candidates), 1)) - 1
    kth_candidate = ranked_candidates[kth_pos] if ranked_candidates else None
    top_tx = float(top_candidate["tx_this_slot"]) if top_candidate is not None else 0.0
    kth_tx = float(kth_candidate["tx_this_slot"]) if kth_candidate is not None else 0.0
    top_power = float(top_candidate["power_avg"]) if top_candidate is not None else 0.0
    top_gain = float(top_candidate["mean_gain_remaining"]) if top_candidate is not None else 0.0
    rho_delay = float(context["channel_rho"]) ** max(int(context["csi_delay_slots"]), 0)
    history_best = context["history_best_index"]
    history_indices = [] if history_best is None else [int(history_best)]
    top1 = ranked_indices[:1]
    top2 = ranked_indices[:2]
    top3 = ranked_indices[:3]
    features = np.asarray(
        [
            np.sin(phase),
            np.cos(phase),
            float(slot_idx) / max(float(args.num_slots - 1), 1.0),
            float(args.num_slots - int(slot_idx)) / max(float(args.num_slots), 1.0),
            rho_delay,
            1.0 - rho_delay,
            top_tx / max(float(args.num_nodes), 1.0),
            kth_tx / max(float(args.num_nodes), 1.0),
            float(context["stale_margin"]),
            top_power,
            top_gain,
            normalized_codebook_distance(index, top1, num_codebook_states),
            normalized_codebook_distance(index, top2, num_codebook_states),
            normalized_codebook_distance(index, top3, num_codebook_states),
            normalized_codebook_distance(index, context["base_seed_indices"], num_codebook_states),
            normalized_codebook_distance(index, context["rotating_indices"], num_codebook_states),
            normalized_codebook_distance(index, history_indices, num_codebook_states),
            1.0 if history_best is not None and index == int(history_best) else 0.0,
            float(context["history_stability"]),
            1.0 if index in set(context["rotating_indices"]) else 0.0,
        ],
        dtype=float,
    )
    return features


def learned_shortlist_feature_matrix(args, candidate_indices, slot_idx, context):
    """处理learned、候选短名单、特征、matrix相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if not candidate_indices:
        return np.zeros((0, len(LEARNED_SHORTLIST_FEATURE_NAMES)), dtype=float)
    return np.vstack(
        [learned_shortlist_feature_vector(args, index, slot_idx, context) for index in candidate_indices]
    )


def score_learned_shortlist_candidates(model, features):
    """对learned、候选短名单、候选集合进行打分或排序，为候选选择、诊断归因或学习标签提供比较依据。"""
    if features.size == 0:
        return np.zeros((0,), dtype=float)
    scale = np.maximum(np.asarray(model["feature_scale"], dtype=float), 1e-12)
    normalized = (features - np.asarray(model["feature_mean"], dtype=float)) / scale
    return normalized @ np.asarray(model["weights"], dtype=float) + float(model["bias"])


def learned_set_shortlist_variants(args, context, max_extra_count):
    """处理learned、set、候选短名单、variants相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    budget = min(int(context["budget"]), int(args.num_codebook_states))
    max_extra_count = max(0, min(int(max_extra_count), max(budget - 1, 0)))
    ranked_indices = list(context["ranked_indices"])
    rotating_indices = list(context["rotating_indices"])
    seed_seen = set(int(index) for index in context["base_seed_indices"])
    learned_pool = [index for index in range(args.num_codebook_states) if index not in seed_seen]
    variants = []
    seen_sets = set()

    for extra_count in range(max_extra_count + 1):
        if extra_count == 0:
            extra_combos = [()]
        elif len(learned_pool) >= extra_count:
            extra_combos = itertools.combinations(learned_pool, extra_count)
        else:
            extra_combos = [tuple(learned_pool)]

        for extra_combo in extra_combos:
            extra_indices = [int(index) for index in extra_combo]
            base_head_count = max(
                1,
                min(int(context["topk_budget"]), budget - len(extra_indices)),
            )
            selected_indices = ordered_unique_prefix(
                (
                    ranked_indices[:base_head_count]
                    + extra_indices
                    + rotating_indices
                    + ranked_indices[base_head_count:]
                ),
                budget,
                args.num_codebook_states,
            )
            key = tuple(selected_indices)
            if key in seen_sets:
                continue
            seen_sets.add(key)
            variants.append(
                {
                    "selected_indices": selected_indices,
                    "extra_indices": [index for index in selected_indices if index in set(extra_indices)],
                    "requested_extra_count": int(extra_count),
                }
            )
    return variants


def learned_set_pairwise_distance_features(selected_indices, num_codebook_states):
    """处理learned、set、pairwise、distance、特征相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    selected = [int(index) for index in selected_indices]
    if len(selected) < 2:
        return 1.0, 1.0
    max_distance = max(float(num_codebook_states) / 2.0, 1.0)
    distances = [
        float(circular_codebook_distance(left, right, num_codebook_states)) / max_distance
        for left, right in itertools.combinations(selected, 2)
    ]
    return float(np.mean(distances)), float(np.min(distances))


def learned_set_shortlist_feature_vector(args, variant, slot_idx, context):
    """处理learned、set、候选短名单、特征、vector相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    selected_indices = [
        int(np.clip(index, 0, args.num_codebook_states - 1))
        for index in variant["selected_indices"]
    ]
    if not selected_indices:
        return np.zeros((len(LEARNED_SET_SHORTLIST_FEATURE_NAMES),), dtype=float)

    candidate_features = learned_shortlist_feature_matrix(args, selected_indices, slot_idx, context)
    feature_mean = np.mean(candidate_features, axis=0)
    feature_max = np.max(candidate_features, axis=0)

    budget = max(float(context["budget"]), 1.0)
    seed_set = set(int(index) for index in context["base_seed_indices"])
    rotating_set = set(int(index) for index in context["rotating_indices"])
    sparse_topk_set = set(int(index) for index in context["ranked_indices"][: int(context["topk_budget"])])
    extra_set = set(int(index) for index in variant.get("extra_indices", []))
    ranked_indices = list(context["ranked_indices"])
    rank_by_index = {int(index): pos for pos, index in enumerate(ranked_indices)}
    rank_denominator = max(float(len(ranked_indices) - 1), 1.0)
    candidate_by_index = context["candidate_by_index"]

    stale_rank_scores = [
        1.0 - float(rank_by_index[index]) / rank_denominator
        if index in rank_by_index
        else 0.0
        for index in selected_indices
    ]
    stale_tx_fractions = [
        float(candidate_by_index[index]["tx_this_slot"]) / max(float(args.num_nodes), 1.0)
        if index in candidate_by_index
        else 0.0
        for index in selected_indices
    ]
    pairwise_mean, pairwise_min = learned_set_pairwise_distance_features(
        selected_indices,
        args.num_codebook_states,
    )
    history_best = context["history_best_index"]
    history_winner_in_set = (
        1.0
        if history_best is not None and int(history_best) in set(selected_indices)
        else 0.0
    )

    set_features = np.asarray(
        [
            len([index for index in selected_indices if index in extra_set]) / budget,
            len([index for index in selected_indices if index in seed_set]) / budget,
            len([index for index in selected_indices if index in rotating_set]) / budget,
            len([index for index in selected_indices if index in sparse_topk_set]) / budget,
            float(np.mean(stale_rank_scores)),
            float(np.max(stale_rank_scores)),
            float(np.mean(stale_tx_fractions)),
            float(np.max(stale_tx_fractions)),
            pairwise_mean,
            pairwise_min,
            history_winner_in_set,
            len(selected_indices) / max(float(args.num_codebook_states), 1.0),
        ],
        dtype=float,
    )
    return np.concatenate([feature_mean, feature_max, set_features]).astype(float)


def learned_set_shortlist_feature_matrix(args, variants, slot_idx, context):
    """处理learned、set、候选短名单、特征、matrix相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if not variants:
        return np.zeros((0, len(LEARNED_SET_SHORTLIST_FEATURE_NAMES)), dtype=float)
    return np.vstack(
        [learned_set_shortlist_feature_vector(args, variant, slot_idx, context) for variant in variants]
    )


def choose_learned_sparse_shortlist_feedback_decision(
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
    topk_fraction=None,
    extra_count=None,
    model=None,
):
    """按照learned、稀疏、候选短名单、聚合反馈、决策规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if base_multiplier is None:
        base_multiplier = getattr(args, "adaptive_sparse_base_multiplier", 2.0)
    if topk_fraction is None:
        topk_fraction = getattr(args, "sparse_topk_fraction", 0.75)
    if extra_count is None:
        extra_count = getattr(args, "learned_shortlist_extra_count", 1)
    if model is None:
        model = getattr(args, "learned_shortlist_model_data", None)
    if model is None:
        raise ValueError("learned shortlist feedback requires a loaded model")

    extra_count = max(0, min(int(extra_count), max(budget - 1, 0)))
    context = learned_shortlist_context(
        env,
        args,
        budget,
        slot_idx,
        decision_error_std,
        episode_seed,
        base_multiplier=base_multiplier,
        topk_fraction=topk_fraction,
        confirmed_history=confirmed_history,
        channel_rho=channel_rho,
        csi_delay_slots=csi_delay_slots,
    )
    candidate_by_index = dict(context["candidate_by_index"])
    base_seen = set(candidate_by_index)
    learned_pool = [index for index in range(args.num_codebook_states) if index not in base_seen]
    learned_extra_indices = []
    if extra_count > 0 and learned_pool:
        features = learned_shortlist_feature_matrix(args, learned_pool, slot_idx, context)
        scores = score_learned_shortlist_candidates(model, features)
        ranked_extra = sorted(
            zip(learned_pool, scores),
            key=lambda item: (float(item[1]), -int(item[0])),
            reverse=True,
        )
        learned_extra_indices = [int(index) for index, _score in ranked_extra[:extra_count]]

    if learned_extra_indices:
        extra_rng = limited.stable_rng(
            episode_seed,
            decision_error_std,
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            len(learned_extra_indices),
            salt=173 + slot_idx,
        )
        extra_candidates = limited.estimated_preview_candidates(
            env,
            args,
            indices=learned_extra_indices,
            error_std=decision_error_std,
            rng=extra_rng,
        )
        candidate_by_index.update({int(candidate["irs_index"]): candidate for candidate in extra_candidates})

    ranked_indices = list(context["ranked_indices"])
    base_head_count = max(1, min(int(context["topk_budget"]), budget - len(learned_extra_indices)))
    selected_indices = ordered_unique_prefix(
        (
            ranked_indices[:base_head_count]
            + learned_extra_indices
            + context["rotating_indices"]
            + ranked_indices[base_head_count:]
        ),
        budget,
        args.num_codebook_states,
    )

    selected_extra_count = len([index for index in selected_indices if int(index) in set(learned_extra_indices)])
    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=167,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["sparse_seed_multiplier"] = float(base_multiplier)
    decision["sparse_topk_fraction"] = float(topk_fraction)
    decision["sparse_seed_count"] = int(len(context["base_seed_indices"]))
    decision["sparse_topk_count"] = int(context["topk_budget"])
    decision["sparse_extra_preview_count"] = int(len(learned_extra_indices))
    decision["learned_shortlist_extra_count"] = int(extra_count)
    decision["learned_shortlist_selected_extra_preview_count"] = float(selected_extra_count)
    decision["adaptive_sparse_margin"] = float(context["stale_margin"])
    decision["adaptive_sparse_history_stability"] = float(context["history_stability"])
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    preview_calls = len(context["base_seed_indices"]) + len(learned_extra_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)


def choose_learned_set_shortlist_feedback_decision(
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
    topk_fraction=None,
    extra_count=None,
    model=None,
):
    """按照learned、set、候选短名单、聚合反馈、决策规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    budget = min(int(budget), args.num_codebook_states)
    if budget <= 0:
        return choose_no_irs_fallback(env, args, slot_idx, decision_error_std, episode_seed)

    if base_multiplier is None:
        base_multiplier = getattr(args, "adaptive_sparse_base_multiplier", 2.0)
    if topk_fraction is None:
        topk_fraction = getattr(args, "sparse_topk_fraction", 0.75)
    if extra_count is None:
        extra_count = getattr(args, "learned_set_extra_count", 1)
    if model is None:
        model = getattr(args, "learned_set_shortlist_model_data", None)
    if model is None:
        raise ValueError("learned set shortlist feedback requires a loaded model")

    extra_count = max(0, min(int(extra_count), max(budget - 1, 0)))
    context = learned_shortlist_context(
        env,
        args,
        budget,
        slot_idx,
        decision_error_std,
        episode_seed,
        base_multiplier=base_multiplier,
        topk_fraction=topk_fraction,
        confirmed_history=confirmed_history,
        channel_rho=channel_rho,
        csi_delay_slots=csi_delay_slots,
    )
    variants = learned_set_shortlist_variants(args, context, extra_count)
    if not variants:
        variants = [
            {
                "selected_indices": limited.grid_indices(args.num_codebook_states, budget, offset=slot_idx),
                "extra_indices": [],
                "requested_extra_count": 0,
            }
        ]
    features = learned_set_shortlist_feature_matrix(args, variants, slot_idx, context)
    scores = score_learned_shortlist_candidates(model, features)
    best_idx = int(np.argmax(scores)) if len(scores) else 0
    best_variant = variants[best_idx]
    selected_indices = [int(index) for index in best_variant["selected_indices"]]

    candidate_by_index = dict(context["candidate_by_index"])
    extra_indices = [index for index in selected_indices if int(index) not in candidate_by_index]
    if extra_indices:
        extra_rng = limited.stable_rng(
            episode_seed,
            decision_error_std,
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            len(extra_indices),
            salt=181 + slot_idx,
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
        feedback_salt=181,
    )

    decision = candidate_by_index[int(confirmed_index)]
    decision["sparse_seed_multiplier"] = float(base_multiplier)
    decision["sparse_topk_fraction"] = float(topk_fraction)
    decision["sparse_seed_count"] = int(len(context["base_seed_indices"]))
    decision["sparse_topk_count"] = int(context["topk_budget"])
    decision["sparse_extra_preview_count"] = int(len(extra_indices))
    decision["learned_shortlist_extra_count"] = int(extra_count)
    decision["learned_shortlist_selected_extra_preview_count"] = float(len(extra_indices))
    decision["learned_set_shortlist_variant_count"] = int(len(variants))
    decision["adaptive_sparse_margin"] = float(context["stale_margin"])
    decision["adaptive_sparse_history_stability"] = float(context["history_stability"])
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    preview_calls = len(context["base_seed_indices"]) + len(extra_indices) + len(selected_indices)
    return decision, preview_calls, len(selected_indices)
