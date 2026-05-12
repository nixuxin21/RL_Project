"""
Lightweight correctness checks for the default MS-AirComp scenario.

These checks intentionally avoid pytest so they can run with:

    ./.venv/bin/python tests/smoke_checks.py
"""

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_policy_comparison import (
    codebook_index_to_action,
    feature_argmax_power_tie_index,
    make_episode_seeds,
)
from evaluate_partial_probing_sweep import (
    POLICY_FIXED_GRID,
    POLICY_HYBRID,
    POLICY_LOCAL,
    POLICY_RANDOM,
    POLICY_ROTATING_GRID,
    grid_indices,
    local_indices,
    select_probe_indices,
    stable_probe_rng,
)
from evaluate_channel_estimation_error_sweep import (
    estimated_preview_candidates,
    make_error_rng,
)
from evaluate_limited_csi_ms_aircomp import (
    adaptive_risk_weight,
    estimated_preview_candidates as limited_estimated_preview_candidates,
    execute_limited_csi_slot,
    risk_aware_candidate,
    true_preview_candidates,
)
from evaluate_bandit_feedback_ms_aircomp import (
    POLICY_RANDOM_PROBE as POLICY_BANDIT_RANDOM_PROBE,
    POLICY_ROTATING_GRID as POLICY_BANDIT_ROTATING_GRID,
    POLICY_THOMPSON_PROBE,
    POLICY_UCB_PROBE,
    initialize_feedback_state,
    observe_probe_feedback,
    select_feedback_probe_indices,
)
from test_env import MSAirCompEnv


def make_args():
    return Namespace(
        num_nodes=50,
        num_slots=10,
        num_irs_elements=64,
        num_codebook_states=16,
        g_th=0.001,
        alpha_th=0.05,
        episodes=8,
        seed=2026,
        feedback_power_weight=0.05,
        power_feedback_noise_std=0.0,
        bandit_lr=0.6,
        bandit_prior=0.0,
        ucb_coeff=0.25,
        thompson_std=0.2,
    )


def make_action(args, irs_index):
    return np.array(
        [
            -1.0,
            -1.0,
            codebook_index_to_action(irs_index, args.num_codebook_states),
        ],
        dtype=np.float32,
    )


def greedy_candidate(env, args):
    candidates = [
        env.preview_codebook_index(index, args.g_th, args.alpha_th)
        for index in range(args.num_codebook_states)
    ]

    def candidate_key(candidate):
        tx_count = int(candidate["tx_this_slot"])
        power_avg = float(candidate["power_avg"])
        mean_gain = float(candidate["mean_gain_remaining"])
        power_tiebreak = -power_avg if tx_count > 0 else 0.0
        return tx_count, power_tiebreak, mean_gain

    return max(candidates, key=candidate_key)


def assert_preview_has_no_side_effects(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    slot_before = env.current_slot
    flags_before = env.transmitted_flags.copy()

    first = env.preview_codebook_index(3, args.g_th, args.alpha_th)
    second = env.preview_codebook_index(3, args.g_th, args.alpha_th)

    assert env.current_slot == slot_before
    assert np.array_equal(env.transmitted_flags, flags_before)
    assert first["tx_this_slot"] == second["tx_this_slot"]
    assert np.isclose(first["power_avg"], second["power_avg"])


def assert_codebook_features_match_preview(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        include_codebook_features=True,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
    )
    obs, _info = env.reset(seed=args.seed)
    features = obs[7 : 7 + args.num_codebook_states]
    expected = np.array(
        [
            env.preview_codebook_index(index, args.g_th, args.alpha_th)["tx_this_slot"] / args.num_nodes
            for index in range(args.num_codebook_states)
        ],
        dtype=np.float32,
    )
    assert np.allclose(features, expected)


def assert_noisy_codebook_features_are_seeded_and_clipped(args):
    exact_env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        include_codebook_features=True,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
    )
    noisy_env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        include_codebook_features=True,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
        codebook_feature_noise_std=0.25,
    )
    noisy_env_again = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        include_codebook_features=True,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
        codebook_feature_noise_std=0.25,
    )

    exact_obs, _info = exact_env.reset(seed=args.seed)
    noisy_obs, _info = noisy_env.reset(seed=args.seed)
    noisy_obs_again, _info = noisy_env_again.reset(seed=args.seed)

    exact_features = exact_obs[7 : 7 + args.num_codebook_states]
    noisy_features = noisy_obs[7 : 7 + args.num_codebook_states]
    noisy_features_again = noisy_obs_again[7 : 7 + args.num_codebook_states]

    assert np.all(noisy_features >= 0.0)
    assert np.all(noisy_features <= 1.0)
    assert np.allclose(noisy_features, noisy_features_again)
    assert not np.allclose(noisy_features, exact_features)


def assert_negative_feature_noise_is_rejected(args):
    try:
        MSAirCompEnv(
            num_nodes=args.num_nodes,
            num_slots=args.num_slots,
            num_irs_elements=args.num_irs_elements,
            num_codebook_states=args.num_codebook_states,
            include_codebook_features=True,
            codebook_feature_noise_std=-0.1,
        )
    except ValueError:
        return
    raise AssertionError("negative codebook feature noise std should raise ValueError")


def assert_power_tie_matches_greedy(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        include_codebook_features=True,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
    )

    for episode_seed in make_episode_seeds(args, args.seed):
        obs, _info = env.reset(seed=episode_seed)
        for _slot in range(args.num_slots):
            power_tie_index, _preview_calls, _tie_count = feature_argmax_power_tie_index(env, args, obs)
            power_tie = env.preview_codebook_index(power_tie_index, args.g_th, args.alpha_th)
            greedy = greedy_candidate(env, args)

            assert power_tie["tx_this_slot"] == greedy["tx_this_slot"]
            assert np.isclose(power_tie["power_avg"], greedy["power_avg"])

            obs, _reward, terminated, truncated, _info = env.step(make_action(args, power_tie_index))
            if terminated or truncated:
                break


def assert_episode_seed_generation_is_stable(args):
    first = make_episode_seeds(args, args.seed)
    second = make_episode_seeds(args, args.seed)
    shifted = make_episode_seeds(args, args.seed + 1)
    assert first == second
    assert first != shifted
    assert len(first) == args.episodes


def assert_partial_probe_selectors_respect_budget(args):
    policies = [
        POLICY_RANDOM,
        POLICY_FIXED_GRID,
        POLICY_ROTATING_GRID,
        POLICY_LOCAL,
        POLICY_HYBRID,
    ]
    for budget in (1, 2, 4, 8, args.num_codebook_states):
        for policy_name in policies:
            rng = stable_probe_rng(args.seed, policy_name, budget)
            indices = select_probe_indices(
                policy_name,
                args,
                budget,
                slot_idx=3,
                state={"previous_index": 7},
                rng=rng,
            )
            assert len(indices) == budget
            assert len(set(indices)) == budget
            assert all(0 <= index < args.num_codebook_states for index in indices)

    assert grid_indices(args.num_codebook_states, args.num_codebook_states) == list(
        range(args.num_codebook_states)
    )
    assert local_indices(7, 1, args.num_codebook_states) == [7]
    assert local_indices(7, 3, args.num_codebook_states) == [7, 6, 8]


def assert_estimated_preview_zero_error_matches_exact(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    indices = [0, 3, 7, 15]
    estimated = estimated_preview_candidates(
        env,
        args,
        indices,
        error_std=0.0,
        rng=make_error_rng(args.seed, 0.0),
    )
    exact = [
        env.preview_codebook_index(index, args.g_th, args.alpha_th)
        for index in indices
    ]

    for estimated_candidate, exact_candidate in zip(estimated, exact):
        assert estimated_candidate["irs_index"] == exact_candidate["irs_index"]
        assert estimated_candidate["tx_this_slot"] == exact_candidate["tx_this_slot"]
        assert np.isclose(estimated_candidate["power_avg"], exact_candidate["power_avg"])
        assert np.isclose(
            estimated_candidate["mean_gain_remaining"],
            exact_candidate["mean_gain_remaining"],
        )


def assert_limited_csi_zero_error_matches_true(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    indices = [0, 4, 7, 15]
    exact = true_preview_candidates(env, args, indices)
    estimated = limited_estimated_preview_candidates(
        env,
        args,
        indices=indices,
        error_std=0.0,
        rng=np.random.default_rng(args.seed),
    )

    for exact_candidate, estimated_candidate in zip(exact, estimated):
        assert exact_candidate["irs_index"] == estimated_candidate["irs_index"]
        assert exact_candidate["tx_this_slot"] == estimated_candidate["tx_this_slot"]
        assert np.array_equal(exact_candidate["valid_mask"], estimated_candidate["valid_mask"])
        assert np.allclose(exact_candidate["required_power"], estimated_candidate["required_power"])


def assert_limited_csi_execution_only_counts_invited_nodes(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    true_candidate = max(
        true_preview_candidates(env, args, range(args.num_codebook_states)),
        key=lambda candidate: candidate["tx_this_slot"],
    )
    assert true_candidate["tx_this_slot"] > 0

    decision_candidate = dict(true_candidate)
    decision_candidate["valid_mask"] = np.zeros(args.num_nodes, dtype=bool)
    info, done = execute_limited_csi_slot(env, args, decision_candidate, true_candidate)

    assert not done
    assert info["tx_this_slot"] == 0
    assert info["scheduled_this_slot"] == 0
    assert info["missed_opportunity_this_slot"] == true_candidate["tx_this_slot"]
    assert int(np.sum(env.transmitted_flags)) == 0
    assert env.current_slot == 1


def assert_risk_aware_candidate_filters_low_reliability(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    base_candidate = max(
        true_preview_candidates(env, args, range(args.num_codebook_states)),
        key=lambda candidate: candidate["tx_this_slot"],
    )
    assert base_candidate["tx_this_slot"] > 0

    candidate = dict(base_candidate)
    reliability = np.zeros(args.num_nodes, dtype=float)
    valid_indices = np.flatnonzero(base_candidate["valid_mask"])
    reliability[valid_indices] = np.linspace(0.45, 0.95, num=len(valid_indices))
    candidate["success_reliability"] = reliability

    adjusted = risk_aware_candidate(
        candidate,
        args,
        slot_idx=0,
        risk_weight=0.5,
        risk_power_weight=0.1,
        risk_invite_threshold=0.75,
    )

    assert adjusted["tx_this_slot"] < base_candidate["tx_this_slot"]
    assert np.all(adjusted["valid_mask"] <= base_candidate["valid_mask"])
    assert np.all(candidate["success_reliability"][adjusted["valid_mask"]] >= 0.75)


def assert_adaptive_risk_weight_tracks_uncertainty_and_deadline(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)

    low_error = adaptive_risk_weight(env, args, error_std=0.05, slot_idx=0, base_weight=0.5)
    high_error = adaptive_risk_weight(env, args, error_std=0.3, slot_idx=0, base_weight=0.5)
    late_slot = adaptive_risk_weight(env, args, error_std=0.3, slot_idx=args.num_slots - 1, base_weight=0.5)

    assert high_error > low_error
    assert late_slot < high_error

    env.transmitted_flags[:] = False
    behind = adaptive_risk_weight(env, args, error_std=0.3, slot_idx=args.num_slots - 2, base_weight=0.5)
    env.transmitted_flags[: args.num_nodes - 2] = True
    ahead = adaptive_risk_weight(env, args, error_std=0.3, slot_idx=args.num_slots - 2, base_weight=0.5)

    assert behind < ahead


def assert_bandit_feedback_observes_only_aggregate_metrics(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    candidate = env.preview_codebook_index(3, args.g_th, args.alpha_th)

    exact_feedback = observe_probe_feedback(
        candidate,
        args,
        feedback_noise_std=0.0,
        rng=np.random.default_rng(args.seed),
    )
    expected_fraction = candidate["tx_this_slot"] / args.num_nodes
    assert np.isclose(exact_feedback["observed_tx_fraction"], expected_fraction)
    assert "valid_mask" not in exact_feedback
    assert "required_power" not in exact_feedback

    noisy_first = observe_probe_feedback(
        candidate,
        args,
        feedback_noise_std=0.2,
        rng=np.random.default_rng(args.seed),
    )
    noisy_second = observe_probe_feedback(
        candidate,
        args,
        feedback_noise_std=0.2,
        rng=np.random.default_rng(args.seed),
    )
    assert noisy_first == noisy_second
    assert 0.0 <= noisy_first["observed_tx_fraction"] <= 1.0


def assert_bandit_probe_selectors_respect_budget(args):
    state = initialize_feedback_state(args)
    state["counts"][:4] = np.array([1.0, 2.0, 3.0, 4.0])
    state["means"][:4] = np.array([0.1, 0.2, 0.3, 0.4])
    policies = [
        POLICY_BANDIT_RANDOM_PROBE,
        POLICY_BANDIT_ROTATING_GRID,
        POLICY_UCB_PROBE,
        POLICY_THOMPSON_PROBE,
    ]
    for budget in (1, 2, 4, 8, args.num_codebook_states):
        for policy_name in policies:
            indices = select_feedback_probe_indices(
                policy_name,
                args,
                budget,
                slot_idx=2,
                state=state,
                rng=np.random.default_rng(args.seed),
            )
            assert len(indices) == budget
            assert len(set(indices)) == budget
            assert all(0 <= index < args.num_codebook_states for index in indices)


def main():
    args = make_args()
    assert_preview_has_no_side_effects(args)
    assert_codebook_features_match_preview(args)
    assert_noisy_codebook_features_are_seeded_and_clipped(args)
    assert_negative_feature_noise_is_rejected(args)
    assert_power_tie_matches_greedy(args)
    assert_episode_seed_generation_is_stable(args)
    assert_partial_probe_selectors_respect_budget(args)
    assert_estimated_preview_zero_error_matches_exact(args)
    assert_limited_csi_zero_error_matches_true(args)
    assert_limited_csi_execution_only_counts_invited_nodes(args)
    assert_risk_aware_candidate_filters_low_reliability(args)
    assert_adaptive_risk_weight_tracks_uncertainty_and_deadline(args)
    assert_bandit_feedback_observes_only_aggregate_metrics(args)
    assert_bandit_probe_selectors_respect_budget(args)
    print("smoke checks passed")


if __name__ == "__main__":
    main()
