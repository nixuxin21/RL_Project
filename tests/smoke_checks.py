"""
Lightweight correctness checks for the default MS-AirComp scenario.

These checks intentionally avoid pytest so they can run with:

    ./.venv/bin/python tests/smoke_checks.py
"""

from argparse import Namespace
from pathlib import Path
import subprocess
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
    summarize_results as summarize_limited_results,
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
from ms_aircomp.adaptive_sparse_policies import (
    choose_adaptive_sparse_topk_feedback_decision,
    choose_adaptive_sparse_topk_v2_feedback_decision,
    choose_adaptive_sparse_topk_v3_feedback_decision,
)
from ms_aircomp.channel_models import (
    build_temporal_channel_trace,
    delayed_channel_state,
)
from ms_aircomp.execution_policies import (
    choose_active_diverse_feedback_decision,
    choose_neighbor_coverage_sparse_topk_feedback_decision,
    choose_sparse_topk_feedback_decision,
)
from ms_aircomp.execution_candidates import execution_candidates
from ms_aircomp.execution_result_summary import result_metadata
import ms_aircomp.execution_policy_registry as execution_policy_registry
from ms_aircomp.invitation_mask_correction import (
    MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN,
    MASK_CORRECTION_MODE_PRUNE_ONLY,
    apply_invitation_mask_correction,
    corrected_target_count,
)
from ms_aircomp.learned_shortlist import (
    LEARNED_SET_SHORTLIST_FEATURE_NAMES,
    LEARNED_SHORTLIST_FEATURE_NAMES,
    choose_learned_set_shortlist_feedback_decision,
    choose_learned_sparse_shortlist_feedback_decision,
)
from ms_aircomp.probe_sets import fill_diverse_codebook_indices
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
        confirmation_feedback_noise_std=0.0,
        confirmation_feedback_power_weight=0.05,
        sparse_topk_seed_multiplier=2.0,
        sparse_topk_fraction=0.75,
        adaptive_sparse_base_multiplier=2.0,
        adaptive_sparse_expanded_multiplier=3.0,
        adaptive_sparse_margin_threshold=0.05,
        adaptive_sparse_v2_preview_cost=0.002,
        adaptive_sparse_v2_uncertainty_weight=0.02,
        adaptive_sparse_v2_urgency_weight=0.5,
        adaptive_sparse_v2_history_weight=0.02,
        adaptive_sparse_history_window=3,
        adaptive_sparse_history_prior_threshold=0.67,
        adaptive_sparse_v3_neighbor_radius=1,
        adaptive_sparse_v3_neighbor_count=2,
        adaptive_sparse_v3_history_count=1,
        learned_shortlist_extra_count=1,
        learned_set_extra_count=1,
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


def assert_invalid_environment_dimensions_are_rejected():
    invalid_configs = [
        {"num_nodes": 0},
        {"num_slots": 0},
        {"num_irs_elements": 0},
        {"num_codebook_states": 1},
    ]
    for config in invalid_configs:
        try:
            MSAirCompEnv(**config)
        except ValueError:
            continue
        raise AssertionError(f"invalid environment config should raise ValueError: {config}")


def assert_environment_step_clips_out_of_bounds_actions(args):
    env_clipped = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env_reference = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env_clipped.reset(seed=args.seed)
    env_reference.reset(seed=args.seed)

    clipped_obs, clipped_reward, clipped_done, clipped_truncated, clipped_info = env_clipped.step(
        np.array([2.0, -3.0, 5.0], dtype=np.float32)
    )
    reference_obs, reference_reward, reference_done, reference_truncated, reference_info = env_reference.step(
        np.array([1.0, -1.0, 1.0], dtype=np.float32)
    )

    assert clipped_info["action_clipped"] is True
    assert reference_info["action_clipped"] is False
    assert clipped_info["irs_index"] == args.num_codebook_states - 1
    assert clipped_info["tx_this_slot"] == reference_info["tx_this_slot"]
    assert clipped_info["total_tx"] == reference_info["total_tx"]
    assert np.isclose(clipped_info["power_avg"], reference_info["power_avg"])
    assert np.isclose(clipped_reward, reference_reward)
    assert clipped_done == reference_done
    assert clipped_truncated == reference_truncated
    assert np.allclose(clipped_obs, reference_obs)

    invalid_env = MSAirCompEnv()
    invalid_env.reset(seed=args.seed)
    try:
        invalid_env.step(np.array([0.0, 0.0], dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong action shape should raise ValueError")

    try:
        invalid_env.step(np.array([np.nan, 0.0, 0.0], dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite action should raise ValueError")


def _assert_cli_value_error(argv):
    result = subprocess.run(
        [sys.executable, *argv],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def assert_mainline_cli_validators_reject_invalid_args():
    _assert_cli_value_error(
        [
            "evaluate_execution_channel_mismatch.py",
            "--episodes",
            "1",
            "--num-seeds",
            "1",
            "--probe-budgets",
            "99",
            "--output-prefix",
            "/tmp/audit_invalid_execution_budget",
            "--no-plots",
        ]
    )
    _assert_cli_value_error(
        [
            "evaluate_execution_channel_mismatch.py",
            "--episodes",
            "1",
            "--num-seeds",
            "1",
            "--fixed-irs-index",
            "99",
            "--output-prefix",
            "/tmp/audit_invalid_execution_fixed",
            "--no-plots",
        ]
    )
    _assert_cli_value_error(
        [
            "evaluate_invitation_mask_correction.py",
            "--episodes",
            "1",
            "--num-seeds",
            "1",
            "--probe-budget",
            "99",
            "--output-prefix",
            "/tmp/audit_invalid_mask_budget",
        ]
    )
    _assert_cli_value_error(
        [
            "evaluate_invitation_mask_correction.py",
            "--episodes",
            "1",
            "--num-seeds",
            "1",
            "--channel-rho-values",
            "1.2",
            "--output-prefix",
            "/tmp/audit_invalid_mask_rho",
        ]
    )
    _assert_cli_value_error(
        [
            "evaluate_invitation_mask_correction.py",
            "--episodes",
            "1",
            "--num-seeds",
            "1",
            "--mask-correction-rerank-modes",
            "oracle",
            "--output-prefix",
            "/tmp/audit_invalid_mask_mode",
        ]
    )


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


def assert_temporal_trace_supports_prehistory(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    history_states, execution_states = build_temporal_channel_trace(
        env,
        args,
        args.seed,
        channel_rho=0.9,
        prehistory_slots=3,
    )

    assert len(history_states) == 3
    assert len(execution_states) == args.num_slots

    slot0_delay3 = delayed_channel_state(
        execution_states,
        slot_idx=0,
        csi_delay_slots=3,
        history_states=history_states,
    )
    slot2_delay3 = delayed_channel_state(
        execution_states,
        slot_idx=2,
        csi_delay_slots=3,
        history_states=history_states,
    )
    slot4_delay3 = delayed_channel_state(
        execution_states,
        slot_idx=4,
        csi_delay_slots=3,
        history_states=history_states,
    )

    assert np.allclose(slot0_delay3["h_d"], history_states[0]["h_d"])
    assert np.allclose(slot2_delay3["h_d"], history_states[-1]["h_d"])
    assert np.allclose(slot4_delay3["h_d"], execution_states[1]["h_d"])


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


def assert_limited_csi_summary_names_failure_slot_units(args):
    result = {
        "name": "unit-test",
        "error_std": 0.0,
        "probe_budget": 2,
        "gain_margin": 1.0,
        "power_margin": 1.0,
        "risk_weight": 0.0,
        "risk_power_weight": 0.0,
        "risk_invite_threshold": 0.0,
        "success_nodes": np.array([50.0, 48.0]),
        "avg_power": np.array([0.1, 0.2]),
        "episode_reward": np.array([1.0, 2.0]),
        "slots_used": np.array([5.0, 5.0]),
        "total_energy": np.array([0.5, 0.6]),
        "scheduled_nodes": np.array([10.0, 12.0]),
        "failed_nodes": np.array([1.0, 2.0]),
        "missed_opportunities": np.array([3.0, 4.0]),
        "true_opportunities": np.array([13.0, 16.0]),
        "failure_slot_count": np.array([2.0, 1.0]),
        "decision_preview_calls_per_slot": np.array([4.0, 4.0]),
        "oracle_tx_gap_mean": np.array([0.5, 0.25]),
        "effective_risk_weight": np.array([0.0, 0.0]),
    }
    row = summarize_limited_results(
        Namespace(num_nodes=args.num_nodes, num_seeds=1),
        [result],
    )[0]

    assert "failure_slot_count_mean" in row
    assert np.isclose(row["failure_slot_count_mean"], 1.5)
    assert np.isclose(row["failure_slot_rate"], 30.0)


def assert_execution_drift_is_index_stable(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    env._last_seed = args.seed
    slot_idx = 3
    execution_error_std = 0.2
    all_candidates = execution_candidates(
        env,
        args,
        indices=range(args.num_codebook_states),
        execution_error_std=execution_error_std,
        slot_idx=slot_idx,
    )
    subset_candidates = execution_candidates(
        env,
        args,
        indices=[5, 2, 5],
        execution_error_std=execution_error_std,
        slot_idx=slot_idx,
    )
    by_index = {int(candidate["irs_index"]): candidate for candidate in all_candidates}
    for candidate in subset_candidates:
        reference = by_index[int(candidate["irs_index"])]
        assert np.allclose(candidate["h_gain"], reference["h_gain"])
        assert np.allclose(candidate["required_power"], reference["required_power"])
        assert np.array_equal(candidate["valid_mask"], reference["valid_mask"])
        assert candidate["tx_this_slot"] == reference["tx_this_slot"]
    assert np.allclose(subset_candidates[0]["h_gain"], subset_candidates[2]["h_gain"])


def assert_invitation_mask_correction_count_rules():
    correction_args = Namespace(num_nodes=10, confirmation_feedback_noise_std=0.1)

    assert corrected_target_count(
        correction_args,
        remaining_count=8,
        stale_count=3,
        feedback_count=7,
        strength=1.0,
        deadband_z=0.0,
        max_delta=-1.0,
    ) == 7
    assert corrected_target_count(
        correction_args,
        remaining_count=8,
        stale_count=3,
        feedback_count=7,
        strength=0.5,
        deadband_z=0.0,
        max_delta=-1.0,
    ) == 5
    assert corrected_target_count(
        correction_args,
        remaining_count=8,
        stale_count=3,
        feedback_count=5,
        strength=1.0,
        deadband_z=2.0,
        max_delta=-1.0,
    ) == 3
    assert corrected_target_count(
        correction_args,
        remaining_count=8,
        stale_count=3,
        feedback_count=7,
        strength=1.0,
        deadband_z=0.0,
        max_delta=1.0,
    ) == 4


def assert_invitation_mask_correction_preserves_noop_masks():
    correction_args = Namespace(num_nodes=6, confirmation_feedback_noise_std=0.0)
    env = Namespace(
        transmitted_flags=np.array([False, False, True, False, False, False], dtype=bool)
    )
    candidate = {
        "valid_mask": np.array([True, False, True, False, False, True], dtype=bool),
        "h_gain": np.array([0.8, 0.9, 1.0, 0.7, 0.6, 0.5], dtype=float),
        "required_power": np.array([0.2, 0.4, 0.1, 0.5, 0.6, 0.3], dtype=float),
        "tx_this_slot": 3,
        "power_avg": 0.2,
    }
    expected_remaining_stale_mask = np.array(
        [True, False, False, False, False, True], dtype=bool
    )

    no_strength = apply_invitation_mask_correction(
        candidate,
        correction_args,
        env,
        feedback={"observed_tx_fraction": 4.0 / 6.0},
        strength=0.0,
        deadband_z=0.0,
        max_delta=-1.0,
    )
    assert np.array_equal(no_strength["valid_mask"], expected_remaining_stale_mask)
    assert no_strength["mask_correction_target_count"] == 2
    assert no_strength["mask_correction_added"] == 0
    assert no_strength["mask_correction_pruned"] == 0
    assert no_strength["mask_correction_applied"] == 0

    deadband_noop = apply_invitation_mask_correction(
        candidate,
        correction_args,
        env,
        feedback={"observed_tx_fraction": 2.0 / 6.0},
        strength=1.0,
        deadband_z=0.0,
        max_delta=-1.0,
    )
    assert np.array_equal(deadband_noop["valid_mask"], expected_remaining_stale_mask)
    assert deadband_noop["mask_correction_target_count"] == 2
    assert deadband_noop["mask_correction_applied"] == 0

    expanded = apply_invitation_mask_correction(
        candidate,
        correction_args,
        env,
        feedback={"observed_tx_fraction": 4.0 / 6.0},
        strength=1.0,
        deadband_z=0.0,
        max_delta=-1.0,
        rerank_mode=MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN,
    )
    expected_expanded_mask = np.array([True, True, False, True, True, False], dtype=bool)
    assert np.array_equal(expanded["valid_mask"], expected_expanded_mask)
    assert expanded["tx_this_slot"] == 4
    assert expanded["mask_correction_rerank_mode"] == MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN
    assert expanded["mask_correction_stale_count"] == 2
    assert expanded["mask_correction_feedback_count"] == 4
    assert expanded["mask_correction_requested_target_count"] == 4
    assert expanded["mask_correction_target_count"] == 4
    assert expanded["mask_correction_added"] == 3
    assert expanded["mask_correction_pruned"] == 1
    assert expanded["mask_correction_requested_delta"] == 2
    assert expanded["mask_correction_target_delta"] == 2
    assert expanded["mask_correction_unmet_additions"] == 0
    assert expanded["mask_correction_applied"] == 1
    assert not expanded["valid_mask"][2]

    prune_only_expansion = apply_invitation_mask_correction(
        candidate,
        correction_args,
        env,
        feedback={"observed_tx_fraction": 4.0 / 6.0},
        strength=1.0,
        deadband_z=0.0,
        max_delta=-1.0,
        rerank_mode=MASK_CORRECTION_MODE_PRUNE_ONLY,
    )
    assert np.array_equal(prune_only_expansion["valid_mask"], expected_remaining_stale_mask)
    assert prune_only_expansion["tx_this_slot"] == 2
    assert prune_only_expansion["mask_correction_rerank_mode"] == MASK_CORRECTION_MODE_PRUNE_ONLY
    assert prune_only_expansion["mask_correction_requested_target_count"] == 4
    assert prune_only_expansion["mask_correction_target_count"] == 2
    assert prune_only_expansion["mask_correction_added"] == 0
    assert prune_only_expansion["mask_correction_pruned"] == 0
    assert prune_only_expansion["mask_correction_requested_delta"] == 2
    assert prune_only_expansion["mask_correction_target_delta"] == 0
    assert prune_only_expansion["mask_correction_unmet_additions"] == 2
    assert prune_only_expansion["mask_correction_applied"] == 0

    prune_only_reduction = apply_invitation_mask_correction(
        candidate,
        correction_args,
        env,
        feedback={"observed_tx_fraction": 1.0 / 6.0},
        strength=1.0,
        deadband_z=0.0,
        max_delta=-1.0,
        rerank_mode=MASK_CORRECTION_MODE_PRUNE_ONLY,
    )
    expected_pruned_mask = np.array([True, False, False, False, False, False], dtype=bool)
    assert np.array_equal(prune_only_reduction["valid_mask"], expected_pruned_mask)
    assert prune_only_reduction["tx_this_slot"] == 1
    assert prune_only_reduction["mask_correction_requested_target_count"] == 1
    assert prune_only_reduction["mask_correction_target_count"] == 1
    assert prune_only_reduction["mask_correction_added"] == 0
    assert prune_only_reduction["mask_correction_pruned"] == 1
    assert prune_only_reduction["mask_correction_requested_delta"] == -1
    assert prune_only_reduction["mask_correction_target_delta"] == -1
    assert prune_only_reduction["mask_correction_unmet_additions"] == 0
    assert prune_only_reduction["mask_correction_applied"] == 1


def assert_execution_policy_registry_expands_labels_and_scenarios():
    registry_args = Namespace(
        policies=[
            "coverage_sparse_topk_feedback",
            "neighbor_coverage_sparse_topk_feedback",
            "adaptive_sparse_topk_v3_feedback",
            "execution_oracle",
        ],
        num_codebook_states=16,
        probe_budgets=[3],
        robust_gain_margins=[1.0],
        robust_power_margins=[1.0],
        risk_weights=[0.5],
        risk_power_weights=[0.0],
        risk_invite_thresholds=[0.75],
        adaptive_risk_base_weights=[0.5],
        opportunity_failure_costs=[1.0],
        opportunity_missed_costs=[1.0],
        opportunity_deadline_gains=[0.0],
        opportunity_backlog_gains=[0.0],
        sparse_topk_seed_multipliers=[4.1],
        sparse_topk_fractions=[0.75],
        coverage_sparse_weights=[0.5],
        coverage_sparse_power_weights=[0.0],
        adaptive_sparse_base_multiplier=2.0,
        adaptive_sparse_expanded_multiplier=3.0,
        adaptive_sparse_margin_thresholds=[0.05],
        adaptive_sparse_v2_preview_costs=[0.002],
        adaptive_sparse_v3_neighbor_radius=1,
        adaptive_sparse_v3_neighbor_count=2,
        adaptive_sparse_v3_history_count=1,
        learned_shortlist_extra_counts=[1],
        learned_set_extra_counts=[1],
        temporal_reliability_z_values=[0.0],
        mismatch_models=[
            execution_policy_registry.MISMATCH_INDEPENDENT,
            execution_policy_registry.MISMATCH_TEMPORAL_AR1,
        ],
        channel_rho_values=[0.7, 0.9],
        csi_delay_slots=[1, 3],
    )

    configs = execution_policy_registry.policy_configs(registry_args)
    assert len(configs) == 4
    assert configs[-1] == {
        "policy_name": execution_policy_registry.POLICY_EXECUTION_ORACLE,
        "budget": 16,
    }

    labels = [execution_policy_registry.policy_label(**config) for config in configs]
    assert labels[0] == (
        "Coverage-Aware Sparse-TopK Feedback Grid B=3"
        " sm=4.1 tf=0.75 cw=0.5 cpw=0"
    )
    assert labels[1] == (
        "Neighbor-Coverage Sparse-TopK Feedback Grid B=3"
        " sm=4.1 tf=0.75 cw=0.5 cpw=0 nr=1 nc=2"
    )
    assert labels[2] == (
        "Adaptive Sparse-TopK V3 Feedback Grid B=3"
        " bm=2 nr=1 nc=2 hc=1 tf=0.75"
    )
    assert labels[3] == execution_policy_registry.POLICY_EXECUTION_ORACLE

    scenarios = list(execution_policy_registry.mismatch_scenarios(registry_args))
    assert scenarios == [
        (execution_policy_registry.MISMATCH_INDEPENDENT, 0.0, 0),
        (execution_policy_registry.MISMATCH_TEMPORAL_AR1, 0.7, 1),
        (execution_policy_registry.MISMATCH_TEMPORAL_AR1, 0.7, 3),
        (execution_policy_registry.MISMATCH_TEMPORAL_AR1, 0.9, 1),
        (execution_policy_registry.MISMATCH_TEMPORAL_AR1, 0.9, 3),
    ]


def assert_execution_result_metadata_marks_diagnostic_boundaries():
    learned = result_metadata({"name": "Learned Sparse Shortlist Feedback Grid B=4"})
    assert learned["result_role"] == "diagnostic"
    assert learned["uses_hidden_training_labels"] == "true"
    assert learned["inference_uses_hidden_current_csi"] == "false"

    oracle = result_metadata({"name": "Temporal Deviation Oracle B=4"})
    assert oracle["result_role"] == "diagnostic_upper_bound"
    assert oracle["uses_hidden_training_labels"] == "false"
    assert oracle["inference_uses_hidden_current_csi"] == "true"

    deployable = result_metadata({"name": "Coverage-Aware Sparse-TopK Feedback Grid B=3"})
    assert deployable["result_role"] == "comparison_reference"
    assert deployable["uses_hidden_training_labels"] == "false"
    assert deployable["inference_uses_hidden_current_csi"] == "false"


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


def assert_active_diverse_feedback_respects_budget(args):
    for budget in (1, 2, 4, 8):
        indices = fill_diverse_codebook_indices([3], budget, args.num_codebook_states)
        assert len(indices) == budget
        assert len(set(indices)) == budget
        assert indices[0] == 3
        assert all(0 <= index < args.num_codebook_states for index in indices)

    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    env._last_seed = args.seed
    slot_before = env.current_slot
    flags_before = env.transmitted_flags.copy()
    budget = 4
    decision, preview_calls, candidate_count = choose_active_diverse_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
    )

    assert env.current_slot == slot_before
    assert np.array_equal(env.transmitted_flags, flags_before)
    assert 0 <= int(decision["irs_index"]) < args.num_codebook_states
    assert candidate_count == budget
    assert 2 * budget <= preview_calls <= 3 * budget


def assert_sparse_topk_feedback_respects_budget(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    env._last_seed = args.seed
    slot_before = env.current_slot
    flags_before = env.transmitted_flags.copy()
    budget = 4
    decision, preview_calls, candidate_count = choose_sparse_topk_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
    )

    assert env.current_slot == slot_before
    assert np.array_equal(env.transmitted_flags, flags_before)
    assert 0 <= int(decision["irs_index"]) < args.num_codebook_states
    assert candidate_count == budget
    assert preview_calls == 3 * budget
    assert decision["sparse_seed_count"] == 2 * budget
    assert decision["sparse_topk_count"] == 3

    decision, preview_calls, candidate_count = choose_sparse_topk_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
        seed_multiplier=3.0,
        topk_fraction=0.75,
    )
    assert candidate_count == budget
    assert preview_calls == 4 * budget
    assert decision["sparse_seed_count"] == 3 * budget
    assert decision["sparse_topk_count"] == 3


def assert_neighbor_coverage_sparse_topk_reallocates_stale_budget(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    env._last_seed = args.seed
    slot_before = env.current_slot
    flags_before = env.transmitted_flags.copy()
    budget = 4

    decision, preview_calls, candidate_count = (
        choose_neighbor_coverage_sparse_topk_feedback_decision(
            env,
            args,
            budget,
            slot_idx=0,
            decision_error_std=0.0,
            execution_error_std=0.0,
            episode_seed=args.seed,
            seed_multiplier=3.0,
            topk_fraction=0.75,
            coverage_weight=0.5,
            power_weight=0.0,
            neighbor_radius=1,
            neighbor_count=2,
        )
    )

    assert env.current_slot == slot_before
    assert np.array_equal(env.transmitted_flags, flags_before)
    assert 0 <= int(decision["irs_index"]) < args.num_codebook_states
    assert candidate_count == budget
    assert preview_calls == 4 * budget
    assert decision["sparse_seed_count"] == 3 * budget
    assert decision["sparse_topk_count"] == 3
    assert decision["adaptive_sparse_v3_neighbor_extra_preview_count"] == 2.0
    assert decision["adaptive_sparse_v3_selected_extra_preview_count"] <= 2.0


def assert_adaptive_sparse_topk_expands_only_on_low_margin(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    env._last_seed = args.seed
    slot_before = env.current_slot
    flags_before = env.transmitted_flags.copy()
    budget = 4

    decision, preview_calls, candidate_count = choose_adaptive_sparse_topk_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
        margin_threshold=0.0,
    )

    assert env.current_slot == slot_before
    assert np.array_equal(env.transmitted_flags, flags_before)
    assert 0 <= int(decision["irs_index"]) < args.num_codebook_states
    assert candidate_count == budget
    assert preview_calls == 3 * budget
    assert decision["adaptive_sparse_expanded"] == 0.0
    assert decision["sparse_seed_count"] == 2 * budget

    decision, preview_calls, candidate_count = choose_adaptive_sparse_topk_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
        margin_threshold=2.0,
    )

    assert candidate_count == budget
    assert preview_calls == 4 * budget
    assert decision["adaptive_sparse_expanded"] == 1.0
    assert decision["sparse_seed_count"] == 3 * budget


def assert_adaptive_sparse_topk_v2_uses_cost_and_history(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    env._last_seed = args.seed
    slot_before = env.current_slot
    flags_before = env.transmitted_flags.copy()
    budget = 4

    decision, preview_calls, candidate_count = choose_adaptive_sparse_topk_v2_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
        margin_threshold=2.0,
        preview_cost=0.0,
        history_weight=0.0,
    )

    assert env.current_slot == slot_before
    assert np.array_equal(env.transmitted_flags, flags_before)
    assert candidate_count == budget
    assert preview_calls == 4 * budget
    assert decision["adaptive_sparse_expanded"] == 1.0
    assert decision["sparse_seed_count"] == 3 * budget
    assert decision["adaptive_sparse_cost_penalty"] == 0.0

    decision, preview_calls, candidate_count = choose_adaptive_sparse_topk_v2_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
        confirmed_history=[0, 0, 0],
        margin_threshold=2.0,
        preview_cost=0.0,
        history_weight=5.0,
    )

    assert candidate_count == budget
    assert preview_calls == 3 * budget
    assert decision["adaptive_sparse_expanded"] == 0.0
    assert decision["adaptive_sparse_history_stability"] == 1.0

    decision, preview_calls, candidate_count = choose_adaptive_sparse_topk_v2_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
        margin_threshold=2.0,
        preview_cost=1.0,
        history_weight=0.0,
    )

    assert candidate_count == budget
    assert preview_calls == 3 * budget
    assert decision["adaptive_sparse_expanded"] == 0.0
    assert decision["adaptive_sparse_cost_penalty"] > 0.0


def assert_adaptive_sparse_topk_v3_uses_history_and_neighbors(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    env._last_seed = args.seed
    slot_before = env.current_slot
    flags_before = env.transmitted_flags.copy()
    budget = 4

    decision, preview_calls, candidate_count = choose_adaptive_sparse_topk_v3_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
        confirmed_history=[0, 0, 0],
        neighbor_count=2,
        history_count=1,
    )

    assert env.current_slot == slot_before
    assert np.array_equal(env.transmitted_flags, flags_before)
    assert candidate_count == budget
    assert 3 * budget <= preview_calls <= 4 * budget
    assert decision["adaptive_sparse_expanded"] == 0.0
    assert decision["adaptive_sparse_v3_history_prior_used"] == 1.0
    assert decision["sparse_seed_count"] == 2 * budget
    assert decision["adaptive_sparse_v3_selected_extra_preview_count"] <= budget

    decision, preview_calls, candidate_count = choose_adaptive_sparse_topk_v3_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
        confirmed_history=[],
        neighbor_count=0,
        history_count=0,
    )

    assert candidate_count == budget
    assert preview_calls == 3 * budget
    assert decision["adaptive_sparse_v3_history_prior_used"] == 0.0
    assert decision["adaptive_sparse_v3_neighbor_extra_preview_count"] == 0.0


def assert_learned_sparse_shortlist_respects_budget(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    env._last_seed = args.seed
    slot_before = env.current_slot
    flags_before = env.transmitted_flags.copy()
    budget = 4
    model = {
        "weights": np.zeros(len(LEARNED_SHORTLIST_FEATURE_NAMES), dtype=float),
        "bias": 0.0,
        "feature_mean": np.zeros(len(LEARNED_SHORTLIST_FEATURE_NAMES), dtype=float),
        "feature_scale": np.ones(len(LEARNED_SHORTLIST_FEATURE_NAMES), dtype=float),
    }

    decision, preview_calls, candidate_count = choose_learned_sparse_shortlist_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
        confirmed_history=[0, 0, 0],
        channel_rho=0.9,
        csi_delay_slots=1,
        extra_count=1,
        model=model,
    )

    assert env.current_slot == slot_before
    assert np.array_equal(env.transmitted_flags, flags_before)
    assert candidate_count == budget
    assert preview_calls == 3 * budget + 1
    assert decision["sparse_seed_count"] == 2 * budget
    assert decision["learned_shortlist_extra_count"] == 1
    assert decision["learned_shortlist_selected_extra_preview_count"] <= 1


def assert_learned_set_shortlist_respects_budget(args):
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )
    env.reset(seed=args.seed)
    env._last_seed = args.seed
    slot_before = env.current_slot
    flags_before = env.transmitted_flags.copy()
    budget = 4
    model = {
        "weights": np.zeros(len(LEARNED_SET_SHORTLIST_FEATURE_NAMES), dtype=float),
        "bias": 0.0,
        "feature_mean": np.zeros(len(LEARNED_SET_SHORTLIST_FEATURE_NAMES), dtype=float),
        "feature_scale": np.ones(len(LEARNED_SET_SHORTLIST_FEATURE_NAMES), dtype=float),
    }

    decision, preview_calls, candidate_count = choose_learned_set_shortlist_feedback_decision(
        env,
        args,
        budget,
        slot_idx=0,
        decision_error_std=0.0,
        execution_error_std=0.0,
        episode_seed=args.seed,
        confirmed_history=[0, 0, 0],
        channel_rho=0.9,
        csi_delay_slots=1,
        extra_count=2,
        model=model,
    )

    assert env.current_slot == slot_before
    assert np.array_equal(env.transmitted_flags, flags_before)
    assert candidate_count == budget
    assert 3 * budget <= preview_calls <= 3 * budget + 2
    assert decision["sparse_seed_count"] == 2 * budget
    assert decision["learned_shortlist_extra_count"] == 2
    assert decision["learned_shortlist_selected_extra_preview_count"] <= 2
    assert decision["learned_set_shortlist_variant_count"] > 1


def main():
    args = make_args()
    assert_preview_has_no_side_effects(args)
    assert_codebook_features_match_preview(args)
    assert_noisy_codebook_features_are_seeded_and_clipped(args)
    assert_negative_feature_noise_is_rejected(args)
    assert_invalid_environment_dimensions_are_rejected()
    assert_environment_step_clips_out_of_bounds_actions(args)
    assert_mainline_cli_validators_reject_invalid_args()
    assert_power_tie_matches_greedy(args)
    assert_episode_seed_generation_is_stable(args)
    assert_temporal_trace_supports_prehistory(args)
    assert_partial_probe_selectors_respect_budget(args)
    assert_estimated_preview_zero_error_matches_exact(args)
    assert_limited_csi_zero_error_matches_true(args)
    assert_limited_csi_execution_only_counts_invited_nodes(args)
    assert_limited_csi_summary_names_failure_slot_units(args)
    assert_execution_drift_is_index_stable(args)
    assert_invitation_mask_correction_count_rules()
    assert_invitation_mask_correction_preserves_noop_masks()
    assert_execution_policy_registry_expands_labels_and_scenarios()
    assert_execution_result_metadata_marks_diagnostic_boundaries()
    assert_risk_aware_candidate_filters_low_reliability(args)
    assert_adaptive_risk_weight_tracks_uncertainty_and_deadline(args)
    assert_bandit_feedback_observes_only_aggregate_metrics(args)
    assert_bandit_probe_selectors_respect_budget(args)
    assert_active_diverse_feedback_respects_budget(args)
    assert_sparse_topk_feedback_respects_budget(args)
    assert_neighbor_coverage_sparse_topk_reallocates_stale_budget(args)
    assert_adaptive_sparse_topk_expands_only_on_low_margin(args)
    assert_adaptive_sparse_topk_v2_uses_cost_and_history(args)
    assert_adaptive_sparse_topk_v3_uses_history_and_neighbors(args)
    assert_learned_sparse_shortlist_respects_budget(args)
    assert_learned_set_shortlist_respects_budget(args)
    print("smoke checks passed")


if __name__ == "__main__":
    main()
