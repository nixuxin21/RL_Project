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


def main():
    args = make_args()
    assert_preview_has_no_side_effects(args)
    assert_codebook_features_match_preview(args)
    assert_power_tie_matches_greedy(args)
    assert_episode_seed_generation_is_stable(args)
    print("smoke checks passed")


if __name__ == "__main__":
    main()
