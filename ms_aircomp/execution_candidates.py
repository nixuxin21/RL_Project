"""Execution-stage candidate builders for MS-AirComp mismatch experiments."""

import numpy as np

import evaluate_limited_csi_ms_aircomp as limited
from ms_aircomp.channel_models import drift_channels, execution_rng

__all__ = [
    "choose_execution_oracle",
    "execution_candidate_for_decision",
    "execution_candidates",
    "execution_oracle_candidate",
]


def execution_candidates(env, args, indices=None, execution_error_std=0.0, slot_idx=0, no_irs=False):
    """Build drifted execution candidates for selected indices or no-IRS."""
    rng = execution_rng(getattr(env, "_last_seed", None), execution_error_std, slot_idx, no_irs=no_irs)
    if no_irs:
        h_ref = limited.effective_channels(env, no_irs=True)
        h_exec = drift_channels(h_ref, execution_error_std, rng)
        return [limited.build_candidate(env, args, -2, h_exec[0], no_irs=True)]

    clean_indices = [int(np.clip(index, 0, args.num_codebook_states - 1)) for index in indices]
    h_ref = limited.effective_channels(env, clean_indices)
    h_exec = drift_channels(h_ref, execution_error_std, rng)
    return [
        limited.build_candidate(env, args, index, h_exec[row_idx])
        for row_idx, index in enumerate(clean_indices)
    ]


def execution_candidate_for_decision(env, args, decision_candidate, execution_error_std, slot_idx):
    """Return drifted execution candidate matching a decision."""
    irs_index = int(decision_candidate["irs_index"])
    if irs_index == -2:
        return execution_candidates(
            env,
            args,
            execution_error_std=execution_error_std,
            slot_idx=slot_idx,
            no_irs=True,
        )[0]
    return execution_candidates(
        env,
        args,
        indices=[irs_index],
        execution_error_std=execution_error_std,
        slot_idx=slot_idx,
    )[0]


def execution_oracle_candidate(env, args, execution_error_std, slot_idx):
    """Return hidden oracle candidate under the drifted execution channel."""
    candidates = execution_candidates(
        env,
        args,
        indices=range(args.num_codebook_states),
        execution_error_std=execution_error_std,
        slot_idx=slot_idx,
    )
    return limited.best_candidate(candidates)


def choose_execution_oracle(env, args, execution_error_std, slot_idx):
    """Choose and invite using the hidden execution channel."""
    oracle = execution_oracle_candidate(env, args, execution_error_std, slot_idx)
    return oracle, args.num_codebook_states, args.num_codebook_states
