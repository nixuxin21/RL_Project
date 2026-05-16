"""Current aggregate-feedback confirmation flow."""

import evaluate_limited_csi_ms_aircomp as limited
from ms_aircomp.channel_models import apply_channel_state, capture_channel_state
from ms_aircomp.execution_candidates import execution_candidates
from ms_aircomp.feedback import confirmation_feedback, confirmed_index_from_feedback

__all__ = ["confirm_index_with_current_feedback"]


def confirm_index_with_current_feedback(
    env,
    args,
    selected_indices,
    execution_error_std,
    slot_idx,
    episode_seed,
    execution_state=None,
    feedback_salt=41,
):
    """Select one IRS index from selected candidates using current aggregate feedback."""
    decision_snapshot = capture_channel_state(env)
    try:
        if execution_state is not None:
            apply_channel_state(env, execution_state)
        current_candidates = [
            execution_candidates(
                env,
                args,
                indices=[index],
                execution_error_std=execution_error_std,
                slot_idx=slot_idx,
            )[0]
            for index in selected_indices
        ]
        feedback_rng = limited.stable_rng(
            episode_seed,
            args.confirmation_feedback_noise_std,
            limited.POLICY_RANDOM_PROBE,
            len(selected_indices),
            salt=int(feedback_salt) + slot_idx,
        )
        feedbacks = [
            confirmation_feedback(
                candidate,
                args,
                args.confirmation_feedback_noise_std,
                args.confirmation_feedback_power_weight,
                feedback_rng,
            )
            for candidate in current_candidates
        ]
        confirmed_index = confirmed_index_from_feedback(selected_indices, feedbacks)
    finally:
        apply_channel_state(env, decision_snapshot)
    return int(confirmed_index), feedbacks
