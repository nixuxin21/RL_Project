"""Episode-level optional metrics and raw slot rows for execution mismatch runs."""

import numpy as np

from ms_aircomp.execution_slot_logging import (
    build_diagnostic_record,
    build_scenario_summary_record,
    build_structured_slot_record,
)
from ms_aircomp.posterior_viability import (
    posterior_summary_from_matrix,
    posterior_viability_matrix,
)

__all__ = [
    "append_decision_metric_values",
    "append_episode_metric_means",
    "append_scenario_summary_record",
    "append_structured_slot_records",
    "build_execution_slot_row",
    "empty_decision_metric_buffers",
    "episode_metric_summary",
    "metric_arrays",
]


DECISION_METRIC_KEYS = (
    "posterior_prior_expected_count",
    "posterior_prior_entropy_mean",
    "posterior_selected_marginal_fraction",
    "posterior_selected_overlap_fraction",
    "posterior_refinement_prior_expected_count",
    "posterior_refinement_posterior_expected_count",
    "posterior_refinement_target_count",
    "posterior_refinement_added",
    "posterior_refinement_pruned",
    "posterior_refinement_applied",
    "aircomp_raw_mse",
    "aircomp_missing_device_mse",
    "aircomp_failed_invitation_mse",
    "aircomp_power_clipping_mse",
    "aircomp_receiver_noise_mse",
    "aircomp_target_variance",
    "aircomp_power_clipped_count",
    "aircomp_power_clipping_rate",
    "aircomp_pmax_device_count",
    "aircomp_energy_per_success",
    "stale_preview_calls",
    "current_probe_calls",
    "total_probe_calls",
    "total_protocol_cost",
)

EPISODE_SUM_KEYS = {
    "aircomp_power_clipped_count",
    "aircomp_pmax_device_count",
    "stale_preview_calls",
    "current_probe_calls",
    "total_probe_calls",
    "total_protocol_cost",
}


def empty_decision_metric_buffers():
    """Return empty buffers for optional per-slot decision diagnostics."""
    return {key: [] for key in DECISION_METRIC_KEYS}


def _decision_metric_values(decision, info=None, preview_calls=0, args=None):
    values = {key: float(decision.get(key, 0.0)) for key in DECISION_METRIC_KEYS}
    if info is None:
        return values
    current_probe_calls = int(decision.get("confirmation_feedback_count", 0))
    stale_preview_calls = max(int(preview_calls) - current_probe_calls, 0)
    slot_cost = (
        float(getattr(args, "protocol_stale_preview_cost", 1.0)) * stale_preview_calls
        + float(getattr(args, "protocol_current_probe_cost", 1.0)) * current_probe_calls
        + float(getattr(args, "protocol_execution_slot_cost", 1.0))
    )
    values.update(
        {
            "aircomp_raw_mse": float(info.get("aircomp_raw_mse", 0.0)),
            "aircomp_missing_device_mse": float(info.get("aircomp_missing_device_mse", 0.0)),
            "aircomp_failed_invitation_mse": float(info.get("aircomp_failed_invitation_mse", 0.0)),
            "aircomp_power_clipping_mse": float(info.get("aircomp_power_clipping_mse", 0.0)),
            "aircomp_receiver_noise_mse": float(info.get("aircomp_receiver_noise_mse", 0.0)),
            "aircomp_target_variance": float(info.get("aircomp_target_variance", 0.0)),
            "aircomp_power_clipped_count": float(info.get("power_clipped_count", 0.0)),
            "aircomp_power_clipping_rate": float(info.get("power_clipping_rate", 0.0)),
            "aircomp_pmax_device_count": float(info.get("pmax_device_count", 0.0)),
            "aircomp_energy_per_success": float(info.get("energy_per_success", 0.0)),
            "stale_preview_calls": float(stale_preview_calls),
            "current_probe_calls": float(current_probe_calls),
            "total_probe_calls": float(preview_calls),
            "total_protocol_cost": float(slot_cost),
        }
    )
    return values


def append_decision_metric_values(buffers, decision, info=None, preview_calls=0, args=None):
    """Append optional posterior/count-refinement diagnostics from one slot decision."""
    values = _decision_metric_values(decision, info=info, preview_calls=preview_calls, args=args)
    for key, value in values.items():
        buffers[key].append(value)
    return values


def append_episode_metric_means(result_buffers, episode_buffers):
    """Append one episode mean for each optional decision diagnostic."""
    summary = episode_metric_summary(episode_buffers)
    for key in episode_buffers:
        result_buffers[key].append(summary[key])
    return summary


def episode_metric_summary(episode_buffers):
    """Return JSON/CSV friendly episode metrics with mean or sum semantics."""
    summary = {}
    for key, values in episode_buffers.items():
        if key in EPISODE_SUM_KEYS:
            summary[key] = float(np.sum(values)) if values else 0.0
        else:
            summary[key] = float(np.mean(values)) if values else 0.0
    return summary


def metric_arrays(result_buffers):
    """Convert optional metric buffers to result arrays for seed aggregation."""
    return {
        key: np.asarray(values, dtype=float)
        for key, values in result_buffers.items()
    }


def append_structured_slot_records(
    legacy_slot_rows,
    structured_slot_rows,
    diagnostic_rows,
    log_base,
    *,
    episode_index,
    episode_seed,
    slot_idx,
    decision,
    info,
    execution_oracle,
    preview_calls,
    oracle_gap,
    decision_metric_values,
    env=None,
    stale_state=None,
):
    """Append legacy, rich slot, and optional diagnostic records for one slot."""
    legacy_slot_rows.append(
        build_execution_slot_row(
            decision_error_std=log_base["decision_error_std"],
            execution_error_std=log_base["execution_error_std"],
            mismatch_model=log_base["mismatch_model"],
            channel_rho=log_base["channel_rho"],
            csi_delay_slots=log_base["csi_delay_slots"],
            display_name=log_base["method_name"],
            budget=log_base["budget"],
            run_index=log_base["run_index"],
            run_seed=log_base["run_seed"],
            episode_index=episode_index,
            episode_seed=episode_seed,
            slot_idx=slot_idx,
            decision=decision,
            info=info,
            preview_calls=preview_calls,
            oracle_gap=oracle_gap,
            decision_metric_values=decision_metric_values,
        )
    )
    common = {
        **log_base,
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "slot_idx": slot_idx,
        "decision": decision,
        "execution_oracle": execution_oracle,
    }
    posterior_summary = None
    if stale_state is not None and env is not None:
        args = log_base["args"]
        probabilities = posterior_viability_matrix(
            stale_state=stale_state,
            codebook=env.codebook,
            args=args,
            p_max=env.P_max,
            channel_rho=log_base["channel_rho"],
            csi_delay_slots=log_base["csi_delay_slots"],
            posterior_mode=args.posterior_mode,
            posterior_num_samples=args.posterior_num_samples,
            posterior_clip_eps=args.posterior_clip_eps,
            posterior_seed_offset=args.posterior_seed_offset,
            episode_seed=episode_seed,
            slot_idx=slot_idx,
        )
        posterior_summary = posterior_summary_from_matrix(
            probabilities,
            info["remaining_mask_before"],
            posterior_mode=args.posterior_mode,
            posterior_num_samples=args.posterior_num_samples,
            posterior_clip_eps=args.posterior_clip_eps,
            posterior_seed_offset=args.posterior_seed_offset,
        )
    structured_slot_rows.append(
        build_structured_slot_record(
            **common,
            info=info,
            preview_calls=preview_calls,
            oracle_gap=oracle_gap,
            full_device_lists=log_base["args"].log_full_device_lists,
        )
    )
    diagnostic_rows.append(
        build_diagnostic_record(
            **common,
            posterior_summary=posterior_summary,
            full_device_lists=log_base["args"].log_full_device_lists,
        )
    )


def append_scenario_summary_record(
    scenario_rows,
    log_base,
    *,
    episode_index,
    episode_seed,
    completion,
    slots_used,
    failed_invitations,
    missed_opportunities,
    oracle_tx_count,
    achieved_tx_count,
    oracle_gap_total,
    oracle_gap_mean,
    nmse,
    energy,
    total_overhead,
    extra_metrics=None,
):
    """Append one raw per-seed/per-scenario episode summary row."""
    scenario_rows.append(
        build_scenario_summary_record(
            **log_base,
            episode_index=episode_index,
            episode_seed=episode_seed,
            completion=completion,
            slots_used=slots_used,
            failed_invitations=failed_invitations,
            missed_opportunities=missed_opportunities,
            oracle_tx_count=oracle_tx_count,
            achieved_tx_count=achieved_tx_count,
            oracle_gap_total=oracle_gap_total,
            oracle_gap_mean=oracle_gap_mean,
            nmse=nmse,
            energy=energy,
            total_overhead=total_overhead,
            extra_metrics=extra_metrics,
        )
    )


def build_execution_slot_row(
    *,
    decision_error_std,
    execution_error_std,
    mismatch_model,
    channel_rho,
    csi_delay_slots,
    display_name,
    budget,
    run_index,
    run_seed,
    episode_index,
    episode_seed,
    slot_idx,
    decision,
    info,
    preview_calls,
    oracle_gap,
    decision_metric_values,
):
    """Build one raw per-seed/per-scenario/per-slot CSV row."""
    row = {
        "decision_error_std": float(decision_error_std),
        "execution_error_std": float(execution_error_std),
        "mismatch_model": mismatch_model,
        "channel_rho": float(channel_rho),
        "csi_delay_slots": int(csi_delay_slots),
        "policy": display_name,
        "probe_budget": int(budget),
        "run_index": int(run_index),
        "run_seed": "" if run_seed is None else int(run_seed),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "slot_idx": int(slot_idx),
        "irs_index": int(decision.get("irs_index", -99)),
        "confirmed_irs_index": int(
            decision.get("confirmed_irs_index", decision.get("irs_index", -99))
        ),
        "scheduled_this_slot": int(info["scheduled_this_slot"]),
        "tx_this_slot": int(info["tx_this_slot"]),
        "failed_this_slot": int(info["failed_this_slot"]),
        "missed_opportunity_this_slot": int(info["missed_opportunity_this_slot"]),
        "true_opportunity_this_slot": int(info["true_opportunity_this_slot"]),
        "total_tx": int(info["total_tx"]),
        "is_complete": bool(info["is_complete"]),
        "termination_reason": info["termination_reason"],
        "power_avg": float(info["power_avg"]),
        "attempted_energy": float(info["attempted_energy"]),
        "aircomp_nmse": float(info.get("aircomp_nmse", 0.0)),
        "aircomp_raw_mse": float(info.get("aircomp_raw_mse", 0.0)),
        "aircomp_missing_device_mse": float(info.get("aircomp_missing_device_mse", 0.0)),
        "aircomp_failed_invitation_mse": float(info.get("aircomp_failed_invitation_mse", 0.0)),
        "aircomp_power_clipping_mse": float(info.get("aircomp_power_clipping_mse", 0.0)),
        "aircomp_receiver_noise_mse": float(info.get("aircomp_receiver_noise_mse", 0.0)),
        "aircomp_target_variance": float(info.get("aircomp_target_variance", 0.0)),
        "aircomp_power_clipped_count": int(info.get("power_clipped_count", 0)),
        "aircomp_power_clipping_rate": float(info.get("power_clipping_rate", 0.0)),
        "aircomp_pmax_device_count": int(info.get("pmax_device_count", 0)),
        "aircomp_energy_per_success": float(info.get("energy_per_success", 0.0)),
        "decision_preview_calls": int(preview_calls),
        "oracle_tx_gap": float(oracle_gap),
    }
    row.update(decision_metric_values)
    return row
