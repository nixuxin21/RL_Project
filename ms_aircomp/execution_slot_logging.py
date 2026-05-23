"""Raw CSV/JSONL logging for execution-mismatch experiments."""

import csv
from datetime import UTC, datetime
import hashlib
import json
import os
import subprocess

from ms_aircomp.experiment_utils import ensure_parent_dir

__all__ = [
    "SCENARIO_SUMMARY_FIELDS",
    "SLOT_CSV_FIELDS",
    "STRUCTURED_SLOT_FIELDS",
    "build_diagnostic_record",
    "build_run_metadata_record",
    "build_scenario_summary_record",
    "build_structured_slot_record",
    "structured_log_context",
    "write_slot_csv",
    "write_structured_logs",
]


SLOT_CSV_FIELDS = [
    "decision_error_std",
    "execution_error_std",
    "mismatch_model",
    "channel_rho",
    "csi_delay_slots",
    "policy",
    "probe_budget",
    "run_index",
    "run_seed",
    "episode_index",
    "episode_seed",
    "slot_idx",
    "irs_index",
    "confirmed_irs_index",
    "scheduled_this_slot",
    "tx_this_slot",
    "failed_this_slot",
    "missed_opportunity_this_slot",
    "true_opportunity_this_slot",
    "total_tx",
    "is_complete",
    "termination_reason",
    "power_avg",
    "attempted_energy",
    "aircomp_nmse",
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
    "decision_preview_calls",
    "stale_preview_calls",
    "current_probe_calls",
    "total_probe_calls",
    "total_protocol_cost",
    "oracle_tx_gap",
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
]

SCENARIO_SUMMARY_FIELDS = [
    "timestamp_utc",
    "git_commit",
    "git_dirty",
    "config_name",
    "method_name",
    "policy_name",
    "run_index",
    "run_seed",
    "episode_index",
    "episode_seed",
    "mismatch_model",
    "rho",
    "csi_delay",
    "K",
    "num_slots",
    "num_irs_elements",
    "num_codebook_states",
    "probe_budget",
    "completion",
    "perfect_indicator",
    "slots_used",
    "failed_invitations",
    "missed_opportunities",
    "oracle_tx_count",
    "achieved_tx_count",
    "oracle_gap",
    "oracle_gap_mean",
    "nmse",
    "aircomp_raw_mse",
    "aircomp_missing_device_mse",
    "aircomp_failed_invitation_mse",
    "aircomp_power_clipping_mse",
    "aircomp_receiver_noise_mse",
    "aircomp_target_variance",
    "energy",
    "energy_per_success",
    "power_clipped_count",
    "power_clipping_rate",
    "pmax_device_count",
    "total_overhead",
    "stale_preview_calls",
    "current_probe_calls",
    "total_probe_calls",
    "total_protocol_cost",
]

STRUCTURED_SLOT_FIELDS = [
    "timestamp_utc",
    "git_commit",
    "config_name",
    "method_name",
    "policy_name",
    "run_index",
    "run_seed",
    "episode_index",
    "episode_seed",
    "mismatch_model",
    "rho",
    "csi_delay",
    "slot_idx",
    "K",
    "num_slots",
    "num_irs_elements",
    "num_codebook_states",
    "probe_budget",
    "chosen_irs_state",
    "confirmed_irs_state",
    "candidate_irs_states",
    "stale_predicted_count_per_candidate",
    "aggregate_feedback_count_per_candidate",
    "aggregate_feedback_score_per_candidate",
    "selected_feedback_count",
    "selected_feedback_score",
    "posterior_probe_selected_indices",
    "posterior_probe_objective_value",
    "posterior_probe_expected_best_count",
    "posterior_probe_coverage_score",
    "posterior_probe_expected_count_per_selected_state",
    "posterior_probe_candidate_prefilter_size",
    "posterior_probe_samples",
    "posterior_probe_beta",
    "posterior_probe_objective",
    "invited_device_mask_hex",
    "invited_device_mask_hash",
    "invited_device_ids",
    "feasible_device_mask_hex",
    "feasible_device_mask_hash",
    "feasible_device_ids",
    "successful_tx_device_mask_hex",
    "successful_tx_device_mask_hash",
    "successful_tx_device_ids",
    "failed_invited_count",
    "missed_feasible_count",
    "remaining_device_count",
    "remaining_device_count_after",
    "tx_this_slot",
    "scheduled_this_slot",
    "true_opportunity_this_slot",
    "power_avg",
    "energy",
    "nmse",
    "aircomp_raw_mse",
    "aircomp_missing_device_mse",
    "aircomp_failed_invitation_mse",
    "aircomp_power_clipping_mse",
    "aircomp_receiver_noise_mse",
    "aircomp_target_variance",
    "energy_per_success",
    "power_clipped_count",
    "power_clipping_rate",
    "pmax_device_count",
    "slot_overhead",
    "stale_preview_calls",
    "current_probe_calls",
    "total_probe_calls",
    "total_protocol_cost",
    "oracle_tx_count",
    "oracle_gap",
]


def write_slot_csv(path, rows):
    """Write raw per-seed, per-scenario, per-slot rows with a stable schema."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SLOT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved slot rows: {path}")


def structured_log_context(cwd):
    """Return run-context metadata available at logging time."""
    commit = ""
    dirty = ""
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if commit_result.returncode == 0:
            commit = commit_result.stdout.strip()
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if status_result.returncode == 0:
            dirty = str(bool(status_result.stdout.strip())).lower()
    except OSError:
        commit = ""
        dirty = ""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
    }


def _json_cell(value):
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mask_bits(mask):
    if mask is None:
        return ""
    return "".join("1" if bool(value) else "0" for value in mask)


def _mask_payload(mask, prefix, full_device_lists=False):
    bits = _mask_bits(mask)
    payload = {
        f"{prefix}_device_mask_hex": hex(int(bits, 2)) if bits else "",
        f"{prefix}_device_mask_hash": hashlib.sha1(bits.encode("ascii")).hexdigest()
        if bits
        else "",
        f"{prefix}_device_ids": "",
    }
    if full_device_lists and bits:
        payload[f"{prefix}_device_ids"] = _json_cell(
            [idx for idx, bit in enumerate(bits) if bit == "1"]
        )
    return payload


def _method_hyperparameters(args, policy_name):
    posterior_invitation_rule = getattr(args, "posterior_invitation_rule", "")
    if "Count-Conditioned Invitation" in str(policy_name):
        posterior_invitation_rule = "posterior_top_y"
    return {
        "policy_name": policy_name,
        "g_th": float(args.g_th),
        "alpha_th": float(args.alpha_th),
        "confirmation_feedback_noise_std": float(args.confirmation_feedback_noise_std),
        "confirmation_feedback_power_weight": float(args.confirmation_feedback_power_weight),
        "sparse_topk_seed_multipliers": list(getattr(args, "sparse_topk_seed_multipliers", [])),
        "sparse_topk_fractions": list(getattr(args, "sparse_topk_fractions", [])),
        "coverage_sparse_weights": list(getattr(args, "coverage_sparse_weights", [])),
        "coverage_sparse_power_weights": list(
            getattr(args, "coverage_sparse_power_weights", [])
        ),
        "posterior_sample_counts": list(getattr(args, "posterior_sample_counts", [])),
        "posterior_uncertainty_scales": list(
            getattr(args, "posterior_uncertainty_scales", [])
        ),
        "posterior_probe_uncertainty_weights": list(
            getattr(args, "posterior_probe_uncertainty_weights", [])
        ),
        "posterior_count_refinement_strengths": list(
            getattr(args, "posterior_count_refinement_strengths", [])
        ),
        "posterior_mean_mode": getattr(args, "posterior_mean_mode", ""),
        "posterior_invitation_rule": posterior_invitation_rule,
        "posterior_invitation_threshold": float(
            getattr(args, "posterior_invitation_threshold", 0.0)
        ),
        "posterior_mode": getattr(args, "posterior_mode", ""),
        "posterior_num_samples": int(getattr(args, "posterior_num_samples", 0)),
        "posterior_clip_eps": float(getattr(args, "posterior_clip_eps", 0.0)),
        "posterior_seed_offset": int(getattr(args, "posterior_seed_offset", 0)),
        "posterior_cardinality_policy": getattr(args, "posterior_cardinality_policy", ""),
        "posterior_cumulative_probability_target": float(
            getattr(args, "posterior_cumulative_probability_target", 0.0)
        ),
        "posterior_lambda_fail": float(getattr(args, "posterior_lambda_fail", 0.0)),
        "posterior_lambda_miss": float(getattr(args, "posterior_lambda_miss", 0.0)),
        "probing_policy": getattr(args, "probing_policy", ""),
        "posterior_probe_budget": int(getattr(args, "posterior_probe_budget", 0)),
        "posterior_probe_samples": int(getattr(args, "posterior_probe_samples", 0)),
        "posterior_probe_beta": float(getattr(args, "posterior_probe_beta", 0.0)),
        "posterior_probe_seed_offset": int(getattr(args, "posterior_probe_seed_offset", 0)),
        "posterior_probe_candidate_prefilter_size": int(
            getattr(args, "posterior_probe_candidate_prefilter_size", 0)
        ),
        "posterior_probe_objective": getattr(args, "posterior_probe_objective", ""),
        "aircomp_signal_model": getattr(args, "aircomp_signal_model", ""),
        "aircomp_signal_variance": float(getattr(args, "aircomp_signal_variance", 0.0)),
        "protocol_stale_preview_cost": float(getattr(args, "protocol_stale_preview_cost", 1.0)),
        "protocol_current_probe_cost": float(getattr(args, "protocol_current_probe_cost", 1.0)),
        "protocol_execution_slot_cost": float(getattr(args, "protocol_execution_slot_cost", 1.0)),
    }


def build_run_metadata_record(
    *,
    args,
    env,
    context,
    config_name,
    method_name,
    policy_name,
    run_index,
    run_seed,
    decision_error_std=None,
    execution_error_std=None,
    mismatch_model,
    channel_rho,
    csi_delay_slots,
    budget,
):
    """Build one JSONL metadata record for a method/scenario/run seed."""
    return {
        **context,
        "config_name": config_name,
        "method_name": method_name,
        "policy_name": policy_name,
        "run_index": int(run_index),
        "seed": None if run_seed is None else int(run_seed),
        "decision_error_std": None if decision_error_std is None else float(decision_error_std),
        "execution_error_std": None if execution_error_std is None else float(execution_error_std),
        "mismatch_model": mismatch_model,
        "rho": float(channel_rho),
        "csi_delay": int(csi_delay_slots),
        "K": int(args.num_nodes),
        "num_slots": int(args.num_slots),
        "num_irs_elements": int(args.num_irs_elements),
        "num_codebook_states": int(args.num_codebook_states),
        "probe_budget": int(budget),
        "noise_variance": float(env.noise_var),
        "thresholds": {
            "g_th": float(args.g_th),
            "alpha_th": float(args.alpha_th),
        },
        "method_hyperparameters": _method_hyperparameters(args, policy_name),
    }


def build_scenario_summary_record(
    *,
    args,
    context,
    config_name,
    method_name,
    policy_name,
    run_index,
    run_seed,
    decision_error_std=None,
    execution_error_std=None,
    episode_index,
    episode_seed,
    mismatch_model,
    channel_rho,
    csi_delay_slots,
    budget,
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
    """Build one per-seed/per-scenario episode summary row."""
    row = {
        **context,
        "config_name": config_name,
        "method_name": method_name,
        "policy_name": policy_name,
        "run_index": int(run_index),
        "run_seed": "" if run_seed is None else int(run_seed),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "mismatch_model": mismatch_model,
        "rho": float(channel_rho),
        "csi_delay": int(csi_delay_slots),
        "K": int(args.num_nodes),
        "num_slots": int(args.num_slots),
        "num_irs_elements": int(args.num_irs_elements),
        "num_codebook_states": int(args.num_codebook_states),
        "probe_budget": int(budget),
        "completion": bool(completion),
        "perfect_indicator": bool(int(achieved_tx_count) >= int(args.num_nodes)),
        "slots_used": int(slots_used),
        "failed_invitations": int(failed_invitations),
        "missed_opportunities": int(missed_opportunities),
        "oracle_tx_count": float(oracle_tx_count),
        "achieved_tx_count": int(achieved_tx_count),
        "oracle_gap": float(oracle_gap_total),
        "oracle_gap_mean": float(oracle_gap_mean),
        "nmse": "" if nmse is None else float(nmse),
        "energy": "" if energy is None else float(energy),
        "total_overhead": "" if total_overhead is None else float(total_overhead),
    }
    if extra_metrics:
        row.update(
            {
                "aircomp_raw_mse": float(extra_metrics.get("aircomp_raw_mse", 0.0)),
                "aircomp_missing_device_mse": float(extra_metrics.get("aircomp_missing_device_mse", 0.0)),
                "aircomp_failed_invitation_mse": float(extra_metrics.get("aircomp_failed_invitation_mse", 0.0)),
                "aircomp_power_clipping_mse": float(extra_metrics.get("aircomp_power_clipping_mse", 0.0)),
                "aircomp_receiver_noise_mse": float(extra_metrics.get("aircomp_receiver_noise_mse", 0.0)),
                "aircomp_target_variance": float(extra_metrics.get("aircomp_target_variance", 0.0)),
                "energy_per_success": float(extra_metrics.get("aircomp_energy_per_success", 0.0)),
                "power_clipped_count": float(extra_metrics.get("aircomp_power_clipped_count", 0.0)),
                "power_clipping_rate": float(extra_metrics.get("aircomp_power_clipping_rate", 0.0)),
                "pmax_device_count": float(extra_metrics.get("aircomp_pmax_device_count", 0.0)),
                "stale_preview_calls": float(extra_metrics.get("stale_preview_calls", 0.0)),
                "current_probe_calls": float(extra_metrics.get("current_probe_calls", 0.0)),
                "total_probe_calls": float(extra_metrics.get("total_probe_calls", 0.0)),
                "total_protocol_cost": float(extra_metrics.get("total_protocol_cost", 0.0)),
            }
        )
    return row


def build_structured_slot_record(
    *,
    args,
    context,
    config_name,
    method_name,
    policy_name,
    run_index,
    run_seed,
    decision_error_std=None,
    execution_error_std=None,
    episode_index,
    episode_seed,
    mismatch_model,
    channel_rho,
    csi_delay_slots,
    budget,
    slot_idx,
    decision,
    info,
    execution_oracle,
    preview_calls,
    oracle_gap,
    full_device_lists=False,
):
    """Build one rich slot-level CSV row for publication-grade analysis."""
    current_probe_calls = int(decision.get("confirmation_feedback_count", 0))
    stale_preview_calls = max(int(preview_calls) - current_probe_calls, 0)
    total_protocol_cost = (
        float(getattr(args, "protocol_stale_preview_cost", 1.0)) * stale_preview_calls
        + float(getattr(args, "protocol_current_probe_cost", 1.0)) * current_probe_calls
        + float(getattr(args, "protocol_execution_slot_cost", 1.0))
    )
    row = {
        **context,
        "config_name": config_name,
        "method_name": method_name,
        "policy_name": policy_name,
        "run_index": int(run_index),
        "run_seed": "" if run_seed is None else int(run_seed),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "mismatch_model": mismatch_model,
        "rho": float(channel_rho),
        "csi_delay": int(csi_delay_slots),
        "slot_idx": int(slot_idx),
        "K": int(args.num_nodes),
        "num_slots": int(args.num_slots),
        "num_irs_elements": int(args.num_irs_elements),
        "num_codebook_states": int(args.num_codebook_states),
        "probe_budget": int(budget),
        "chosen_irs_state": int(decision.get("irs_index", -99)),
        "confirmed_irs_state": int(
            decision.get("confirmed_irs_index", decision.get("irs_index", -99))
        ),
        "candidate_irs_states": _json_cell(decision.get("candidate_irs_indices")),
        "stale_predicted_count_per_candidate": _json_cell(
            decision.get("candidate_stale_predicted_counts")
        ),
        "aggregate_feedback_count_per_candidate": _json_cell(
            decision.get("candidate_aggregate_feedback_counts")
        ),
        "aggregate_feedback_score_per_candidate": _json_cell(
            decision.get("candidate_aggregate_feedback_scores")
        ),
        "selected_feedback_count": decision.get("selected_feedback_count", ""),
        "selected_feedback_score": decision.get("selected_feedback_score", ""),
        "posterior_probe_selected_indices": _json_cell(decision.get("posterior_probe_selected_indices")),
        "posterior_probe_objective_value": decision.get("posterior_probe_objective_value", ""),
        "posterior_probe_expected_best_count": decision.get("posterior_probe_expected_best_count", ""),
        "posterior_probe_coverage_score": decision.get("posterior_probe_coverage_score", ""),
        "posterior_probe_expected_count_per_selected_state": _json_cell(
            decision.get("posterior_probe_expected_count_per_selected_state")
        ),
        "posterior_probe_candidate_prefilter_size": decision.get(
            "posterior_probe_candidate_prefilter_size",
            "",
        ),
        "posterior_probe_samples": decision.get("posterior_probe_samples", ""),
        "posterior_probe_beta": decision.get("posterior_probe_beta", ""),
        "posterior_probe_objective": decision.get("posterior_probe_objective", ""),
        "failed_invited_count": int(info["failed_this_slot"]),
        "missed_feasible_count": int(info["missed_opportunity_this_slot"]),
        "remaining_device_count": int(info.get("remaining_count_before", 0)),
        "remaining_device_count_after": int(info.get("remaining_count_after", 0)),
        "tx_this_slot": int(info["tx_this_slot"]),
        "scheduled_this_slot": int(info["scheduled_this_slot"]),
        "true_opportunity_this_slot": int(info["true_opportunity_this_slot"]),
        "power_avg": float(info["power_avg"]),
        "energy": "" if "attempted_energy" not in info else float(info["attempted_energy"]),
        "nmse": "" if "aircomp_nmse" not in info else float(info["aircomp_nmse"]),
        "aircomp_raw_mse": float(info.get("aircomp_raw_mse", 0.0)),
        "aircomp_missing_device_mse": float(info.get("aircomp_missing_device_mse", 0.0)),
        "aircomp_failed_invitation_mse": float(info.get("aircomp_failed_invitation_mse", 0.0)),
        "aircomp_power_clipping_mse": float(info.get("aircomp_power_clipping_mse", 0.0)),
        "aircomp_receiver_noise_mse": float(info.get("aircomp_receiver_noise_mse", 0.0)),
        "aircomp_target_variance": float(info.get("aircomp_target_variance", 0.0)),
        "energy_per_success": float(info.get("energy_per_success", 0.0)),
        "power_clipped_count": int(info.get("power_clipped_count", 0)),
        "power_clipping_rate": float(info.get("power_clipping_rate", 0.0)),
        "pmax_device_count": int(info.get("pmax_device_count", 0)),
        "slot_overhead": int(preview_calls),
        "stale_preview_calls": int(stale_preview_calls),
        "current_probe_calls": int(current_probe_calls),
        "total_probe_calls": int(preview_calls),
        "total_protocol_cost": float(total_protocol_cost),
        "oracle_tx_count": int(execution_oracle.get("tx_this_slot", 0)),
        "oracle_gap": float(oracle_gap),
    }
    row.update(_mask_payload(info.get("invited_mask"), "invited", full_device_lists))
    row.update(_mask_payload(info.get("feasible_mask"), "feasible", full_device_lists))
    row.update(_mask_payload(info.get("success_mask"), "successful_tx", full_device_lists))
    return row


def build_diagnostic_record(
    *,
    args,
    context,
    config_name,
    method_name,
    policy_name,
    run_index,
    run_seed,
    decision_error_std=None,
    execution_error_std=None,
    episode_index,
    episode_seed,
    mismatch_model,
    channel_rho,
    csi_delay_slots,
    budget,
    slot_idx,
    decision,
    execution_oracle,
    posterior_summary=None,
    full_device_lists=False,
):
    """Build optional nested diagnostics as JSONL."""
    record = {
        **context,
        "config_name": config_name,
        "method_name": method_name,
        "policy_name": policy_name,
        "run_index": int(run_index),
        "run_seed": None if run_seed is None else int(run_seed),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "mismatch_model": mismatch_model,
        "rho": float(channel_rho),
        "csi_delay": int(csi_delay_slots),
        "probe_budget": int(budget),
        "slot_idx": int(slot_idx),
        "K": int(args.num_nodes),
        "oracle_irs_state": int(execution_oracle.get("irs_index", -99)),
        "deployable_candidate_set": list(decision.get("deployable_candidate_set", [])),
        "candidate_irs_states": list(decision.get("candidate_irs_indices", [])),
    }
    for source_key, prefix in (
        ("valid_mask", "oracle_invitation"),
        ("stale_invitation_mask", "stale_invitation"),
        ("corrected_invitation_mask", "corrected_invitation"),
        ("posterior_invitation_mask", "posterior_invitation"),
        ("deployable_oracle_invitation_mask", "deployable_oracle_invitation"),
    ):
        mask_source = execution_oracle if source_key == "valid_mask" else decision
        payload = _mask_payload(mask_source.get(source_key), prefix, full_device_lists)
        record.update(payload)
    if posterior_summary is not None:
        record.update(posterior_summary)
        expected_counts = posterior_summary.get(
            "posterior_expected_feasible_count_per_irs_state",
            [],
        )
        chosen_index = int(decision.get("irs_index", -1))
        if 0 <= chosen_index < len(expected_counts):
            record["posterior_chosen_irs_expected_feasible_count"] = float(
                expected_counts[chosen_index]
            )
    for key in (
        "posterior_invitation_prior_summary",
        "posterior_invitation_feedback_count",
        "posterior_invitation_marginal_summary",
        "posterior_invitation_selected_cardinality",
        "posterior_invitation_cardinality_policy",
        "posterior_invitation_stale_overlap_count",
        "posterior_invitation_stale_overlap_fraction",
        "posterior_probe_selected_indices",
        "posterior_probe_objective_value",
        "posterior_probe_expected_best_count",
        "posterior_probe_coverage_score",
        "posterior_probe_expected_count_per_selected_state",
        "posterior_probe_candidate_prefilter_indices",
        "posterior_probe_candidate_prefilter_size",
        "posterior_probe_samples",
        "posterior_probe_beta",
        "posterior_probe_objective",
        "posterior_probe_remaining_device_count",
        "posterior_probe_computed_state_count",
    ):
        if key in decision:
            record[key] = decision[key]
    invited = decision.get("posterior_invitation_mask")
    oracle = execution_oracle.get("valid_mask")
    if invited is not None and oracle is not None:
        invited = [bool(value) for value in invited]
        oracle = [bool(value) for value in oracle]
        overlap_count = sum(1 for lhs, rhs in zip(invited, oracle) if lhs and rhs)
        record["posterior_invitation_oracle_overlap_count"] = int(overlap_count)
        record["posterior_invitation_oracle_overlap_fraction"] = (
            float(overlap_count) / float(max(sum(invited), 1))
        )
    return record


def _write_jsonl(path, rows):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as jsonfile:
        for row in rows:
            jsonfile.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"Saved: {path}")


def _write_csv(path, fields, rows):
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def write_structured_logs(output_dir, run_metadata_rows, scenario_rows, slot_rows, diagnostic_rows):
    """Write publication-grade structured logs beside the legacy outputs."""
    os.makedirs(output_dir, exist_ok=True)
    _write_jsonl(os.path.join(output_dir, "run_metadata.jsonl"), run_metadata_rows)
    _write_csv(os.path.join(output_dir, "scenario_summary.csv"), SCENARIO_SUMMARY_FIELDS, scenario_rows)
    _write_csv(os.path.join(output_dir, "slot_records.csv"), STRUCTURED_SLOT_FIELDS, slot_rows)
    _write_jsonl(os.path.join(output_dir, "diagnostic_records.jsonl"), diagnostic_rows)
