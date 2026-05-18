"""Result aggregation and CSV summaries for execution-mismatch runs."""

import csv

import numpy as np

from ms_aircomp.experiment_utils import ensure_parent_dir

__all__ = [
    "CSV_FIELDS",
    "NUMERIC_RESULT_KEYS",
    "OPTIONAL_RESULT_DEFAULTS",
    "aggregate_seed_results",
    "metric_mean_ci",
    "result_array",
    "result_metadata",
    "result_value",
    "seed_summary",
    "summarize_results",
    "write_csv",
]


NUMERIC_RESULT_KEYS = (
    "success_nodes",
    "avg_power",
    "episode_reward",
    "slots_used",
    "total_energy",
    "scheduled_nodes",
    "failed_nodes",
    "missed_opportunities",
    "true_opportunities",
    "failure_slot_fraction",
    "decision_preview_calls_per_slot",
    "oracle_tx_gap_mean",
    "effective_risk_weight",
    "adaptive_sparse_expanded",
    "adaptive_sparse_margin",
    "adaptive_sparse_effective_margin_threshold",
    "adaptive_sparse_expand_score",
    "adaptive_sparse_history_stability",
    "adaptive_sparse_urgency",
    "adaptive_sparse_cost_penalty",
    "adaptive_sparse_v3_history_prior_used",
    "adaptive_sparse_v3_neighbor_extra_preview_count",
    "adaptive_sparse_v3_selected_extra_preview_count",
    "learned_shortlist_selected_extra_preview_count",
    "coverage_sparse_selected_marginal_fraction",
    "coverage_sparse_selected_overlap_fraction",
)

OPTIONAL_RESULT_DEFAULTS = {
    "gain_margin": 1.0,
    "power_margin": 1.0,
    "risk_weight": 0.0,
    "risk_power_weight": 0.0,
    "risk_invite_threshold": 0.0,
    "adaptive_risk_base_weight": 0.0,
    "opportunity_failure_cost": 0.0,
    "opportunity_missed_cost": 0.0,
    "opportunity_deadline_gain": 0.0,
    "opportunity_backlog_gain": 0.0,
    "temporal_reliability_z": 0.0,
    "sparse_topk_seed_multiplier": 2.0,
    "sparse_topk_fraction": 0.75,
    "coverage_sparse_weight": 0.5,
    "coverage_sparse_power_weight": 0.0,
    "adaptive_sparse_base_multiplier": 2.0,
    "adaptive_sparse_expanded_multiplier": 3.0,
    "adaptive_sparse_margin_threshold": 0.05,
    "adaptive_sparse_v2_preview_cost": 0.002,
    "adaptive_sparse_v2_uncertainty_weight": 0.02,
    "adaptive_sparse_v2_urgency_weight": 0.5,
    "adaptive_sparse_v2_history_weight": 0.02,
    "adaptive_sparse_history_window": 3,
    "adaptive_sparse_v3_neighbor_radius": 1,
    "adaptive_sparse_v3_neighbor_count": 2,
    "adaptive_sparse_v3_history_count": 1,
    "learned_shortlist_extra_count": 1,
}

METADATA_FIELDS = (
    "result_role",
    "uses_hidden_training_labels",
    "inference_uses_hidden_current_csi",
    "supervision_signal",
)

DEFAULT_METADATA = {
    "result_role": "comparison_reference",
    "uses_hidden_training_labels": "false",
    "inference_uses_hidden_current_csi": "false",
    "supervision_signal": "none",
}

LEARNED_HIDDEN_LABEL_MARKERS = (
    "Learned",
    "DAgger",
    "Window Temporal Deviation",
    "Gated Temporal Deviation",
)

HIDDEN_INFERENCE_MARKERS = (
    "Execution Oracle",
    "Temporal Deviation Oracle",
)

CSV_FIELDS = [
    "decision_error_std",
    "execution_error_std",
    "mismatch_model",
    "channel_rho",
    "csi_delay_slots",
    "probe_budget",
    "policy",
    "result_role",
    "uses_hidden_training_labels",
    "inference_uses_hidden_current_csi",
    "supervision_signal",
    "episodes",
    "num_seeds",
    "num_nodes",
    "num_slots",
    "num_irs_elements",
    "num_codebook_states",
    "g_th",
    "alpha_th",
    "gain_margin",
    "power_margin",
    "risk_weight",
    "risk_power_weight",
    "risk_invite_threshold",
    "adaptive_risk_base_weight",
    "opportunity_failure_cost",
    "opportunity_missed_cost",
    "opportunity_deadline_gain",
    "opportunity_backlog_gain",
    "temporal_reliability_z",
    "sparse_topk_seed_multiplier",
    "sparse_topk_fraction",
    "coverage_sparse_weight",
    "coverage_sparse_power_weight",
    "adaptive_sparse_base_multiplier",
    "adaptive_sparse_expanded_multiplier",
    "adaptive_sparse_margin_threshold",
    "adaptive_sparse_v2_preview_cost",
    "adaptive_sparse_v2_uncertainty_weight",
    "adaptive_sparse_v2_urgency_weight",
    "adaptive_sparse_v2_history_weight",
    "adaptive_sparse_history_window",
    "adaptive_sparse_v3_neighbor_radius",
    "adaptive_sparse_v3_neighbor_count",
    "adaptive_sparse_v3_history_count",
    "learned_shortlist_extra_count",
    "success_mean",
    "success_ci95",
    "success_rate_mean",
    "perfect_rate",
    "slots_mean",
    "slots_ci95",
    "avg_power",
    "total_energy_mean",
    "total_energy_ci95",
    "scheduled_nodes_mean",
    "failed_nodes_mean",
    "missed_opportunities_mean",
    "true_opportunities_mean",
    "failure_slot_rate",
    "decision_preview_calls_per_slot_mean",
    "oracle_tx_gap_mean",
    "effective_risk_weight_mean",
    "adaptive_sparse_expansion_rate",
    "adaptive_sparse_margin_mean",
    "adaptive_sparse_effective_margin_threshold_mean",
    "adaptive_sparse_expand_score_mean",
    "adaptive_sparse_history_stability_mean",
    "adaptive_sparse_urgency_mean",
    "adaptive_sparse_cost_penalty_mean",
    "adaptive_sparse_v3_history_prior_rate",
    "adaptive_sparse_v3_neighbor_extra_preview_mean",
    "adaptive_sparse_v3_selected_extra_preview_mean",
    "learned_shortlist_selected_extra_preview_mean",
    "coverage_sparse_selected_marginal_fraction_mean",
    "coverage_sparse_selected_overlap_fraction_mean",
    "avg_reward",
]


def seed_summary(result):
    """Compress one run seed result into seed-level means."""
    return {key: float(np.mean(result_array(result, key))) for key in NUMERIC_RESULT_KEYS}


def result_array(result, key):
    """Return a result metric array, defaulting optional diagnostics to zeros."""
    if key in result:
        return np.asarray(result[key], dtype=float)
    if key == "failure_slot_fraction" and "failure_slots" in result:
        return np.asarray(result["failure_slots"], dtype=float)
    episode_count = len(result["success_nodes"])
    return np.zeros(episode_count, dtype=float)


def result_value(result, key):
    """Return a scalar metadata field with defaults for nonapplicable diagnostics."""
    return result.get(key, OPTIONAL_RESULT_DEFAULTS[key])


def result_metadata(result):
    """Return paper-boundary metadata for one execution result."""
    name = str(result.get("policy", result.get("name", "")))
    metadata = dict(DEFAULT_METADATA)
    if any(marker in name for marker in HIDDEN_INFERENCE_MARKERS):
        metadata.update(
            {
                "result_role": "diagnostic_upper_bound",
                "inference_uses_hidden_current_csi": "true",
                "supervision_signal": "hidden_current_channel_oracle_at_evaluation",
            }
        )
    elif any(marker in name for marker in LEARNED_HIDDEN_LABEL_MARKERS):
        metadata.update(
            {
                "result_role": "diagnostic",
                "uses_hidden_training_labels": "true",
                "supervision_signal": "hidden_current_channel_supervised_targets",
            }
        )
    for key in METADATA_FIELDS:
        if key in result:
            metadata[key] = str(result[key])
    return metadata


def aggregate_seed_results(seed_result_sets):
    """Aggregate matching result lists across run seeds."""
    if not seed_result_sets:
        return []
    aggregated_results = []
    result_count = len(seed_result_sets[0])
    for result_idx in range(result_count):
        parts = [seed_results[result_idx] for seed_results in seed_result_sets]
        aggregated = {
            "name": parts[0]["name"],
            "decision_error_std": parts[0]["decision_error_std"],
            "execution_error_std": parts[0]["execution_error_std"],
            "mismatch_model": parts[0]["mismatch_model"],
            "channel_rho": parts[0]["channel_rho"],
            "csi_delay_slots": parts[0]["csi_delay_slots"],
            "probe_budget": parts[0]["probe_budget"],
            "gain_margin": result_value(parts[0], "gain_margin"),
            "power_margin": result_value(parts[0], "power_margin"),
            "risk_weight": result_value(parts[0], "risk_weight"),
            "risk_power_weight": result_value(parts[0], "risk_power_weight"),
            "risk_invite_threshold": result_value(parts[0], "risk_invite_threshold"),
            "adaptive_risk_base_weight": result_value(parts[0], "adaptive_risk_base_weight"),
            "opportunity_failure_cost": result_value(parts[0], "opportunity_failure_cost"),
            "opportunity_missed_cost": result_value(parts[0], "opportunity_missed_cost"),
            "opportunity_deadline_gain": result_value(parts[0], "opportunity_deadline_gain"),
            "opportunity_backlog_gain": result_value(parts[0], "opportunity_backlog_gain"),
            "temporal_reliability_z": result_value(parts[0], "temporal_reliability_z"),
            "sparse_topk_seed_multiplier": result_value(parts[0], "sparse_topk_seed_multiplier"),
            "sparse_topk_fraction": result_value(parts[0], "sparse_topk_fraction"),
            "coverage_sparse_weight": result_value(parts[0], "coverage_sparse_weight"),
            "coverage_sparse_power_weight": result_value(parts[0], "coverage_sparse_power_weight"),
            "adaptive_sparse_base_multiplier": result_value(parts[0], "adaptive_sparse_base_multiplier"),
            "adaptive_sparse_expanded_multiplier": result_value(parts[0], "adaptive_sparse_expanded_multiplier"),
            "adaptive_sparse_margin_threshold": result_value(parts[0], "adaptive_sparse_margin_threshold"),
            "adaptive_sparse_v2_preview_cost": result_value(parts[0], "adaptive_sparse_v2_preview_cost"),
            "adaptive_sparse_v2_uncertainty_weight": result_value(
                parts[0],
                "adaptive_sparse_v2_uncertainty_weight",
            ),
            "adaptive_sparse_v2_urgency_weight": result_value(parts[0], "adaptive_sparse_v2_urgency_weight"),
            "adaptive_sparse_v2_history_weight": result_value(parts[0], "adaptive_sparse_v2_history_weight"),
            "adaptive_sparse_history_window": result_value(parts[0], "adaptive_sparse_history_window"),
            "adaptive_sparse_v3_neighbor_radius": result_value(parts[0], "adaptive_sparse_v3_neighbor_radius"),
            "adaptive_sparse_v3_neighbor_count": result_value(parts[0], "adaptive_sparse_v3_neighbor_count"),
            "adaptive_sparse_v3_history_count": result_value(parts[0], "adaptive_sparse_v3_history_count"),
            "learned_shortlist_extra_count": result_value(parts[0], "learned_shortlist_extra_count"),
        }
        aggregated.update(result_metadata(parts[0]))
        for key in NUMERIC_RESULT_KEYS:
            aggregated[key] = np.concatenate([result_array(part, key) for part in parts])
        aggregated["seed_summaries"] = [seed_summary(part) for part in parts]
        aggregated_results.append(aggregated)
    return aggregated_results


def metric_mean_ci(result, key):
    """Compute overall mean and run-seed 95 percent CI."""
    seed_values = np.asarray(
        [summary[key] for summary in result.get("seed_summaries", [seed_summary(result)])],
        dtype=float,
    )
    mean_value = float(np.mean(result[key]))
    if len(seed_values) <= 1:
        return mean_value, 0.0
    ci95 = 1.96 * float(np.std(seed_values, ddof=1)) / np.sqrt(len(seed_values))
    return mean_value, ci95


def summarize_results(args, results):
    """Convert aggregated results into CSV rows."""
    rows = []
    for result in results:
        success_mean, success_ci95 = metric_mean_ci(result, "success_nodes")
        slots_mean, slots_ci95 = metric_mean_ci(result, "slots_used")
        energy_mean, energy_ci95 = metric_mean_ci(result, "total_energy")
        rows.append(
            {
                "decision_error_std": float(result["decision_error_std"]),
                "execution_error_std": float(result["execution_error_std"]),
                "mismatch_model": result["mismatch_model"],
                "channel_rho": float(result["channel_rho"]),
                "csi_delay_slots": int(result["csi_delay_slots"]),
                "probe_budget": int(result["probe_budget"]),
                "policy": result["name"],
                **result_metadata(result),
                "episodes": len(result["success_nodes"]),
                "num_seeds": args.num_seeds,
                "num_nodes": args.num_nodes,
                "num_slots": args.num_slots,
                "num_irs_elements": args.num_irs_elements,
                "num_codebook_states": args.num_codebook_states,
                "g_th": args.g_th,
                "alpha_th": args.alpha_th,
                "gain_margin": float(result["gain_margin"]),
                "power_margin": float(result["power_margin"]),
                "risk_weight": float(result["risk_weight"]),
                "risk_power_weight": float(result["risk_power_weight"]),
                "risk_invite_threshold": float(result["risk_invite_threshold"]),
                "adaptive_risk_base_weight": float(result["adaptive_risk_base_weight"]),
                "opportunity_failure_cost": float(result["opportunity_failure_cost"]),
                "opportunity_missed_cost": float(result["opportunity_missed_cost"]),
                "opportunity_deadline_gain": float(result["opportunity_deadline_gain"]),
                "opportunity_backlog_gain": float(result["opportunity_backlog_gain"]),
                "temporal_reliability_z": float(result["temporal_reliability_z"]),
                "sparse_topk_seed_multiplier": float(result["sparse_topk_seed_multiplier"]),
                "sparse_topk_fraction": float(result["sparse_topk_fraction"]),
                "coverage_sparse_weight": float(result["coverage_sparse_weight"]),
                "coverage_sparse_power_weight": float(result["coverage_sparse_power_weight"]),
                "adaptive_sparse_base_multiplier": float(result["adaptive_sparse_base_multiplier"]),
                "adaptive_sparse_expanded_multiplier": float(result["adaptive_sparse_expanded_multiplier"]),
                "adaptive_sparse_margin_threshold": float(result["adaptive_sparse_margin_threshold"]),
                "adaptive_sparse_v2_preview_cost": float(result["adaptive_sparse_v2_preview_cost"]),
                "adaptive_sparse_v2_uncertainty_weight": float(
                    result["adaptive_sparse_v2_uncertainty_weight"]
                ),
                "adaptive_sparse_v2_urgency_weight": float(result["adaptive_sparse_v2_urgency_weight"]),
                "adaptive_sparse_v2_history_weight": float(result["adaptive_sparse_v2_history_weight"]),
                "adaptive_sparse_history_window": int(result["adaptive_sparse_history_window"]),
                "adaptive_sparse_v3_neighbor_radius": int(result["adaptive_sparse_v3_neighbor_radius"]),
                "adaptive_sparse_v3_neighbor_count": int(result["adaptive_sparse_v3_neighbor_count"]),
                "adaptive_sparse_v3_history_count": int(result["adaptive_sparse_v3_history_count"]),
                "learned_shortlist_extra_count": int(result["learned_shortlist_extra_count"]),
                "success_mean": success_mean,
                "success_ci95": success_ci95,
                "success_rate_mean": success_mean / args.num_nodes,
                "perfect_rate": float(np.mean(result["success_nodes"] == args.num_nodes) * 100.0),
                "slots_mean": slots_mean,
                "slots_ci95": slots_ci95,
                "avg_power": float(np.mean(result["avg_power"])),
                "total_energy_mean": energy_mean,
                "total_energy_ci95": energy_ci95,
                "scheduled_nodes_mean": float(np.mean(result["scheduled_nodes"])),
                "failed_nodes_mean": float(np.mean(result["failed_nodes"])),
                "missed_opportunities_mean": float(np.mean(result["missed_opportunities"])),
                "true_opportunities_mean": float(np.mean(result["true_opportunities"])),
                "failure_slot_rate": float(np.mean(result["failure_slot_fraction"]) * 100.0),
                "decision_preview_calls_per_slot_mean": float(
                    np.mean(result["decision_preview_calls_per_slot"])
                ),
                "oracle_tx_gap_mean": float(np.mean(result["oracle_tx_gap_mean"])),
                "effective_risk_weight_mean": float(np.mean(result["effective_risk_weight"])),
                "adaptive_sparse_expansion_rate": float(np.mean(result["adaptive_sparse_expanded"])),
                "adaptive_sparse_margin_mean": float(np.mean(result["adaptive_sparse_margin"])),
                "adaptive_sparse_effective_margin_threshold_mean": float(
                    np.mean(result["adaptive_sparse_effective_margin_threshold"])
                ),
                "adaptive_sparse_expand_score_mean": float(
                    np.mean(result["adaptive_sparse_expand_score"])
                ),
                "adaptive_sparse_history_stability_mean": float(
                    np.mean(result["adaptive_sparse_history_stability"])
                ),
                "adaptive_sparse_urgency_mean": float(np.mean(result["adaptive_sparse_urgency"])),
                "adaptive_sparse_cost_penalty_mean": float(
                    np.mean(result["adaptive_sparse_cost_penalty"])
                ),
                "adaptive_sparse_v3_history_prior_rate": float(
                    np.mean(result["adaptive_sparse_v3_history_prior_used"])
                ),
                "adaptive_sparse_v3_neighbor_extra_preview_mean": float(
                    np.mean(result["adaptive_sparse_v3_neighbor_extra_preview_count"])
                ),
                "adaptive_sparse_v3_selected_extra_preview_mean": float(
                    np.mean(result["adaptive_sparse_v3_selected_extra_preview_count"])
                ),
                "learned_shortlist_selected_extra_preview_mean": float(
                    np.mean(result["learned_shortlist_selected_extra_preview_count"])
                ),
                "coverage_sparse_selected_marginal_fraction_mean": float(
                    np.mean(result["coverage_sparse_selected_marginal_fraction"])
                ),
                "coverage_sparse_selected_overlap_fraction_mean": float(
                    np.mean(result["coverage_sparse_selected_overlap_fraction"])
                ),
                "avg_reward": float(np.mean(result["episode_reward"])),
            }
        )
    return rows


def write_csv(path, rows):
    """Write execution mismatch summary CSV."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")
