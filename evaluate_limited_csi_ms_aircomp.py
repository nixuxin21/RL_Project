"""
Limited-CSI IRS-assisted multi-slot AirComp evaluation.

This script separates what a policy can infer from limited/noisy CSI from what
actually succeeds on the true channel. A policy first probes a subset of IRS
codebook indices, builds an estimated schedulable-node mask, chooses an IRS
index, and invites only those estimated-valid nodes. Execution is then verified
against the true channel; nodes that were not invited are not auto-scheduled.

The goal is to evaluate whether dynamic IRS selection still helps multi-slot
AirComp when full per-codebook CSI/features are unavailable.
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_policy_comparison import (
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
)
from ms_aircomp.limited_csi import (
    POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
    POLICY_EST_GREEDY,
    POLICY_EXACT_GREEDY,
    POLICY_FIXED_IRS,
    POLICY_NO_IRS,
    POLICY_OFFSETS,
    POLICY_RANDOM_PROBE,
    POLICY_RISK_AWARE_ROTATING_GRID,
    POLICY_ROBUST_ROTATING_GRID,
    POLICY_ROTATING_GRID,
    adaptive_risk_weight,
    best_candidate,
    best_risk_aware_candidate,
    build_candidate,
    candidate_key,
    choose_policy_candidate,
    effective_channels,
    effective_risk_invite_threshold,
    estimate_success_reliability,
    estimated_preview_candidates,
    execute_limited_csi_slot,
    grid_indices,
    oracle_candidate,
    parse_float_list,
    parse_int_list,
    risk_aware_candidate,
    risk_aware_candidate_key,
    select_indices,
    stable_rng,
    success_gain_threshold,
    true_candidate_for_decision,
    true_preview_candidates,
    unique_fill,
)
from test_env import MSAirCompEnv


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
    "failure_slot_count",
    "decision_preview_calls_per_slot",
    "oracle_tx_gap_mean",
    "effective_risk_weight",
)


def parse_args():
    """Parse limited-CSI evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate IRS-assisted MS-AirComp under limited/noisy CSI."
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--probe-budgets", default="2,4,8")
    parser.add_argument("--error-std-values", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--robust-gain-margins", default="1.25")
    parser.add_argument("--robust-power-margins", default="0.9")
    parser.add_argument("--risk-weights", default="0.5")
    parser.add_argument("--risk-power-weights", default="0.1")
    parser.add_argument("--risk-invite-thresholds", default="0.5")
    parser.add_argument("--adaptive-risk-base-weights", default="0.5")
    parser.add_argument("--adaptive-risk-error-ref", type=float, default=0.3)
    parser.add_argument("--adaptive-risk-error-gain", type=float, default=1.0)
    parser.add_argument("--adaptive-risk-deadline-relief", type=float, default=0.6)
    parser.add_argument("--adaptive-risk-backlog-relief", type=float, default=0.8)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--fixed-irs-index", type=int, default=7)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """Validate sizes, budgets, error levels, and robust margins."""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    for name in ("num_nodes", "num_slots", "num_irs_elements", "num_codebook_states"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")
    if args.g_th <= 0:
        raise ValueError("--g-th must be positive")
    if args.alpha_th <= 0:
        raise ValueError("--alpha-th must be positive")

    budgets = parse_int_list(args.probe_budgets)
    if not budgets:
        raise ValueError("--probe-budgets must contain at least one value")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("--probe-budgets must contain positive integers")
    args.probe_budgets = sorted({min(int(budget), args.num_codebook_states) for budget in budgets})

    args.error_std_values = parse_float_list(args.error_std_values)
    if not args.error_std_values:
        raise ValueError("--error-std-values must contain at least one value")
    if any(value < 0.0 for value in args.error_std_values):
        raise ValueError("--error-std-values must be non-negative")

    args.robust_gain_margins = parse_float_list(args.robust_gain_margins)
    args.robust_power_margins = parse_float_list(args.robust_power_margins)
    if not args.robust_gain_margins or not args.robust_power_margins:
        raise ValueError("robust margin lists must not be empty")
    if any(value <= 0.0 for value in args.robust_gain_margins):
        raise ValueError("--robust-gain-margins must be positive")
    if any(value <= 0.0 for value in args.robust_power_margins):
        raise ValueError("--robust-power-margins must be positive")

    args.risk_weights = parse_float_list(args.risk_weights)
    args.risk_power_weights = parse_float_list(args.risk_power_weights)
    args.risk_invite_thresholds = parse_float_list(args.risk_invite_thresholds)
    if not args.risk_weights or not args.risk_power_weights or not args.risk_invite_thresholds:
        raise ValueError("risk-aware parameter lists must not be empty")
    if any(value < 0.0 for value in args.risk_weights):
        raise ValueError("--risk-weights must be non-negative")
    if any(value < 0.0 for value in args.risk_power_weights):
        raise ValueError("--risk-power-weights must be non-negative")
    if any(value < 0.0 or value > 1.0 for value in args.risk_invite_thresholds):
        raise ValueError("--risk-invite-thresholds must be in [0, 1]")

    args.adaptive_risk_base_weights = parse_float_list(args.adaptive_risk_base_weights)
    if not args.adaptive_risk_base_weights:
        raise ValueError("--adaptive-risk-base-weights must contain at least one value")
    if any(value < 0.0 for value in args.adaptive_risk_base_weights):
        raise ValueError("--adaptive-risk-base-weights must be non-negative")
    if args.adaptive_risk_error_ref <= 0.0:
        raise ValueError("--adaptive-risk-error-ref must be positive")
    if args.adaptive_risk_error_gain < 0.0:
        raise ValueError("--adaptive-risk-error-gain must be non-negative")
    if args.adaptive_risk_deadline_relief < 0.0:
        raise ValueError("--adaptive-risk-deadline-relief must be non-negative")
    if args.adaptive_risk_backlog_relief < 0.0:
        raise ValueError("--adaptive-risk-backlog-relief must be non-negative")

    args.fixed_irs_index = int(np.clip(args.fixed_irs_index, 0, args.num_codebook_states - 1))


def resolve_output_prefix(args):
    """Resolve output prefix for CSVs and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    budget_label = "-".join(format_float_for_suffix(value) for value in args.probe_budgets)
    error_label = "-".join(format_float_for_suffix(value) for value in args.error_std_values)
    gain_label = "-".join(format_float_for_suffix(value) for value in args.robust_gain_margins)
    power_label = "-".join(format_float_for_suffix(value) for value in args.robust_power_margins)
    risk_label = "-".join(format_float_for_suffix(value) for value in args.risk_weights)
    risk_power_label = "-".join(format_float_for_suffix(value) for value in args.risk_power_weights)
    risk_threshold_label = "-".join(format_float_for_suffix(value) for value in args.risk_invite_thresholds)
    adaptive_risk_label = "-".join(
        format_float_for_suffix(value) for value in args.adaptive_risk_base_weights
    )
    suffix = (
        f"ep{args.episodes}_runs{args.num_seeds}_{seed_label}_b{budget_label}_"
        f"err{error_label}_gm{gain_label}_pm{power_label}_"
        f"rw{risk_label}_rp{risk_power_label}_rt{risk_threshold_label}_"
        f"arw{adaptive_risk_label}"
    )
    output_prefix = os.path.join("results", "limited_csi", f"limited_csi_ms_aircomp_{suffix}")
    ensure_parent_dir(output_prefix)
    return output_prefix


def make_env(args):
    """Create the base codebook environment used for limited-CSI evaluation."""
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode="codebook",
    )


def evaluate_policy(
    episode_seeds,
    args,
    error_std,
    policy_name,
    budget,
    gain_margin=1.0,
    power_margin=1.0,
    risk_weight=0.0,
    risk_power_weight=0.0,
    risk_invite_threshold=0.0,
    adaptive_risk_error_ref=0.3,
    adaptive_risk_error_gain=1.0,
    adaptive_risk_deadline_relief=0.6,
    adaptive_risk_backlog_relief=0.8,
):
    """Evaluate one limited-CSI policy for one error level and budget."""
    env = make_env(args)
    success_nodes = []
    avg_power = []
    rewards = []
    slots_used = []
    total_energy = []
    scheduled_nodes = []
    failed_nodes = []
    missed_opportunities = []
    true_opportunities = []
    failure_slot_counts = []
    preview_calls_per_slot = []
    oracle_tx_gap_mean = []
    effective_risk_weights = []

    print(
        f"Running {policy_name} err={error_std:g} B={budget} "
        f"gm={gain_margin:g} pm={power_margin:g} "
        f"rw={risk_weight:g} rp={risk_power_weight:g} rt={risk_invite_threshold:g}..."
    )
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        episode_power = []
        episode_reward = 0.0
        episode_energy = 0.0
        episode_scheduled = 0
        episode_failed = 0
        episode_missed = 0
        episode_true_opportunities = 0
        episode_failure_slots = 0
        episode_preview_calls = []
        episode_oracle_gaps = []
        episode_effective_risk_weights = []
        total_tx = 0
        episode_slots = args.num_slots

        for slot_idx in range(args.num_slots):
            oracle = oracle_candidate(env, args)
            decision, preview_calls, _candidate_count = choose_policy_candidate(
                env,
                args,
                policy_name,
                budget,
                slot_idx,
                error_std,
                episode_seed,
                gain_margin=gain_margin,
                power_margin=power_margin,
                risk_weight=risk_weight,
                risk_power_weight=risk_power_weight,
                risk_invite_threshold=risk_invite_threshold,
                adaptive_risk_error_ref=adaptive_risk_error_ref,
                adaptive_risk_error_gain=adaptive_risk_error_gain,
                adaptive_risk_deadline_relief=adaptive_risk_deadline_relief,
                adaptive_risk_backlog_relief=adaptive_risk_backlog_relief,
            )
            true_selected = true_candidate_for_decision(env, args, decision)
            info, done = execute_limited_csi_slot(env, args, decision, true_selected)

            total_tx = int(info["total_tx"])
            episode_slots = int(info["slots_used"])
            episode_reward += float(info["reward"])
            episode_energy += float(info["attempted_energy"])
            episode_scheduled += int(info["scheduled_this_slot"])
            episode_failed += int(info["failed_this_slot"])
            episode_missed += int(info["missed_opportunity_this_slot"])
            episode_true_opportunities += int(info["true_opportunity_this_slot"])
            episode_failure_slots += int(info["failed_this_slot"] > 0)
            episode_preview_calls.append(int(preview_calls))
            episode_effective_risk_weights.append(float(decision.get("effective_risk_weight", 0.0)))
            episode_oracle_gaps.append(
                max(0.0, float(oracle["tx_this_slot"]) - float(true_selected["tx_this_slot"]))
            )
            if info["scheduled_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            if done:
                break

        success_nodes.append(total_tx)
        avg_power.append(float(np.mean(episode_power)) if episode_power else 0.0)
        rewards.append(float(episode_reward))
        slots_used.append(int(episode_slots))
        total_energy.append(float(episode_energy))
        scheduled_nodes.append(float(episode_scheduled))
        failed_nodes.append(float(episode_failed))
        missed_opportunities.append(float(episode_missed))
        true_opportunities.append(float(episode_true_opportunities))
        failure_slot_counts.append(float(episode_failure_slots))
        active_slots = max(len(episode_preview_calls), 1)
        preview_calls_per_slot.append(float(sum(episode_preview_calls)) / active_slots)
        oracle_tx_gap_mean.append(float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0)
        effective_risk_weights.append(
            float(np.mean(episode_effective_risk_weights)) if episode_effective_risk_weights else 0.0
        )

        print_progress(policy_name, error_std, budget, ep, args.episodes, success_nodes, args.num_nodes)

    return {
        "name": policy_name,
        "error_std": float(error_std),
        "probe_budget": int(budget),
        "gain_margin": float(gain_margin),
        "power_margin": float(power_margin),
        "risk_weight": float(risk_weight),
        "risk_power_weight": float(risk_power_weight),
        "risk_invite_threshold": float(risk_invite_threshold),
        "success_nodes": np.asarray(success_nodes, dtype=float),
        "avg_power": np.asarray(avg_power, dtype=float),
        "episode_reward": np.asarray(rewards, dtype=float),
        "slots_used": np.asarray(slots_used, dtype=float),
        "total_energy": np.asarray(total_energy, dtype=float),
        "scheduled_nodes": np.asarray(scheduled_nodes, dtype=float),
        "failed_nodes": np.asarray(failed_nodes, dtype=float),
        "missed_opportunities": np.asarray(missed_opportunities, dtype=float),
        "true_opportunities": np.asarray(true_opportunities, dtype=float),
        "failure_slot_count": np.asarray(failure_slot_counts, dtype=float),
        "decision_preview_calls_per_slot": np.asarray(preview_calls_per_slot, dtype=float),
        "oracle_tx_gap_mean": np.asarray(oracle_tx_gap_mean, dtype=float),
        "effective_risk_weight": np.asarray(effective_risk_weights, dtype=float),
    }


def print_progress(name, error_std, budget, ep, episodes, success_nodes, num_nodes):
    """Print progress at 10 percent intervals."""
    interval = max(episodes // 10, 1)
    if ep % interval == 0 or ep == episodes:
        recent = np.mean(success_nodes[-interval:])
        print(
            f"  {name} err={error_std:g} B={budget}: "
            f"[{ep:04d}/{episodes:04d}] recent success {recent:.2f}/{num_nodes}"
        )


def policy_suite_for_error(args, episode_seeds, error_std):
    """Run all policy/budget combinations for one error level and run seed."""
    results = [
        evaluate_policy(
            episode_seeds,
            args,
            error_std,
            POLICY_NO_IRS,
            budget=0,
        ),
        evaluate_policy(
            episode_seeds,
            args,
            error_std,
            POLICY_FIXED_IRS,
            budget=1,
        ),
        evaluate_policy(
            episode_seeds,
            args,
            error_std,
            POLICY_EXACT_GREEDY,
            budget=args.num_codebook_states,
        ),
        evaluate_policy(
            episode_seeds,
            args,
            error_std,
            POLICY_EST_GREEDY,
            budget=args.num_codebook_states,
        ),
    ]

    for budget in args.probe_budgets:
        results.append(evaluate_policy(episode_seeds, args, error_std, POLICY_RANDOM_PROBE, budget=budget))
        results.append(evaluate_policy(episode_seeds, args, error_std, POLICY_ROTATING_GRID, budget=budget))
        for gain_margin in args.robust_gain_margins:
            for power_margin in args.robust_power_margins:
                results.append(
                    evaluate_policy(
                        episode_seeds,
                        args,
                        error_std,
                        POLICY_ROBUST_ROTATING_GRID,
                        budget=budget,
                        gain_margin=gain_margin,
                        power_margin=power_margin,
                    )
                )
        for risk_weight in args.risk_weights:
            for risk_power_weight in args.risk_power_weights:
                for risk_invite_threshold in args.risk_invite_thresholds:
                    results.append(
                        evaluate_policy(
                            episode_seeds,
                            args,
                            error_std,
                            POLICY_RISK_AWARE_ROTATING_GRID,
                            budget=budget,
                            risk_weight=risk_weight,
                            risk_power_weight=risk_power_weight,
                            risk_invite_threshold=risk_invite_threshold,
                        )
                    )
        for risk_weight in args.adaptive_risk_base_weights:
            for risk_power_weight in args.risk_power_weights:
                for risk_invite_threshold in args.risk_invite_thresholds:
                    results.append(
                        evaluate_policy(
                            episode_seeds,
                            args,
                            error_std,
                            POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
                            budget=budget,
                            risk_weight=risk_weight,
                            risk_power_weight=risk_power_weight,
                            risk_invite_threshold=risk_invite_threshold,
                            adaptive_risk_error_ref=args.adaptive_risk_error_ref,
                            adaptive_risk_error_gain=args.adaptive_risk_error_gain,
                            adaptive_risk_deadline_relief=args.adaptive_risk_deadline_relief,
                            adaptive_risk_backlog_relief=args.adaptive_risk_backlog_relief,
                        )
                    )
    return results


def seed_summary(result):
    """Compress one run seed result to seed-level means."""
    return {key: float(np.mean(result[key])) for key in NUMERIC_RESULT_KEYS}


def aggregate_seed_results(seed_result_sets):
    """Aggregate result lists across run seeds."""
    if not seed_result_sets:
        return []

    aggregated_results = []
    result_count = len(seed_result_sets[0])
    for result_idx in range(result_count):
        parts = [seed_results[result_idx] for seed_results in seed_result_sets]
        aggregated = {
            "name": parts[0]["name"],
            "error_std": parts[0]["error_std"],
            "probe_budget": parts[0]["probe_budget"],
            "gain_margin": parts[0]["gain_margin"],
            "power_margin": parts[0]["power_margin"],
            "risk_weight": parts[0]["risk_weight"],
            "risk_power_weight": parts[0]["risk_power_weight"],
            "risk_invite_threshold": parts[0]["risk_invite_threshold"],
        }
        for key in NUMERIC_RESULT_KEYS:
            aggregated[key] = np.concatenate([part[key] for part in parts])
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


def safe_rate(numerator, denominator):
    """Return a percentage rate with a zero-denominator guard."""
    total_denominator = float(np.sum(denominator))
    if total_denominator <= 0.0:
        return 0.0
    return float(np.sum(numerator) / total_denominator * 100.0)


def summarize_results(args, results):
    """Convert aggregated results to CSV summary rows."""
    rows = []
    for result in results:
        success_mean, success_ci95 = metric_mean_ci(result, "success_nodes")
        slots_mean, slots_ci95 = metric_mean_ci(result, "slots_used")
        energy_mean, energy_ci95 = metric_mean_ci(result, "total_energy")
        scheduled_mean = float(np.mean(result["scheduled_nodes"]))
        failed_mean = float(np.mean(result["failed_nodes"]))
        true_opp_mean = float(np.mean(result["true_opportunities"]))
        missed_opp_mean = float(np.mean(result["missed_opportunities"]))
        rows.append(
            {
                "error_std": float(result["error_std"]),
                "probe_budget": int(result["probe_budget"]),
                "policy": result["name"],
                "gain_margin": float(result["gain_margin"]),
                "power_margin": float(result["power_margin"]),
                "risk_weight": float(result["risk_weight"]),
                "risk_power_weight": float(result["risk_power_weight"]),
                "risk_invite_threshold": float(result["risk_invite_threshold"]),
                "episodes": len(result["success_nodes"]),
                "num_seeds": args.num_seeds,
                "success_mean": success_mean,
                "success_ci95": success_ci95,
                "success_rate_mean": success_mean / args.num_nodes,
                "perfect_rate": float(np.mean(result["success_nodes"] == args.num_nodes) * 100.0),
                "slots_mean": slots_mean,
                "slots_ci95": slots_ci95,
                "avg_power": float(np.mean(result["avg_power"])),
                "total_energy_mean": energy_mean,
                "total_energy_ci95": energy_ci95,
                "scheduled_nodes_mean": scheduled_mean,
                "failed_nodes_mean": failed_mean,
                "true_opportunities_mean": true_opp_mean,
                "missed_opportunities_mean": missed_opp_mean,
                "false_positive_rate": safe_rate(result["failed_nodes"], result["scheduled_nodes"]),
                "execution_failure_rate": safe_rate(result["failed_nodes"], result["scheduled_nodes"]),
                "failure_slot_count_mean": float(np.mean(result["failure_slot_count"])),
                "failure_slot_rate": safe_rate(result["failure_slot_count"], result["slots_used"]),
                "missed_opportunity_rate": safe_rate(
                    result["missed_opportunities"],
                    result["true_opportunities"],
                ),
                "decision_preview_calls_per_slot_mean": float(
                    np.mean(result["decision_preview_calls_per_slot"])
                ),
                "oracle_tx_gap_mean": float(np.mean(result["oracle_tx_gap_mean"])),
                "effective_risk_weight_mean": float(np.mean(result["effective_risk_weight"])),
                "avg_reward": float(np.mean(result["episode_reward"])),
            }
        )
    return rows


def write_csv(path, rows):
    """Write limited-CSI summary CSV."""
    ensure_parent_dir(path)
    fieldnames = [
        "error_std",
        "probe_budget",
        "policy",
        "gain_margin",
        "power_margin",
        "risk_weight",
        "risk_power_weight",
        "risk_invite_threshold",
        "episodes",
        "num_seeds",
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
        "true_opportunities_mean",
        "missed_opportunities_mean",
        "false_positive_rate",
        "execution_failure_rate",
        "failure_slot_count_mean",
        "failure_slot_rate",
        "missed_opportunity_rate",
        "decision_preview_calls_per_slot_mean",
        "oracle_tx_gap_mean",
        "effective_risk_weight_mean",
        "avg_reward",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def policy_label(row):
    """Build a compact plot/table label for one summary row."""
    label = row["policy"]
    if row["probe_budget"] not in {0, row.get("num_codebook_states", -1)}:
        label += f" B={int(row['probe_budget'])}"
    if row["policy"] == POLICY_ROBUST_ROTATING_GRID:
        label += f" gm={float(row['gain_margin']):g} pm={float(row['power_margin']):g}"
    if row["policy"] in {POLICY_RISK_AWARE_ROTATING_GRID, POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID}:
        label += (
            f" rw={float(row['risk_weight']):g}"
            f" rp={float(row['risk_power_weight']):g}"
            f" rt={float(row['risk_invite_threshold']):g}"
        )
    return label


def print_summary(rows):
    """Print a compact limited-CSI summary."""
    print("=" * 158)
    print("Limited-CSI MS-AirComp Summary")
    print("=" * 158)
    print(
        f"{'Err':>6} {'B':>3} {'Policy':<36} {'gm':>5} {'pm':>5} {'rw':>5} {'erw':>5} {'rt':>5} "
        f"{'Success':>9} {'Perfect%':>9} {'Slots':>8} {'Energy':>10} "
        f"{'Fail%':>8} {'MissOpp%':>9} {'Preview':>8} {'Gap':>7}"
    )
    for row in rows:
        print(
            f"{row['error_std']:>6.3f} {row['probe_budget']:>3} {row['policy']:<36} "
            f"{row['gain_margin']:>5.2f} {row['power_margin']:>5.2f} "
            f"{row['risk_weight']:>5.2f} {row['effective_risk_weight_mean']:>5.2f} "
            f"{row['risk_invite_threshold']:>5.2f} "
            f"{row['success_mean']:>9.3f} {row['perfect_rate']:>8.2f}% "
            f"{row['slots_mean']:>8.3f} {row['total_energy_mean']:>10.3f} "
            f"{row['execution_failure_rate']:>7.2f}% {row['missed_opportunity_rate']:>8.2f}% "
            f"{row['decision_preview_calls_per_slot_mean']:>8.2f} {row['oracle_tx_gap_mean']:>7.3f}"
        )


def plot_results(rows, args, output_prefix):
    """Plot perfect coverage, latency, and execution failure vs CSI error."""
    labels = []
    for row in rows:
        label = policy_label(row)
        if label not in labels:
            labels.append(label)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(idx % 20) for idx, label in enumerate(labels)}

    for label in labels:
        label_rows = sorted(
            [row for row in rows if policy_label(row) == label],
            key=lambda row: row["error_std"],
        )
        x = [row["error_std"] for row in label_rows]
        axes[0].plot(
            x,
            [row["perfect_rate"] for row in label_rows],
            marker="o",
            linewidth=1.7,
            label=label,
            color=colors[label],
        )
        axes[1].plot(
            x,
            [row["slots_mean"] for row in label_rows],
            marker="o",
            linewidth=1.7,
            label=label,
            color=colors[label],
        )
        axes[2].plot(
            x,
            [row["execution_failure_rate"] for row in label_rows],
            marker="o",
            linewidth=1.7,
            label=label,
            color=colors[label],
        )

    axes[0].set_title("Perfect Coverage vs CSI Error")
    axes[0].set_ylabel("Perfect Episodes (%)")
    axes[0].set_ylim(0.0, 103.0)
    axes[1].set_title("Latency vs CSI Error")
    axes[1].set_ylabel("Slots Used")
    axes[1].set_ylim(0.0, args.num_slots + 1)
    axes[2].set_title("Execution Failure vs CSI Error")
    axes[2].set_ylabel("Failed Invited Nodes (%)")

    for ax in axes:
        ax.set_xlabel("Equivalent Channel Error Std")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)

    fig.tight_layout()
    path = f"{output_prefix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    """Run limited-CSI MS-AirComp evaluation."""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)
    run_seeds = make_run_seeds(args)
    episode_seed_sets = [make_episode_seeds(args, run_seed) for run_seed in run_seeds]

    print("=" * 96)
    print(
        f"Limited-CSI MS-AirComp: episodes={args.episodes}, num_seeds={args.num_seeds}, "
        f"errors={args.error_std_values}, budgets={args.probe_budgets}"
    )
    print(
        f"Robust margins: gain={args.robust_gain_margins}, power={args.robust_power_margins}, "
        f"risk_weights={args.risk_weights}, risk_power={args.risk_power_weights}, "
        f"risk_thresholds={args.risk_invite_thresholds}, "
        f"adaptive_risk_base={args.adaptive_risk_base_weights}, "
        f"adaptive_error_ref={args.adaptive_risk_error_ref:g}, output_prefix={output_prefix}"
    )
    print("=" * 96)

    rows = []
    for error_std in args.error_std_values:
        print("=" * 96)
        print(f"Limited-CSI error std={error_std:g}")
        print("=" * 96)
        seed_result_sets = []
        for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
            print(f"Run seed [{run_idx}/{len(run_seeds)}]: {run_seeds[run_idx - 1]}")
            seed_result_sets.append(policy_suite_for_error(args, episode_seeds, error_std))
        rows.extend(summarize_results(args, aggregate_seed_results(seed_result_sets)))

    print_summary(rows)
    write_csv(f"{output_prefix}.csv", rows)
    if not args.no_plots:
        plot_results(rows, args, output_prefix)


if __name__ == "__main__":
    main()
