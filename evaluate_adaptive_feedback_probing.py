"""
Evaluate adaptive rotating backup probing under aggregate bandit feedback.

The policy uses Rotating Feedback Probe B=1 as the default action. After the
primary rotating probe is observed, it spends one extra probe only when the
observed feasible-node count is below the required remaining completion rate:

    observed_tx_count < gate_ratio * remaining_nodes / remaining_slots

This tests whether conditional backup probing has value before adding a learned
gate or learned backup selector.
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evaluate_bandit_feedback_ms_aircomp as bandit
import evaluate_bandit_feedback_stress_sweep as stress
from evaluate_policy_comparison import (
    codebook_index_to_action,
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
    update_energy,
)


POLICY_ADAPTIVE_ROTATING_BACKUP = "Adaptive Rotating Backup"
ADAPTIVE_OFFSET = 0xD1B54A32

BACKUP_STRATEGIES = {
    "next",
    "opposite",
    "least_recent",
    "best_history",
    "hybrid",
}

STRATEGY_OFFSETS = {
    "next": 0x01000193,
    "opposite": 0x811C9DC5,
    "least_recent": 0x85EBCA6B,
    "best_history": 0xC2B2AE35,
    "hybrid": 0x27D4EB2F,
}

CSV_FIELDS = [
    "scenario",
    "scenario_label",
    "scenario_description",
    "num_nodes",
    "num_slots",
    "num_irs_elements",
    "num_codebook_states",
    "g_th",
    "alpha_th",
    "slot_cost",
    "probe_cost",
    "utility_mean",
    "total_probe_calls_mean",
    "feedback_noise_std",
    "probe_budget",
    "budget_fraction",
    "policy",
    "backup_strategy",
    "gate_ratio",
    "adaptive_trigger_rate",
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
    "probe_calls_per_slot_mean",
    "oracle_match_rate",
    "oracle_tx_gap_mean",
    "observed_score_mean",
    "observed_tx_fraction_mean",
    "avg_reward",
]

NUMERIC_RESULT_KEYS = bandit.NUMERIC_RESULT_KEYS + ("adaptive_trigger_rate",)


def parse_csv_items(value):
    """Parse a comma-separated list, preserving non-empty strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    """Parse adaptive probing arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive rotating backup probing under noisy aggregate feedback."
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--scenarios", default="short_slots,compound_hard")
    parser.add_argument("--feedback-noise-std-values", default="0.2,0.5")
    parser.add_argument("--gate-ratios", default="0.7,0.9,1.1")
    parser.add_argument("--backup-strategies", default="next,least_recent,best_history,hybrid")
    parser.add_argument("--probe-budgets", default="1,2")
    parser.add_argument("--baseline-policies", default="no_irs,random_irs,oracle,full_feedback")
    parser.add_argument("--probe-policies", default="rotating,ucb,thompson")
    parser.add_argument("--slot-cost", type=float, default=0.10)
    parser.add_argument("--probe-cost", type=float, default=0.005)
    parser.add_argument("--power-feedback-noise-std", type=float, default=0.0)
    parser.add_argument("--feedback-power-weight", type=float, default=0.05)
    parser.add_argument("--ucb-coeff", type=float, default=0.25)
    parser.add_argument("--thompson-std", type=float, default=0.20)
    parser.add_argument("--bandit-lr", type=float, default=0.60)
    parser.add_argument("--bandit-prior", type=float, default=0.0)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--fixed-irs-index", type=int, default=7)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """Validate CLI arguments and parse comma-separated options."""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if args.slot_cost < 0.0:
        raise ValueError("--slot-cost must be non-negative")
    if args.probe_cost < 0.0:
        raise ValueError("--probe-cost must be non-negative")

    if args.scenarios == "all":
        args.scenarios = list(stress.SCENARIO_PRESETS)
    else:
        args.scenarios = parse_csv_items(args.scenarios)
    unknown_scenarios = [name for name in args.scenarios if name not in stress.SCENARIO_PRESETS]
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {unknown_scenarios}")

    args.feedback_noise_std_values = bandit.parse_float_list(args.feedback_noise_std_values)
    if not args.feedback_noise_std_values:
        raise ValueError("--feedback-noise-std-values must contain at least one value")
    if any(value < 0.0 for value in args.feedback_noise_std_values):
        raise ValueError("--feedback-noise-std-values must be non-negative")

    args.gate_ratios = bandit.parse_float_list(args.gate_ratios)
    if not args.gate_ratios:
        raise ValueError("--gate-ratios must contain at least one value")
    if any(value < 0.0 for value in args.gate_ratios):
        raise ValueError("--gate-ratios must be non-negative")

    args.backup_strategies = parse_csv_items(args.backup_strategies)
    unknown_strategies = [name for name in args.backup_strategies if name not in BACKUP_STRATEGIES]
    if unknown_strategies:
        raise ValueError(f"Unknown backup strategies: {unknown_strategies}")

    args.baseline_policies = parse_csv_items(args.baseline_policies)
    unknown_baselines = [
        name for name in args.baseline_policies if name not in stress.BASELINE_POLICIES
    ]
    if unknown_baselines:
        raise ValueError(f"Unknown baseline policies: {unknown_baselines}")

    args.probe_policies = parse_csv_items(args.probe_policies)
    unknown_probe_policies = [
        name for name in args.probe_policies if name not in stress.PROBE_POLICIES
    ]
    if unknown_probe_policies:
        raise ValueError(f"Unknown probe policies: {unknown_probe_policies}")


def build_bandit_args(args, scenario_name):
    """Build the namespace expected by the aggregate-feedback evaluator."""
    config = stress.scenario_config(scenario_name)
    bandit_args = argparse.Namespace(
        episodes=args.episodes,
        seed=args.seed,
        num_seeds=args.num_seeds,
        seed_stride=args.seed_stride,
        probe_budgets=args.probe_budgets,
        feedback_noise_std_values=",".join(str(value) for value in args.feedback_noise_std_values),
        power_feedback_noise_std=args.power_feedback_noise_std,
        feedback_power_weight=args.feedback_power_weight,
        ucb_coeff=args.ucb_coeff,
        thompson_std=args.thompson_std,
        bandit_lr=args.bandit_lr,
        bandit_prior=args.bandit_prior,
        num_nodes=config["num_nodes"],
        num_slots=config["num_slots"],
        num_irs_elements=config["num_irs_elements"],
        num_codebook_states=config["num_codebook_states"],
        g_th=args.g_th,
        alpha_th=args.alpha_th,
        fixed_irs_index=args.fixed_irs_index,
        slot_cost=args.slot_cost,
        probe_cost=args.probe_cost,
        output_prefix=None,
        no_plots=args.no_plots,
    )
    bandit.validate_args(bandit_args)
    return bandit_args


def resolve_output_prefix(args):
    """Resolve output prefix for CSVs and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    scenario_label = "-".join(args.scenarios)
    noise_label = "-".join(
        format_float_for_suffix(value) for value in args.feedback_noise_std_values
    )
    gate_label = "-".join(format_float_for_suffix(value) for value in args.gate_ratios)
    strategy_label = "-".join(args.backup_strategies)
    suffix = (
        f"ep{args.episodes}_runs{args.num_seeds}_{seed_label}_{scenario_label}_"
        f"fbnoise{noise_label}_gate{gate_label}_{strategy_label}"
    )
    output_prefix = os.path.join("results", "bandit_feedback", f"adaptive_feedback_probe_{suffix}")
    ensure_parent_dir(output_prefix)
    return output_prefix


def adaptive_rng(episode_seed, feedback_noise_std, gate_ratio, backup_strategy, salt=0):
    """Create deterministic RNG streams for adaptive probing."""
    if episode_seed is None:
        return np.random.default_rng()
    noise_tag = int(round(float(feedback_noise_std) * 1_000_000))
    gate_tag = int(round(float(gate_ratio) * 1_000_000))
    seed = (
        int(episode_seed)
        + ADAPTIVE_OFFSET
        + noise_tag * 0x85EBCA6B
        + gate_tag * 0x9E3779B1
        + STRATEGY_OFFSETS[backup_strategy]
        + int(salt) * 0x165667B1
    ) % (2**32)
    return np.random.default_rng(seed)


def initialize_adaptive_state(args):
    """Initialize observable per-codebook feedback history."""
    c_count = int(args.num_codebook_states)
    return {
        "counts": np.zeros(c_count, dtype=float),
        "means": np.full(c_count, float(args.bandit_prior), dtype=float),
        "age": np.full(c_count, float(args.num_slots + 1), dtype=float),
    }


def update_adaptive_state(state, feedbacks, args):
    """Update observable feedback history after one decision slot."""
    state["age"] += 1.0
    lr = float(args.bandit_lr)
    for feedback in feedbacks:
        index = int(feedback["irs_index"])
        old_count = float(state["counts"][index])
        old_mean = float(state["means"][index])
        score = float(feedback["observed_score"])
        if old_count <= 0.0:
            new_mean = score
        else:
            new_mean = (1.0 - lr) * old_mean + lr * score
        state["counts"][index] = old_count + 1.0
        state["means"][index] = new_mean
        state["age"][index] = 0.0


def rotating_primary_index(args, slot_idx):
    """Return the B=1 rotating codebook index for this slot."""
    return int(bandit.grid_indices(args.num_codebook_states, 1, offset=slot_idx)[0])


def best_excluding(values, excluded_index, prefer_high=True):
    """Return the best index by value while excluding one index."""
    best_index = None
    best_value = None
    for index, value in enumerate(values):
        if int(index) == int(excluded_index):
            continue
        clean_value = float(value)
        if best_index is None:
            best_index = int(index)
            best_value = clean_value
            continue
        if prefer_high and clean_value > best_value:
            best_index = int(index)
            best_value = clean_value
        elif not prefer_high and clean_value < best_value:
            best_index = int(index)
            best_value = clean_value
    if best_index is None:
        return int(excluded_index)
    return int(best_index)


def choose_backup_index(args, state, primary_index, backup_strategy):
    """Choose one backup codebook using only past aggregate feedback history."""
    c_count = int(args.num_codebook_states)
    primary_index = int(primary_index)

    if backup_strategy == "next":
        return int((primary_index + 1) % c_count)

    if backup_strategy == "opposite":
        return int((primary_index + max(c_count // 2, 1)) % c_count)

    if backup_strategy == "least_recent":
        return best_excluding(state["age"], primary_index, prefer_high=True)

    counts = np.asarray(state["counts"], dtype=float)
    means = np.asarray(state["means"], dtype=float)
    if backup_strategy == "best_history":
        if np.any(np.delete(counts, primary_index) > 0.0):
            masked_means = means.copy()
            masked_means[counts <= 0.0] = -np.inf
            return best_excluding(masked_means, primary_index, prefer_high=True)
        return int((primary_index + 1) % c_count)

    if backup_strategy == "hybrid":
        unobserved = counts <= 0.0
        unobserved[primary_index] = False
        if np.any(unobserved):
            masked_age = np.where(unobserved, state["age"], -np.inf)
            return best_excluding(masked_age, primary_index, prefer_high=True)
        return best_excluding(means, primary_index, prefer_high=True)

    raise ValueError(f"Unknown backup strategy: {backup_strategy}")


def should_probe_backup(env, args, primary_feedback, gate_ratio):
    """Return whether the primary rotating feedback is too weak for the deadline."""
    remaining_nodes = int(args.num_nodes - np.sum(env.transmitted_flags))
    remaining_slots = max(int(args.num_slots - env.current_slot), 1)
    required_tx_per_slot = float(remaining_nodes) / float(remaining_slots)
    observed_tx_count = float(primary_feedback["observed_tx_fraction"]) * float(args.num_nodes)
    return observed_tx_count < float(gate_ratio) * required_tx_per_slot


def choose_adaptive_candidate(
    env,
    args,
    slot_idx,
    feedback_noise_std,
    state,
    rng,
    gate_ratio,
    backup_strategy,
):
    """Probe rotating first, then conditionally probe one backup codebook."""
    primary_index = rotating_primary_index(args, slot_idx)
    primary_candidate = bandit.preview_codebook_candidate(env, args, primary_index)
    primary_feedback = bandit.observe_probe_feedback(
        primary_candidate,
        args,
        feedback_noise_std,
        rng,
    )
    candidates = [primary_candidate]
    feedbacks = [primary_feedback]
    triggered = should_probe_backup(env, args, primary_feedback, gate_ratio)

    if triggered and args.num_codebook_states > 1:
        backup_index = choose_backup_index(args, state, primary_index, backup_strategy)
        if int(backup_index) != int(primary_index):
            backup_candidate = bandit.preview_codebook_candidate(env, args, backup_index)
            backup_feedback = bandit.observe_probe_feedback(
                backup_candidate,
                args,
                feedback_noise_std,
                rng,
            )
            candidates.append(backup_candidate)
            feedbacks.append(backup_feedback)

    update_adaptive_state(state, feedbacks, args)
    selected = bandit.select_from_feedback(candidates, feedbacks)
    return selected, feedbacks, len(feedbacks), float(triggered)


def evaluate_adaptive_policy(
    episode_seeds,
    args,
    feedback_noise_std,
    gate_ratio,
    backup_strategy,
    base_action,
):
    """Evaluate one adaptive rotating backup configuration."""
    env = bandit.make_env(args, irs_phase_mode="codebook")
    policy_name = adaptive_policy_name(backup_strategy, gate_ratio)
    success_nodes = []
    avg_power = []
    rewards = []
    slots_used = []
    total_energy = []
    probe_calls_per_slot = []
    oracle_match_rate = []
    oracle_tx_gap_mean = []
    observed_score_mean = []
    observed_tx_fraction_mean = []
    adaptive_trigger_rate = []

    print(f"Running {policy_name} noise={feedback_noise_std:g}...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        rng = adaptive_rng(episode_seed, feedback_noise_std, gate_ratio, backup_strategy)
        state = initialize_adaptive_state(args)
        episode_power = []
        episode_reward = 0.0
        episode_energy = 0.0
        episode_slots = args.num_slots
        episode_probe_calls = []
        episode_oracle_matches = []
        episode_oracle_gaps = []
        episode_observed_scores = []
        episode_observed_tx_fractions = []
        episode_triggers = []
        total_tx = 0

        for slot_idx in range(args.num_slots):
            oracle = bandit.full_oracle_candidate(env, args)
            selected, feedbacks, probe_calls, triggered = choose_adaptive_candidate(
                env,
                args,
                slot_idx,
                feedback_noise_std,
                state,
                rng,
                gate_ratio,
                backup_strategy,
            )
            selected_index = int(selected["irs_index"])

            episode_probe_calls.append(int(probe_calls))
            episode_triggers.append(float(triggered))
            episode_oracle_matches.append(float(selected_index == int(oracle["irs_index"])))
            episode_oracle_gaps.append(
                max(0.0, float(oracle["tx_this_slot"]) - float(selected["tx_this_slot"]))
            )
            for feedback in feedbacks:
                episode_observed_scores.append(float(feedback["observed_score"]))
                episode_observed_tx_fractions.append(float(feedback["observed_tx_fraction"]))

            action = base_action.copy()
            action[2] = codebook_index_to_action(selected_index, args.num_codebook_states)

            _obs, reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            episode_reward += float(reward)
            episode_slots = int(info.get("slots_used", slot_idx + 1))
            episode_energy = update_energy(episode_energy, info)
            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))
            if terminated or truncated:
                break

        success_nodes.append(total_tx)
        avg_power.append(float(np.mean(episode_power)) if episode_power else 0.0)
        rewards.append(float(episode_reward))
        slots_used.append(int(episode_slots))
        total_energy.append(float(episode_energy))
        active_slots = max(len(episode_probe_calls), 1)
        probe_calls_per_slot.append(float(sum(episode_probe_calls)) / active_slots)
        oracle_match_rate.append(
            float(np.mean(episode_oracle_matches)) if episode_oracle_matches else 0.0
        )
        oracle_tx_gap_mean.append(
            float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0
        )
        observed_score_mean.append(
            float(np.mean(episode_observed_scores)) if episode_observed_scores else 0.0
        )
        observed_tx_fraction_mean.append(
            float(np.mean(episode_observed_tx_fractions)) if episode_observed_tx_fractions else 0.0
        )
        adaptive_trigger_rate.append(
            float(np.mean(episode_triggers)) if episode_triggers else 0.0
        )

        bandit.print_progress(
            policy_name,
            feedback_noise_std,
            2,
            ep,
            args.episodes,
            success_nodes,
            args.num_nodes,
        )

    return {
        "name": policy_name,
        "feedback_noise_std": float(feedback_noise_std),
        "probe_budget": 2,
        "backup_strategy": backup_strategy,
        "gate_ratio": float(gate_ratio),
        "success_nodes": np.asarray(success_nodes, dtype=float),
        "avg_power": np.asarray(avg_power, dtype=float),
        "episode_reward": np.asarray(rewards, dtype=float),
        "slots_used": np.asarray(slots_used, dtype=float),
        "total_energy": np.asarray(total_energy, dtype=float),
        "probe_calls_per_slot": np.asarray(probe_calls_per_slot, dtype=float),
        "oracle_match_rate": np.asarray(oracle_match_rate, dtype=float),
        "oracle_tx_gap_mean": np.asarray(oracle_tx_gap_mean, dtype=float),
        "observed_score_mean": np.asarray(observed_score_mean, dtype=float),
        "observed_tx_fraction_mean": np.asarray(observed_tx_fraction_mean, dtype=float),
        "adaptive_trigger_rate": np.asarray(adaptive_trigger_rate, dtype=float),
    }


def adaptive_policy_name(backup_strategy, gate_ratio):
    """Return a stable display name for one adaptive configuration."""
    return f"{POLICY_ADAPTIVE_ROTATING_BACKUP} {backup_strategy} r={gate_ratio:g}"


def seed_summary(result):
    """Compress one run seed result to seed-level means."""
    summary = {}
    for key in NUMERIC_RESULT_KEYS:
        if key in result:
            summary[key] = float(np.mean(result[key]))
        else:
            summary[key] = 0.0
    return summary


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
            "feedback_noise_std": parts[0]["feedback_noise_std"],
            "probe_budget": parts[0]["probe_budget"],
            "backup_strategy": parts[0].get("backup_strategy", ""),
            "gate_ratio": parts[0].get("gate_ratio", ""),
        }
        for key in NUMERIC_RESULT_KEYS:
            arrays = [
                part[key] if key in part else np.zeros_like(part["success_nodes"], dtype=float)
                for part in parts
            ]
            aggregated[key] = np.concatenate(arrays)
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
    """Convert aggregated results to rows."""
    rows = []
    for result in results:
        success_mean, success_ci95 = metric_mean_ci(result, "success_nodes")
        slots_mean, slots_ci95 = metric_mean_ci(result, "slots_used")
        energy_mean, energy_ci95 = metric_mean_ci(result, "total_energy")
        total_probe_calls = float(np.mean(result["probe_calls_per_slot"])) * slots_mean
        utility = (
            success_mean
            - float(args.slot_cost) * slots_mean
            - float(args.probe_cost) * total_probe_calls
        )
        rows.append(
            {
                "feedback_noise_std": float(result["feedback_noise_std"]),
                "probe_budget": int(result["probe_budget"]),
                "budget_fraction": float(result["probe_budget"]) / args.num_codebook_states,
                "policy": result["name"],
                "backup_strategy": result.get("backup_strategy", ""),
                "gate_ratio": result.get("gate_ratio", ""),
                "adaptive_trigger_rate": float(np.mean(result["adaptive_trigger_rate"])),
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
                "probe_calls_per_slot_mean": float(np.mean(result["probe_calls_per_slot"])),
                "total_probe_calls_mean": total_probe_calls,
                "utility_mean": utility,
                "oracle_match_rate": float(np.mean(result["oracle_match_rate"]) * 100.0),
                "oracle_tx_gap_mean": float(np.mean(result["oracle_tx_gap_mean"])),
                "observed_score_mean": float(np.mean(result["observed_score_mean"])),
                "observed_tx_fraction_mean": float(np.mean(result["observed_tx_fraction_mean"])),
                "avg_reward": float(np.mean(result["episode_reward"])),
            }
        )
    return rows


def attach_scenario_metadata(rows, args, scenario_name, bandit_args):
    """Add scenario metadata to summary rows."""
    preset = stress.SCENARIO_PRESETS[scenario_name]
    for row in rows:
        row.update(
            {
                "scenario": scenario_name,
                "scenario_label": preset["label"],
                "scenario_description": preset["description"],
                "num_nodes": bandit_args.num_nodes,
                "num_slots": bandit_args.num_slots,
                "num_irs_elements": bandit_args.num_irs_elements,
                "num_codebook_states": bandit_args.num_codebook_states,
                "g_th": bandit_args.g_th,
                "alpha_th": bandit_args.alpha_th,
                "slot_cost": args.slot_cost,
                "probe_cost": args.probe_cost,
            }
        )
    return rows


def evaluate_suite(args, bandit_args, episode_seed_sets, feedback_noise_std, base_action):
    """Run selected baselines, probe policies, and adaptive configurations."""
    seed_result_sets = []
    baseline_policy_names = [stress.BASELINE_POLICIES[name] for name in args.baseline_policies]
    probe_policy_names = [stress.PROBE_POLICIES[name] for name in args.probe_policies]
    run_seeds = make_run_seeds(bandit_args)

    for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
        print(f"Run seed [{run_idx}/{len(episode_seed_sets)}]: {run_seeds[run_idx - 1]}")
        seed_results = []

        for policy_name in baseline_policy_names:
            budget = (
                bandit_args.num_codebook_states
                if policy_name in {bandit.POLICY_ORACLE_FULL, bandit.POLICY_FULL_FEEDBACK}
                else 0
            )
            seed_results.append(
                bandit.evaluate_policy(
                    episode_seeds,
                    bandit_args,
                    feedback_noise_std,
                    policy_name,
                    budget,
                    base_action,
                )
            )

        for budget in bandit_args.probe_budgets:
            for policy_name in probe_policy_names:
                seed_results.append(
                    bandit.evaluate_policy(
                        episode_seeds,
                        bandit_args,
                        feedback_noise_std,
                        policy_name,
                        budget,
                        base_action,
                    )
                )

        for gate_ratio in args.gate_ratios:
            for backup_strategy in args.backup_strategies:
                seed_results.append(
                    evaluate_adaptive_policy(
                        episode_seeds,
                        bandit_args,
                        feedback_noise_std,
                        gate_ratio,
                        backup_strategy,
                        base_action,
                    )
                )

        seed_result_sets.append(seed_results)

    return summarize_results(bandit_args, aggregate_seed_results(seed_result_sets))


def write_csv(path, rows):
    """Write the adaptive probing summary CSV."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def is_oracle(row):
    """Return whether a row is an offline oracle diagnostic."""
    return row["policy"] == bandit.POLICY_ORACLE_FULL


def compact_policy_label(row):
    """Return compact labels for console and plots."""
    policy = str(row["policy"])
    if policy.startswith(POLICY_ADAPTIVE_ROTATING_BACKUP):
        strategy = row.get("backup_strategy", "")
        ratio = row.get("gate_ratio", "")
        return f"Adaptive {strategy} r={ratio:g}" if ratio != "" else f"Adaptive {strategy}"
    label = bandit.policy_label(row)
    return label.replace(" Feedback Probe", "").replace(" Feedback", "")


def print_summary(rows):
    """Print best policies per scenario/noise pair."""
    print("=" * 144)
    print("Adaptive Feedback Probing Summary")
    print("=" * 144)
    print(
        f"{'Scenario':<16} {'Noise':>6} {'Best non-oracle':<40} {'Utility':>9} "
        f"{'Success':>9} {'Perfect%':>9} {'Slots':>7} {'Probes':>8} {'Trig%':>8} "
        f"{'Oracle':>9}"
    )
    groups = sorted({(row["scenario"], row["feedback_noise_std"]) for row in rows})
    for scenario_name, noise in groups:
        subset = [
            row
            for row in rows
            if row["scenario"] == scenario_name and row["feedback_noise_std"] == noise
        ]
        non_oracle = [row for row in subset if not is_oracle(row)]
        oracle_rows = [row for row in subset if is_oracle(row)]
        best = max(non_oracle, key=lambda row: row["utility_mean"])
        oracle_success = oracle_rows[0]["success_mean"] if oracle_rows else float("nan")
        print(
            f"{scenario_name:<16} {noise:>6.3f} {compact_policy_label(best):<40} "
            f"{best['utility_mean']:>9.3f} {best['success_mean']:>9.3f} "
            f"{best['perfect_rate']:>8.2f}% {best['slots_mean']:>7.3f} "
            f"{best['total_probe_calls_mean']:>8.2f} "
            f"{100.0 * float(best['adaptive_trigger_rate']):>7.2f}% "
            f"{oracle_success:>9.3f}"
        )


def plot_results(rows, output_prefix):
    """Plot latency/probe tradeoffs for each scenario."""
    scenarios = [name for name in stress.SCENARIO_PRESETS if any(row["scenario"] == name for row in rows)]
    if not scenarios:
        return

    labels = []
    for row in rows:
        if is_oracle(row):
            continue
        label = compact_policy_label(row)
        if label not in labels:
            labels.append(label)

    cols = 2
    plot_rows = int(np.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(plot_rows, cols, figsize=(15, 5 * plot_rows), squeeze=False)
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(idx % 20) for idx, label in enumerate(labels)}

    for ax, scenario_name in zip(axes.ravel(), scenarios):
        subset = [row for row in rows if row["scenario"] == scenario_name and not is_oracle(row)]
        for row in subset:
            label = compact_policy_label(row)
            marker = "s" if str(row["policy"]).startswith(POLICY_ADAPTIVE_ROTATING_BACKUP) else "o"
            ax.scatter(
                row["total_probe_calls_mean"],
                row["slots_mean"],
                s=32 + row["perfect_rate"] * 0.35,
                color=colors[label],
                alpha=0.78,
                marker=marker,
                label=label,
            )
        best = max(subset, key=lambda row: row["utility_mean"])
        ax.scatter(
            best["total_probe_calls_mean"],
            best["slots_mean"],
            s=190,
            facecolors="none",
            edgecolors="black",
            linewidths=1.7,
        )
        ax.set_title(stress.SCENARIO_PRESETS[scenario_name]["label"])
        ax.set_xlabel("Total Probe Calls / Episode")
        ax.set_ylabel("Slots Used")
        ax.grid(True, linestyle="--", alpha=0.35)

    for ax in axes.ravel()[len(scenarios):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="lower center",
        ncol=min(4, max(1, len(dedup))),
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = f"{output_prefix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    """Run adaptive feedback probing experiments."""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)

    print("=" * 96)
    print(
        f"Adaptive feedback probing: episodes={args.episodes}, num_seeds={args.num_seeds}, "
        f"scenarios={args.scenarios}, feedback_noise={args.feedback_noise_std_values}, "
        f"gate_ratios={args.gate_ratios}, backup_strategies={args.backup_strategies}"
    )
    print(
        f"Utility: success - {args.slot_cost:g} * slots - {args.probe_cost:g} * total_probes; "
        f"output_prefix={output_prefix}"
    )
    print("=" * 96)

    all_rows = []
    for scenario_name in args.scenarios:
        bandit_args = build_bandit_args(args, scenario_name)
        base_action = bandit.make_base_action(bandit_args)
        run_seeds = make_run_seeds(bandit_args)
        episode_seed_sets = [make_episode_seeds(bandit_args, run_seed) for run_seed in run_seeds]
        print("=" * 96)
        print(f"Scenario: {scenario_name} ({stress.SCENARIO_PRESETS[scenario_name]['description']})")
        print("=" * 96)

        for feedback_noise_std in bandit_args.feedback_noise_std_values:
            print("=" * 96)
            print(f"Aggregate feedback noise std={feedback_noise_std:g}")
            print("=" * 96)
            rows = evaluate_suite(
                args,
                bandit_args,
                episode_seed_sets,
                feedback_noise_std,
                base_action,
            )
            all_rows.extend(attach_scenario_metadata(rows, args, scenario_name, bandit_args))

    print_summary(all_rows)
    write_csv(f"{output_prefix}.csv", all_rows)
    if not args.no_plots:
        plot_results(all_rows, output_prefix)


if __name__ == "__main__":
    main()
