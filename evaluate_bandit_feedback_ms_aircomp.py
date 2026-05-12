"""
Bandit-feedback IRS-assisted multi-slot AirComp evaluation.

This experiment is stricter than feature argmax, partial preview, and the
limited-CSI equivalent-channel sweep. A policy does not observe per-codebook CSI
or per-node schedulability masks. It may probe only B IRS codebook indices per
decision slot and receives noisy aggregate feedback for each probed index:
estimated feasible-node fraction and average transmit power. The data slot is
then executed on the true channel.

The goal is to model the research setting where IRS codebook selection is an
online probing problem under limited feedback rather than a full-CSI preview
problem.
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_partial_probing_sweep import grid_indices, unique_fill
from evaluate_policy_comparison import (
    codebook_index_to_action,
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
    physical_to_action,
    update_energy,
)
from test_env import MSAirCompEnv


POLICY_NO_IRS = "No IRS"
POLICY_FIXED_IRS = "Fixed IRS"
POLICY_RANDOM_IRS = "Random IRS"
POLICY_ORACLE_FULL = "Oracle Full Preview"
POLICY_FULL_FEEDBACK = "Full Noisy Feedback"
POLICY_RANDOM_PROBE = "Random Feedback Probe"
POLICY_ROTATING_GRID = "Rotating Feedback Probe"
POLICY_UCB_PROBE = "UCB Feedback Probe"
POLICY_THOMPSON_PROBE = "Thompson Feedback Probe"


PROBE_POLICIES = {
    POLICY_RANDOM_PROBE,
    POLICY_ROTATING_GRID,
    POLICY_UCB_PROBE,
    POLICY_THOMPSON_PROBE,
}


POLICY_OFFSETS = {
    POLICY_NO_IRS: 0x082EFA98,
    POLICY_FIXED_IRS: 0x299F31D0,
    POLICY_RANDOM_IRS: 0xA4093822,
    POLICY_ORACLE_FULL: 0x243F6A88,
    POLICY_FULL_FEEDBACK: 0x85A308D3,
    POLICY_RANDOM_PROBE: 0x13198A2E,
    POLICY_ROTATING_GRID: 0x03707344,
    POLICY_UCB_PROBE: 0x452821E6,
    POLICY_THOMPSON_PROBE: 0xBE5466CF,
}


NUMERIC_RESULT_KEYS = (
    "success_nodes",
    "avg_power",
    "episode_reward",
    "slots_used",
    "total_energy",
    "probe_calls_per_slot",
    "oracle_match_rate",
    "oracle_tx_gap_mean",
    "observed_score_mean",
    "observed_tx_fraction_mean",
)


def parse_int_list(value):
    """Parse a comma-separated integer list such as '1,2,4,8'."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    """Parse a comma-separated float list such as '0,0.05,0.1'."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    """Parse bandit-feedback evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate IRS-assisted MS-AirComp with noisy aggregate probing feedback."
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--probe-budgets", default="1,2,4,8")
    parser.add_argument("--feedback-noise-std-values", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--power-feedback-noise-std", type=float, default=0.0)
    parser.add_argument("--feedback-power-weight", type=float, default=0.05)
    parser.add_argument("--ucb-coeff", type=float, default=0.25)
    parser.add_argument("--thompson-std", type=float, default=0.20)
    parser.add_argument("--bandit-lr", type=float, default=0.60)
    parser.add_argument("--bandit-prior", type=float, default=0.0)
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
    """Validate sizes, budgets, and feedback parameters."""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    for name in ("num_nodes", "num_slots", "num_irs_elements", "num_codebook_states"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.g_th <= 0.0:
        raise ValueError("--g-th must be positive")
    if args.alpha_th <= 0.0:
        raise ValueError("--alpha-th must be positive")
    if args.power_feedback_noise_std < 0.0:
        raise ValueError("--power-feedback-noise-std must be non-negative")
    if args.feedback_power_weight < 0.0:
        raise ValueError("--feedback-power-weight must be non-negative")
    if args.ucb_coeff < 0.0:
        raise ValueError("--ucb-coeff must be non-negative")
    if args.thompson_std < 0.0:
        raise ValueError("--thompson-std must be non-negative")
    if not 0.0 <= args.bandit_lr <= 1.0:
        raise ValueError("--bandit-lr must be in [0, 1]")

    budgets = parse_int_list(args.probe_budgets)
    if not budgets:
        raise ValueError("--probe-budgets must contain at least one value")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("--probe-budgets must contain positive integers")
    args.probe_budgets = sorted({min(int(budget), args.num_codebook_states) for budget in budgets})

    args.feedback_noise_std_values = parse_float_list(args.feedback_noise_std_values)
    if not args.feedback_noise_std_values:
        raise ValueError("--feedback-noise-std-values must contain at least one value")
    if any(value < 0.0 for value in args.feedback_noise_std_values):
        raise ValueError("--feedback-noise-std-values must be non-negative")

    args.fixed_irs_index = int(np.clip(args.fixed_irs_index, 0, args.num_codebook_states - 1))


def resolve_output_prefix(args):
    """Resolve output prefix for CSVs and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    seed_label = "unseeded" if args.seed < 0 else f"seed{args.seed}"
    budget_label = "-".join(format_float_for_suffix(value) for value in args.probe_budgets)
    noise_label = "-".join(format_float_for_suffix(value) for value in args.feedback_noise_std_values)
    suffix = (
        f"ep{args.episodes}_runs{args.num_seeds}_{seed_label}_b{budget_label}_"
        f"fbnoise{noise_label}"
    )
    output_prefix = os.path.join("results", "bandit_feedback", f"bandit_feedback_ms_aircomp_{suffix}")
    ensure_parent_dir(output_prefix)
    return output_prefix


def make_env(args, irs_phase_mode="codebook"):
    """Create the MS-AirComp environment for the selected IRS mode."""
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode=irs_phase_mode,
    )


def make_base_action(args):
    """Build the fixed g_th/alpha_th action prefix shared by policies."""
    g_action = physical_to_action(args.g_th, low=0.001, scale=0.05)
    alpha_action = physical_to_action(args.alpha_th, low=0.05, scale=0.05)
    return np.array([g_action, alpha_action, 0.0], dtype=np.float32)


def stable_rng(episode_seed, policy_name, budget, feedback_noise_std, salt=0):
    """Create deterministic RNG streams for probing and noisy feedback."""
    if episode_seed is None:
        return np.random.default_rng()
    noise_tag = int(round(float(feedback_noise_std) * 1_000_000))
    seed = (
        int(episode_seed)
        + POLICY_OFFSETS[policy_name]
        + int(budget) * 0x9E3779B1
        + noise_tag * 0x85EBCA6B
        + int(salt) * 0x165667B1
    ) % (2**32)
    return np.random.default_rng(seed)


def candidate_key(candidate):
    """True greedy ranking key used only for oracle diagnostics."""
    tx_count = int(candidate["tx_this_slot"])
    power_avg = float(candidate["power_avg"])
    mean_gain = float(candidate["mean_gain_remaining"])
    power_tiebreak = -power_avg if tx_count > 0 else 0.0
    return tx_count, power_tiebreak, mean_gain


def best_candidate(candidates):
    """Return the best candidate under the true greedy ranking."""
    return max(candidates, key=candidate_key)


def preview_codebook_candidate(env, args, index):
    """Return hidden true metrics for one codebook index."""
    return env.preview_codebook_index(index, args.g_th, args.alpha_th)


def no_irs_candidate(env, args):
    """Return hidden true metrics for the no-IRS direct link."""
    metrics = env._compute_slot_metrics(  # pylint: disable=protected-access
        args.g_th,
        args.alpha_th,
        np.zeros(env.M, dtype=np.complex128),
    )
    return {
        "irs_index": -2,
        "tx_this_slot": int(metrics["tx_this_slot"]),
        "power_avg": float(metrics["power_avg"]),
        "mean_gain_remaining": float(metrics["mean_gain_remaining"]),
    }


def full_oracle_candidate(env, args):
    """Return the full-CSI greedy IRS candidate for offline diagnostics."""
    return best_candidate(
        [
            preview_codebook_candidate(env, args, index)
            for index in range(args.num_codebook_states)
        ]
    )


def observe_probe_feedback(candidate, args, feedback_noise_std, rng):
    """
    Convert hidden true candidate metrics into noisy aggregate probe feedback.

    The policy observes only aggregate feasible-node fraction and aggregate power,
    not node identities or equivalent channels.
    """
    true_tx_fraction = float(candidate["tx_this_slot"]) / float(max(args.num_nodes, 1))
    observed_tx_fraction = true_tx_fraction
    if feedback_noise_std > 0.0:
        observed_tx_fraction += float(rng.normal(0.0, feedback_noise_std))
    observed_tx_fraction = float(np.clip(observed_tx_fraction, 0.0, 1.0))

    observed_power = float(candidate["power_avg"])
    if args.power_feedback_noise_std > 0.0:
        observed_power += float(rng.normal(0.0, args.power_feedback_noise_std))
    observed_power = max(0.0, observed_power)

    observed_score = observed_tx_fraction - float(args.feedback_power_weight) * observed_power
    observed_score += float(rng.uniform(0.0, 1e-9))
    return {
        "irs_index": int(candidate["irs_index"]),
        "observed_tx_fraction": observed_tx_fraction,
        "observed_power": observed_power,
        "observed_score": float(observed_score),
    }


def initialize_feedback_state(args):
    """Initialize per-episode bandit state."""
    return {
        "counts": np.zeros(args.num_codebook_states, dtype=float),
        "means": np.full(args.num_codebook_states, float(args.bandit_prior), dtype=float),
    }


def update_feedback_state(state, feedbacks, args):
    """Update online aggregate-feedback estimates."""
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


def top_indices(scores, budget, num_codebook_states, rng):
    """Select top-scoring unique codebook indices with deterministic fill."""
    jitter = rng.uniform(0.0, 1e-9, size=num_codebook_states)
    order = np.argsort(-(np.asarray(scores, dtype=float) + jitter))
    return unique_fill(order, budget, num_codebook_states)


def select_feedback_probe_indices(policy_name, args, budget, slot_idx, state, rng):
    """Choose which IRS codebooks to probe using only past aggregate feedback."""
    c_count = int(args.num_codebook_states)
    budget = min(int(budget), c_count)
    if budget >= c_count:
        return list(range(c_count))

    if policy_name == POLICY_RANDOM_PROBE:
        return [int(index) for index in rng.choice(c_count, size=budget, replace=False)]

    if policy_name == POLICY_ROTATING_GRID:
        return grid_indices(c_count, budget, offset=slot_idx)

    counts = np.asarray(state["counts"], dtype=float)
    means = np.asarray(state["means"], dtype=float)
    if policy_name == POLICY_UCB_PROBE:
        total_count = max(float(np.sum(counts)), 1.0)
        bonus = float(args.ucb_coeff) * np.sqrt(np.log(total_count + 1.0) / (counts + 1.0))
        return top_indices(means + bonus, budget, c_count, rng)

    if policy_name == POLICY_THOMPSON_PROBE:
        std = float(args.thompson_std) / np.sqrt(counts + 1.0)
        samples = rng.normal(loc=means, scale=std)
        return top_indices(samples, budget, c_count, rng)

    raise ValueError(f"Unknown aggregate-feedback probing policy: {policy_name}")


def select_from_feedback(candidates, feedbacks):
    """Choose one probed candidate from noisy aggregate feedback."""
    feedback_by_index = {int(feedback["irs_index"]): feedback for feedback in feedbacks}

    def key(candidate):
        feedback = feedback_by_index[int(candidate["irs_index"])]
        return (
            float(feedback["observed_score"]),
            float(feedback["observed_tx_fraction"]),
            -float(feedback["observed_power"]),
        )

    return max(candidates, key=key)


def choose_policy_candidate(env, args, policy_name, budget, slot_idx, feedback_noise_std, state, rng):
    """Choose one codebook/no-IRS candidate and return visible feedback metadata."""
    if policy_name == POLICY_NO_IRS:
        return no_irs_candidate(env, args), [], 0

    if policy_name == POLICY_FIXED_IRS:
        return preview_codebook_candidate(env, args, args.fixed_irs_index), [], 0

    if policy_name == POLICY_RANDOM_IRS:
        index = int(rng.integers(0, args.num_codebook_states))
        return preview_codebook_candidate(env, args, index), [], 0

    if policy_name == POLICY_ORACLE_FULL:
        candidates = [
            preview_codebook_candidate(env, args, index)
            for index in range(args.num_codebook_states)
        ]
        return best_candidate(candidates), [], args.num_codebook_states

    if policy_name == POLICY_FULL_FEEDBACK:
        indices = list(range(args.num_codebook_states))
    else:
        indices = select_feedback_probe_indices(policy_name, args, budget, slot_idx, state, rng)

    candidates = [preview_codebook_candidate(env, args, index) for index in indices]
    feedbacks = [
        observe_probe_feedback(candidate, args, feedback_noise_std, rng)
        for candidate in candidates
    ]
    update_feedback_state(state, feedbacks, args)
    return select_from_feedback(candidates, feedbacks), feedbacks, len(indices)


def evaluate_policy(episode_seeds, args, feedback_noise_std, policy_name, budget, base_action):
    """Evaluate one policy for one feedback-noise level and probe budget."""
    irs_phase_mode = "none" if policy_name == POLICY_NO_IRS else "codebook"
    env = make_env(args, irs_phase_mode=irs_phase_mode)
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

    print(f"Running {policy_name} noise={feedback_noise_std:g} B={budget}...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        rng = stable_rng(episode_seed, policy_name, budget, feedback_noise_std)
        state = initialize_feedback_state(args)
        episode_power = []
        episode_reward = 0.0
        episode_energy = 0.0
        episode_slots = args.num_slots
        episode_probe_calls = []
        episode_oracle_matches = []
        episode_oracle_gaps = []
        episode_observed_scores = []
        episode_observed_tx_fractions = []
        total_tx = 0

        for slot_idx in range(args.num_slots):
            oracle = full_oracle_candidate(env, args)
            selected, feedbacks, probe_calls = choose_policy_candidate(
                env,
                args,
                policy_name,
                budget,
                slot_idx,
                feedback_noise_std,
                state,
                rng,
            )
            selected_index = int(selected["irs_index"])

            episode_probe_calls.append(int(probe_calls))
            episode_oracle_matches.append(float(selected_index == int(oracle["irs_index"])))
            episode_oracle_gaps.append(
                max(0.0, float(oracle["tx_this_slot"]) - float(selected["tx_this_slot"]))
            )
            for feedback in feedbacks:
                episode_observed_scores.append(float(feedback["observed_score"]))
                episode_observed_tx_fractions.append(float(feedback["observed_tx_fraction"]))

            action = base_action.copy()
            if selected_index >= 0:
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
        oracle_match_rate.append(float(np.mean(episode_oracle_matches)) if episode_oracle_matches else 0.0)
        oracle_tx_gap_mean.append(float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0)
        observed_score_mean.append(
            float(np.mean(episode_observed_scores)) if episode_observed_scores else 0.0
        )
        observed_tx_fraction_mean.append(
            float(np.mean(episode_observed_tx_fractions)) if episode_observed_tx_fractions else 0.0
        )

        print_progress(policy_name, feedback_noise_std, budget, ep, args.episodes, success_nodes, args.num_nodes)

    return {
        "name": policy_name,
        "feedback_noise_std": float(feedback_noise_std),
        "probe_budget": int(budget),
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
    }


def print_progress(name, feedback_noise_std, budget, ep, episodes, success_nodes, num_nodes):
    """Print progress at 10 percent intervals."""
    interval = max(episodes // 10, 1)
    if ep % interval == 0 or ep == episodes:
        recent = np.mean(success_nodes[-interval:])
        print(
            f"  {name} noise={feedback_noise_std:g} B={budget}: "
            f"[{ep:04d}/{episodes:04d}] recent success {recent:.2f}/{num_nodes}"
        )


def policy_suite_for_noise(args, episode_seeds, feedback_noise_std, base_action):
    """Run all aggregate-feedback policies for one noise level and run seed."""
    results = [
        evaluate_policy(episode_seeds, args, feedback_noise_std, POLICY_NO_IRS, 0, base_action),
        evaluate_policy(episode_seeds, args, feedback_noise_std, POLICY_FIXED_IRS, 0, base_action),
        evaluate_policy(episode_seeds, args, feedback_noise_std, POLICY_RANDOM_IRS, 0, base_action),
        evaluate_policy(
            episode_seeds,
            args,
            feedback_noise_std,
            POLICY_ORACLE_FULL,
            args.num_codebook_states,
            base_action,
        ),
        evaluate_policy(
            episode_seeds,
            args,
            feedback_noise_std,
            POLICY_FULL_FEEDBACK,
            args.num_codebook_states,
            base_action,
        ),
    ]

    for budget in args.probe_budgets:
        for policy_name in (
            POLICY_RANDOM_PROBE,
            POLICY_ROTATING_GRID,
            POLICY_UCB_PROBE,
            POLICY_THOMPSON_PROBE,
        ):
            results.append(
                evaluate_policy(
                    episode_seeds,
                    args,
                    feedback_noise_std,
                    policy_name,
                    budget,
                    base_action,
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
            "feedback_noise_std": parts[0]["feedback_noise_std"],
            "probe_budget": parts[0]["probe_budget"],
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


def summarize_results(args, results):
    """Convert aggregated results to CSV summary rows."""
    rows = []
    for result in results:
        success_mean, success_ci95 = metric_mean_ci(result, "success_nodes")
        slots_mean, slots_ci95 = metric_mean_ci(result, "slots_used")
        energy_mean, energy_ci95 = metric_mean_ci(result, "total_energy")
        rows.append(
            {
                "feedback_noise_std": float(result["feedback_noise_std"]),
                "probe_budget": int(result["probe_budget"]),
                "budget_fraction": float(result["probe_budget"]) / args.num_codebook_states,
                "policy": result["name"],
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
                "oracle_match_rate": float(np.mean(result["oracle_match_rate"]) * 100.0),
                "oracle_tx_gap_mean": float(np.mean(result["oracle_tx_gap_mean"])),
                "observed_score_mean": float(np.mean(result["observed_score_mean"])),
                "observed_tx_fraction_mean": float(np.mean(result["observed_tx_fraction_mean"])),
                "avg_reward": float(np.mean(result["episode_reward"])),
            }
        )
    return rows


def write_csv(path, rows):
    """Write aggregate-feedback summary CSV."""
    ensure_parent_dir(path)
    fieldnames = [
        "feedback_noise_std",
        "probe_budget",
        "budget_fraction",
        "policy",
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
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def policy_label(row):
    """Return a compact label for plots and summaries."""
    policy = row["policy"]
    if policy in PROBE_POLICIES:
        return f"{policy} B={int(row['probe_budget'])}"
    return policy


def print_summary(rows):
    """Print a compact aggregate-feedback summary."""
    print("=" * 146)
    print("Bandit-Feedback MS-AirComp Summary")
    print("=" * 146)
    print(
        f"{'Noise':>6} {'B':>3} {'Policy':<30} {'Success':>9} {'Perfect%':>9} "
        f"{'Slots':>8} {'Energy':>10} {'Probes':>8} {'Oracle%':>9} {'Gap':>7} "
        f"{'ObsTx':>7}"
    )
    for row in rows:
        print(
            f"{row['feedback_noise_std']:>6.3f} {row['probe_budget']:>3} "
            f"{row['policy']:<30} {row['success_mean']:>9.3f} "
            f"{row['perfect_rate']:>8.2f}% {row['slots_mean']:>8.3f} "
            f"{row['total_energy_mean']:>10.3f} {row['probe_calls_per_slot_mean']:>8.2f} "
            f"{row['oracle_match_rate']:>8.2f}% {row['oracle_tx_gap_mean']:>7.3f} "
            f"{row['observed_tx_fraction_mean']:>7.3f}"
        )


def plot_results(rows, args, output_prefix):
    """Plot success, perfect coverage, latency, and oracle gap vs feedback noise."""
    labels = []
    for row in rows:
        label = policy_label(row)
        if label not in labels:
            labels.append(label)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(idx % 20) for idx, label in enumerate(labels)}

    for label in labels:
        label_rows = sorted(
            [row for row in rows if policy_label(row) == label],
            key=lambda item: item["feedback_noise_std"],
        )
        x = [row["feedback_noise_std"] for row in label_rows]
        axes[0, 0].plot(
            x,
            [row["success_mean"] for row in label_rows],
            marker="o",
            linewidth=1.6,
            label=label,
            color=colors[label],
        )
        axes[0, 1].plot(
            x,
            [row["perfect_rate"] for row in label_rows],
            marker="o",
            linewidth=1.6,
            label=label,
            color=colors[label],
        )
        axes[1, 0].plot(
            x,
            [row["slots_mean"] for row in label_rows],
            marker="o",
            linewidth=1.6,
            label=label,
            color=colors[label],
        )
        axes[1, 1].plot(
            x,
            [row["oracle_tx_gap_mean"] for row in label_rows],
            marker="o",
            linewidth=1.6,
            label=label,
            color=colors[label],
        )

    axes[0, 0].set_title("Success vs Aggregate Feedback Noise")
    axes[0, 0].set_ylabel("Successful Nodes")
    axes[0, 0].set_ylim(0.0, args.num_nodes + 1)
    axes[0, 1].set_title("Perfect Coverage vs Aggregate Feedback Noise")
    axes[0, 1].set_ylabel("Perfect Episodes (%)")
    axes[0, 1].set_ylim(0.0, 103.0)
    axes[1, 0].set_title("Latency vs Aggregate Feedback Noise")
    axes[1, 0].set_ylabel("Slots Used")
    axes[1, 0].set_ylim(0.0, args.num_slots + 1)
    axes[1, 1].set_title("Per-Slot Oracle Tx Gap")
    axes[1, 1].set_ylabel("Missed Tx Count")

    for ax in axes.ravel():
        ax.set_xlabel("Noisy Aggregate Feedback Std")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)

    fig.tight_layout()
    path = f"{output_prefix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    """Run bandit-feedback MS-AirComp evaluation."""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)
    base_action = make_base_action(args)
    run_seeds = make_run_seeds(args)
    episode_seed_sets = [make_episode_seeds(args, run_seed) for run_seed in run_seeds]

    print("=" * 96)
    print(
        f"Bandit-feedback MS-AirComp: episodes={args.episodes}, num_seeds={args.num_seeds}, "
        f"feedback_noise={args.feedback_noise_std_values}, budgets={args.probe_budgets}"
    )
    print(
        f"Fixed transmission parameters: g_th={args.g_th}, alpha_th={args.alpha_th}; "
        f"feedback_power_weight={args.feedback_power_weight}, output_prefix={output_prefix}"
    )
    print("=" * 96)

    rows = []
    for feedback_noise_std in args.feedback_noise_std_values:
        print("=" * 96)
        print(f"Aggregate feedback noise std={feedback_noise_std:g}")
        print("=" * 96)
        seed_result_sets = []
        for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
            print(f"Run seed [{run_idx}/{len(run_seeds)}]: {run_seeds[run_idx - 1]}")
            seed_result_sets.append(
                policy_suite_for_noise(args, episode_seeds, feedback_noise_std, base_action)
            )
        rows.extend(summarize_results(args, aggregate_seed_results(seed_result_sets)))

    print_summary(rows)
    write_csv(f"{output_prefix}.csv", rows)
    if not args.no_plots:
        plot_results(rows, args, output_prefix)


if __name__ == "__main__":
    main()
