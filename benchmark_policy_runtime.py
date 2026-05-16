"""
Runtime benchmark for MS-AirComp IRS policies.

The benchmark separates policy decision time from environment step time:

- decision time: choosing an action or IRS codebook from the current observation;
- env step time: applying the action and producing the next observation.

For feature-based policies, env step time includes the current implementation cost
of generating the next observation's codebook features. The
decision_preview_calls_per_slot metric follows the project convention: it counts
extra preview calls made during policy decision after codebook features are
already available.
"""

import argparse
import csv
import os
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from ms_aircomp.experiment_utils import (
    codebook_index_to_action,
    ensure_parent_dir,
    make_episode_seed_list,
    physical_to_action,
)
from test_env import MSAirCompEnv
from train_codebook_aware_agent import FixedTransmissionActionWrapper


POLICY_FEATURE_ARGMAX = "Feature Argmax IRS"
POLICY_FEATURE_ARGMAX_POWER_TIE = "Feature Argmax PowerTie IRS"
POLICY_GREEDY = "Greedy IRS"
POLICY_SAC = "SAC"
POLICY_CODEBOOK_AWARE = "Codebook-Aware SAC"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark policy decision runtime.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--model-dir", default="./rl_models")
    parser.add_argument("--sac-model-name", default="sac_final_model_v3.zip")
    parser.add_argument("--sac-stats-name", default="vec_normalize.pkl")
    parser.add_argument("--codebook-aware-model-name", default="sac_codebook_aware_irs_selector.zip")
    parser.add_argument("--codebook-aware-stats-name", default="vec_normalize_codebook_aware_irs_selector.pkl")
    parser.add_argument("--skip-sac", action="store_true")
    parser.add_argument("--skip-codebook-aware-sac", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def validate_args(args):
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    for name in ("num_nodes", "num_slots", "num_irs_elements", "num_codebook_states"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")


def resolve_output(args):
    if args.output is None:
        args.output = os.path.join(
            "results",
            "runtime",
            f"runtime_benchmark_ep{args.episodes}_seed{args.seed}.csv",
        )
    ensure_parent_dir(args.output)


def make_base_action(args, irs_index=0):
    return np.array(
        [
            physical_to_action(args.g_th, low=0.001, scale=0.05),
            physical_to_action(args.alpha_th, low=0.05, scale=0.05),
            codebook_index_to_action(irs_index, args.num_codebook_states),
        ],
        dtype=np.float32,
    )


def make_env(args, irs_phase_mode="codebook", include_codebook_features=False):
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode=irs_phase_mode,
        include_codebook_features=include_codebook_features,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
    )


def feature_argmax_candidates(obs, args):
    features = obs[7 : 7 + args.num_codebook_states]
    max_feature = float(np.max(features))
    return np.flatnonzero(np.isclose(features, max_feature, rtol=0.0, atol=1e-7))


def feature_argmax_power_tie_index(env, args, obs):
    candidate_indices = feature_argmax_candidates(obs, args)
    tie_count = int(len(candidate_indices))
    if tie_count <= 1:
        return int(candidate_indices[0]), 0, tie_count

    candidates = [
        env.preview_codebook_index(codebook_index, args.g_th, args.alpha_th)
        for codebook_index in candidate_indices
    ]

    def candidate_key(candidate):
        tx_count = int(candidate["tx_this_slot"])
        power_avg = float(candidate["power_avg"])
        mean_gain = float(candidate["mean_gain_remaining"])
        power_tiebreak = -power_avg if tx_count > 0 else 0.0
        return tx_count, power_tiebreak, mean_gain

    best_candidate = max(candidates, key=candidate_key)
    return int(best_candidate["irs_index"]), tie_count, tie_count


def greedy_index(env, args):
    candidates = [
        env.preview_codebook_index(codebook_index, args.g_th, args.alpha_th)
        for codebook_index in range(args.num_codebook_states)
    ]

    def candidate_key(candidate):
        tx_count = int(candidate["tx_this_slot"])
        power_avg = float(candidate["power_avg"])
        mean_gain = float(candidate["mean_gain_remaining"])
        power_tiebreak = -power_avg if tx_count > 0 else 0.0
        return tx_count, power_tiebreak, mean_gain

    best_candidate = max(candidates, key=candidate_key)
    return int(best_candidate["irs_index"])


def update_energy(episode_energy, info):
    return episode_energy + float(info["power_avg"]) * int(info["tx_this_slot"])


def append_episode_metrics(success_nodes, slots_used, total_energy, total_tx, episode_slots, episode_energy):
    success_nodes.append(int(total_tx))
    slots_used.append(int(episode_slots))
    total_energy.append(float(episode_energy))


def ns_to_ms(values):
    return np.asarray(values, dtype=float) / 1_000_000.0


def summarize_times(values_ns):
    values_ms = ns_to_ms(values_ns)
    if len(values_ms) == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(values_ms)),
        float(np.percentile(values_ms, 50)),
        float(np.percentile(values_ms, 95)),
    )


def finalize_summary(
    policy,
    args,
    success_nodes,
    slots_used,
    total_energy,
    decision_times_ns,
    env_step_times_ns,
    episode_wall_times_ns,
    preview_calls_per_slot,
    tie_candidates_mean,
):
    decision_mean, decision_p50, decision_p95 = summarize_times(decision_times_ns)
    step_mean, step_p50, step_p95 = summarize_times(env_step_times_ns)
    wall_mean, wall_p50, wall_p95 = summarize_times(episode_wall_times_ns)
    success = np.asarray(success_nodes, dtype=float)
    slots = np.asarray(slots_used, dtype=float)
    energies = np.asarray(total_energy, dtype=float)
    return {
        "policy": policy,
        "episodes": len(success_nodes),
        "decisions": len(decision_times_ns),
        "success_mean": float(np.mean(success)) if len(success) else 0.0,
        "perfect_rate": float(np.mean(success == args.num_nodes) * 100.0) if len(success) else 0.0,
        "slots_mean": float(np.mean(slots)) if len(slots) else 0.0,
        "total_energy_mean": float(np.mean(energies)) if len(energies) else 0.0,
        "decision_time_mean_ms": decision_mean,
        "decision_time_p50_ms": decision_p50,
        "decision_time_p95_ms": decision_p95,
        "env_step_time_mean_ms": step_mean,
        "env_step_time_p50_ms": step_p50,
        "env_step_time_p95_ms": step_p95,
        "episode_wall_time_mean_ms": wall_mean,
        "episode_wall_time_p50_ms": wall_p50,
        "episode_wall_time_p95_ms": wall_p95,
        "decision_preview_calls_per_slot_mean": float(np.mean(preview_calls_per_slot))
        if preview_calls_per_slot
        else 0.0,
        "tie_candidates_mean": float(np.mean(tie_candidates_mean)) if tie_candidates_mean else 0.0,
    }


def benchmark_rule_policy(policy, episode_seeds, args):
    include_features = policy in {POLICY_FEATURE_ARGMAX, POLICY_FEATURE_ARGMAX_POWER_TIE}
    env = make_env(args, include_codebook_features=include_features)
    base_action = make_base_action(args)
    success_nodes = []
    slots_used = []
    total_energy = []
    decision_times_ns = []
    env_step_times_ns = []
    episode_wall_times_ns = []
    preview_calls_per_slot = []
    tie_candidates_mean = []

    print(f"Benchmarking {policy}...")
    for episode_seed in episode_seeds:
        episode_start = time.perf_counter_ns()
        obs, _info = env.reset(seed=episode_seed)
        episode_energy = 0.0
        episode_slots = args.num_slots
        episode_preview_calls = 0
        episode_tie_candidates = []
        total_tx = 0

        for slot in range(1, args.num_slots + 1):
            decision_start = time.perf_counter_ns()
            if policy == POLICY_FEATURE_ARGMAX:
                candidates = feature_argmax_candidates(obs, args)
                irs_index = int(candidates[0])
                preview_calls = 0
                tie_count = int(len(candidates))
            elif policy == POLICY_FEATURE_ARGMAX_POWER_TIE:
                irs_index, preview_calls, tie_count = feature_argmax_power_tie_index(env, args, obs)
            else:
                irs_index = greedy_index(env, args)
                preview_calls = args.num_codebook_states
                tie_count = args.num_codebook_states
            decision_times_ns.append(time.perf_counter_ns() - decision_start)

            action = base_action.copy()
            action[2] = codebook_index_to_action(irs_index, args.num_codebook_states)

            step_start = time.perf_counter_ns()
            obs, _reward, terminated, truncated, info = env.step(action)
            env_step_times_ns.append(time.perf_counter_ns() - step_start)

            total_tx = int(info["total_tx"])
            episode_slots = int(info.get("slots_used", slot))
            episode_energy = update_energy(episode_energy, info)
            episode_preview_calls += int(preview_calls)
            episode_tie_candidates.append(int(tie_count))

            if terminated or truncated:
                break

        active_slots = max(len(episode_tie_candidates), 1)
        preview_calls_per_slot.append(float(episode_preview_calls) / active_slots)
        tie_candidates_mean.append(float(np.mean(episode_tie_candidates)) if episode_tie_candidates else 0.0)
        append_episode_metrics(success_nodes, slots_used, total_energy, total_tx, episode_slots, episode_energy)
        episode_wall_times_ns.append(time.perf_counter_ns() - episode_start)

    return finalize_summary(
        policy,
        args,
        success_nodes,
        slots_used,
        total_energy,
        decision_times_ns,
        env_step_times_ns,
        episode_wall_times_ns,
        preview_calls_per_slot,
        tie_candidates_mean,
    )


def benchmark_sac_policy(episode_seeds, args):
    model_path = os.path.join(args.model_dir, args.sac_model_name)
    stats_path = os.path.join(args.model_dir, args.sac_stats_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAC model not found: {model_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"SAC VecNormalize stats not found: {stats_path}")

    raw_venv = DummyVecEnv([lambda: make_env(args, "codebook", include_codebook_features=False)])
    stacked_venv = VecFrameStack(raw_venv, n_stack=4)
    venv = VecNormalize.load(stats_path, stacked_venv)
    venv.training = False
    venv.norm_reward = False
    model = SAC.load(model_path, env=venv)
    try:
        return benchmark_vec_model_policy(POLICY_SAC, model, venv, episode_seeds, args)
    finally:
        venv.close()


def benchmark_codebook_aware_sac_policy(episode_seeds, args):
    model_path = os.path.join(args.model_dir, args.codebook_aware_model_name)
    stats_path = os.path.join(args.model_dir, args.codebook_aware_stats_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Codebook-Aware SAC model not found: {model_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Codebook-Aware SAC VecNormalize stats not found: {stats_path}")

    def make_wrapped_env():
        env = make_env(args, "codebook", include_codebook_features=True)
        return FixedTransmissionActionWrapper(env, g_th=args.g_th, alpha_th=args.alpha_th)

    raw_venv = DummyVecEnv([make_wrapped_env])
    stacked_venv = VecFrameStack(raw_venv, n_stack=4)
    venv = VecNormalize.load(stats_path, stacked_venv)
    venv.training = False
    venv.norm_reward = False
    model = SAC.load(model_path, env=venv)
    try:
        return benchmark_vec_model_policy(POLICY_CODEBOOK_AWARE, model, venv, episode_seeds, args)
    finally:
        venv.close()


def benchmark_vec_model_policy(policy, model, venv, episode_seeds, args):
    success_nodes = []
    slots_used = []
    total_energy = []
    decision_times_ns = []
    env_step_times_ns = []
    episode_wall_times_ns = []
    preview_calls_per_slot = []
    tie_candidates_mean = []

    print(f"Benchmarking {policy}...")
    for episode_seed in episode_seeds:
        episode_start = time.perf_counter_ns()
        venv.seed(int(episode_seed))
        obs = venv.reset()
        episode_energy = 0.0
        episode_slots = args.num_slots
        total_tx = 0
        active_slots = 0

        for slot in range(1, args.num_slots + 1):
            decision_start = time.perf_counter_ns()
            action, _states = model.predict(obs, deterministic=True)
            decision_times_ns.append(time.perf_counter_ns() - decision_start)

            step_start = time.perf_counter_ns()
            obs, _reward, done, info_list = venv.step(action)
            env_step_times_ns.append(time.perf_counter_ns() - step_start)

            info = info_list[0]
            total_tx = int(info["total_tx"])
            episode_slots = int(info.get("slots_used", slot))
            episode_energy = update_energy(episode_energy, info)
            active_slots += 1

            if done[0]:
                break

        preview_calls_per_slot.append(0.0)
        tie_candidates_mean.append(0.0)
        append_episode_metrics(success_nodes, slots_used, total_energy, total_tx, episode_slots, episode_energy)
        episode_wall_times_ns.append(time.perf_counter_ns() - episode_start)

    return finalize_summary(
        policy,
        args,
        success_nodes,
        slots_used,
        total_energy,
        decision_times_ns,
        env_step_times_ns,
        episode_wall_times_ns,
        preview_calls_per_slot,
        tie_candidates_mean,
    )


def write_csv(path, rows):
    fieldnames = [
        "policy",
        "episodes",
        "decisions",
        "success_mean",
        "perfect_rate",
        "slots_mean",
        "total_energy_mean",
        "decision_time_mean_ms",
        "decision_time_p50_ms",
        "decision_time_p95_ms",
        "env_step_time_mean_ms",
        "env_step_time_p50_ms",
        "env_step_time_p95_ms",
        "episode_wall_time_mean_ms",
        "episode_wall_time_p50_ms",
        "episode_wall_time_p95_ms",
        "decision_preview_calls_per_slot_mean",
        "tie_candidates_mean",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def print_summary(rows):
    print("=" * 128)
    print("Runtime Benchmark Summary")
    print("=" * 128)
    print(
        f"{'Policy':<30} {'Success':>8} {'Slots':>7} {'Energy':>10} "
        f"{'Dec ms':>9} {'P95 ms':>9} {'Step ms':>9} {'Prev':>7} {'Tie':>7}"
    )
    for row in rows:
        print(
            f"{row['policy']:<30} {row['success_mean']:>8.2f} {row['slots_mean']:>7.2f} "
            f"{row['total_energy_mean']:>10.3f} {row['decision_time_mean_ms']:>9.4f} "
            f"{row['decision_time_p95_ms']:>9.4f} {row['env_step_time_mean_ms']:>9.4f} "
            f"{row['decision_preview_calls_per_slot_mean']:>7.2f} {row['tie_candidates_mean']:>7.2f}"
        )


def main():
    args = parse_args()
    validate_args(args)
    resolve_output(args)
    episode_seeds = make_episode_seed_list(args.seed, args.episodes)

    rows = [
        benchmark_rule_policy(POLICY_FEATURE_ARGMAX, episode_seeds, args),
        benchmark_rule_policy(POLICY_FEATURE_ARGMAX_POWER_TIE, episode_seeds, args),
        benchmark_rule_policy(POLICY_GREEDY, episode_seeds, args),
    ]
    if not args.skip_sac:
        rows.append(benchmark_sac_policy(episode_seeds, args))
    if not args.skip_codebook_aware_sac:
        rows.append(benchmark_codebook_aware_sac_policy(episode_seeds, args))

    print_summary(rows)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
