"""
Train and evaluate a temporal deviation selector for stale-CSI probing.

The model learns an offset relative to the rotating-grid probe window. Training
uses hidden current-channel outcomes as supervised targets, but evaluation only
uses observable episode state and history from previous probes/executions.
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import evaluate_execution_channel_mismatch as mismatch
import evaluate_limited_csi_ms_aircomp as limited
from evaluate_policy_comparison import (
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
)
from ms_aircomp.channel_models import (
    apply_channel_state,
    build_temporal_channel_states,
    capture_channel_state,
    delayed_channel_state,
)
from ms_aircomp.execution_candidates import (
    execution_candidate_for_decision,
    execution_oracle_candidate,
)


POLICY_LEARNED_TEMPORAL_DEVIATION = "Learned Temporal Deviation"
POLICY_WINDOW_TEMPORAL_DEVIATION = "Window Temporal Deviation"
POLICY_GATED_TEMPORAL_DEVIATION = "Gated Temporal Deviation"
POLICY_GATED_WINDOW_TEMPORAL_DEVIATION = "Gated Window Temporal Deviation"
POLICY_DAGGER_TEMPORAL_DEVIATION = "DAgger Temporal Deviation"
POLICY_DAGGER_WINDOW_TEMPORAL_DEVIATION = "DAgger Window Temporal Deviation"
POLICY_DAGGER_GATED_TEMPORAL_DEVIATION = "DAgger Gated Temporal Deviation"
POLICY_DAGGER_GATED_WINDOW_TEMPORAL_DEVIATION = "DAgger Gated Window Temporal Deviation"
LEARNED_TEMPORAL_OFFSET = 0xA3C59AC3


def parse_args():
    """Parse temporal-deviation training and evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Train a learned temporal deviation selector for limited-CSI probing."
    )
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--val-episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-eval-seeds", type=int, default=3)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--probe-budgets", default="4")
    parser.add_argument("--channel-rho-values", default="0.7,0.9,0.98")
    parser.add_argument("--csi-delay-slots", default="1,2,3")
    parser.add_argument("--offsets", default="-3,-2,-1,0,1,2,3")
    parser.add_argument("--feature-mode", choices=["global", "window"], default="global")
    parser.add_argument("--gate-margin-thresholds", default="0")
    parser.add_argument("--behavior-random-prob", type=float, default=0.7)
    parser.add_argument("--dagger-iterations", type=int, default=0)
    parser.add_argument("--dagger-episodes", type=int, default=0)
    parser.add_argument("--dagger-beta-start", type=float, default=0.5)
    parser.add_argument("--dagger-beta-end", type=float, default=0.0)
    parser.add_argument("--decision-error-std", type=float, default=0.0)
    parser.add_argument("--execution-error-std", type=float, default=0.0)
    parser.add_argument("--target-power-weight", type=float, default=0.02)
    parser.add_argument("--target-failure-weight", type=float, default=0.20)
    parser.add_argument("--history-lr", type=float, default=0.60)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=5)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--fixed-irs-index", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """Validate arguments and parse list-like values."""
    for name in (
        "train_episodes",
        "val_episodes",
        "eval_episodes",
        "num_eval_seeds",
        "num_nodes",
        "num_slots",
        "num_irs_elements",
        "num_codebook_states",
        "hidden_size",
        "hidden_layers",
        "epochs",
        "batch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")
    if not 0.0 <= args.behavior_random_prob <= 1.0:
        raise ValueError("--behavior-random-prob must be in [0, 1]")
    if args.dagger_iterations < 0:
        raise ValueError("--dagger-iterations must be non-negative")
    if args.dagger_iterations > 0 and args.dagger_episodes <= 0:
        raise ValueError("--dagger-episodes must be positive when --dagger-iterations is enabled")
    if args.dagger_episodes < 0:
        raise ValueError("--dagger-episodes must be non-negative")
    if not 0.0 <= args.dagger_beta_start <= 1.0 or not 0.0 <= args.dagger_beta_end <= 1.0:
        raise ValueError("--dagger-beta-start and --dagger-beta-end must be in [0, 1]")
    if not 0.0 <= args.history_lr <= 1.0:
        raise ValueError("--history-lr must be in [0, 1]")
    if args.decision_error_std < 0.0 or args.execution_error_std < 0.0:
        raise ValueError("error std values must be non-negative")
    if args.target_power_weight < 0.0 or args.target_failure_weight < 0.0:
        raise ValueError("target weights must be non-negative")
    if args.g_th <= 0.0 or args.alpha_th <= 0.0:
        raise ValueError("--g-th and --alpha-th must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")

    args.probe_budgets = sorted(
        {min(int(value), args.num_codebook_states) for value in limited.parse_int_list(args.probe_budgets)}
    )
    args.channel_rho_values = limited.parse_float_list(args.channel_rho_values)
    args.csi_delay_slots = limited.parse_int_list(args.csi_delay_slots)
    args.offsets = limited.parse_int_list(args.offsets)
    args.gate_margin_thresholds = limited.parse_float_list(args.gate_margin_thresholds)
    if not args.probe_budgets:
        raise ValueError("--probe-budgets must not be empty")
    if not args.channel_rho_values or any(value < 0.0 or value > 1.0 for value in args.channel_rho_values):
        raise ValueError("--channel-rho-values must be non-empty and in [0, 1]")
    if not args.csi_delay_slots or any(value < 0 for value in args.csi_delay_slots):
        raise ValueError("--csi-delay-slots must be non-empty and non-negative")
    if not args.offsets:
        raise ValueError("--offsets must not be empty")
    if not args.gate_margin_thresholds or any(value < 0.0 for value in args.gate_margin_thresholds):
        raise ValueError("--gate-margin-thresholds must be non-empty and non-negative")
    if any(value > 0.0 for value in args.gate_margin_thresholds) and 0 not in args.offsets:
        raise ValueError("--offsets must include 0 when positive gate margins are enabled")
    args.fixed_irs_index = int(np.clip(args.fixed_irs_index, 0, args.num_codebook_states - 1))


def build_eval_args(args, episodes=None, num_seeds=None):
    """Build the namespace expected by execution-mismatch helpers."""
    return argparse.Namespace(
        episodes=args.eval_episodes if episodes is None else int(episodes),
        seed=args.seed,
        num_seeds=args.num_eval_seeds if num_seeds is None else int(num_seeds),
        seed_stride=args.seed_stride,
        probe_budgets=list(args.probe_budgets),
        mismatch_models=[mismatch.MISMATCH_TEMPORAL_AR1],
        channel_rho_values=list(args.channel_rho_values),
        csi_delay_slots=list(args.csi_delay_slots),
        decision_error_std_values=[float(args.decision_error_std)],
        execution_error_std_values=[float(args.execution_error_std)],
        policies=[],
        robust_gain_margins=[1.25],
        robust_power_margins=[0.9],
        risk_weights=[0.5],
        risk_power_weights=[0.1],
        risk_invite_thresholds=[0.5],
        adaptive_risk_base_weights=[0.5],
        adaptive_risk_error_ref=0.3,
        adaptive_risk_error_gain=1.0,
        adaptive_risk_deadline_relief=0.6,
        adaptive_risk_backlog_relief=0.8,
        opportunity_failure_costs=[1.0],
        opportunity_missed_costs=[1.0],
        opportunity_deadline_gains=[0.5],
        opportunity_backlog_gains=[0.5],
        temporal_reliability_z_values=[1.0],
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        g_th=args.g_th,
        alpha_th=args.alpha_th,
        fixed_irs_index=args.fixed_irs_index,
        output_prefix=None,
        no_plots=args.no_plots,
        offsets=list(args.offsets),
        decision_error_std=float(args.decision_error_std),
        execution_error_std=float(args.execution_error_std),
        target_power_weight=float(args.target_power_weight),
        target_failure_weight=float(args.target_failure_weight),
        history_lr=float(args.history_lr),
        feature_mode=args.feature_mode,
        gate_margin_thresholds=list(args.gate_margin_thresholds),
        dagger_iterations=int(getattr(args, "dagger_iterations", 0)),
        learned_policy_name=learned_policy_name(args),
        P_max=1.0,
    )


def resolve_output_prefix(args):
    """Resolve output prefix for model, CSV, and diagnostics."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    rho_label = "-".join(format_float_for_suffix(value) for value in args.channel_rho_values)
    delay_label = "-".join(str(value) for value in args.csi_delay_slots)
    budget_label = "-".join(str(value) for value in args.probe_budgets)
    suffix = (
        f"train{args.train_episodes}_eval{args.eval_episodes}_runs{args.num_eval_seeds}_"
        f"rho{rho_label}_delay{delay_label}_b{budget_label}_"
        f"offsets{len(args.offsets)}"
    )
    if args.feature_mode != "global":
        suffix += f"_{args.feature_mode}feat"
    if any(value > 0.0 for value in args.gate_margin_thresholds):
        gate_label = "-".join(format_float_for_suffix(value) for value in args.gate_margin_thresholds)
        suffix += f"_gatemargin{gate_label}"
    if args.dagger_iterations > 0:
        suffix += (
            f"_dagger{args.dagger_iterations}x{args.dagger_episodes}_"
            f"beta{format_float_for_suffix(args.dagger_beta_start)}-"
            f"{format_float_for_suffix(args.dagger_beta_end)}"
        )
    output_prefix = os.path.join(
        "results",
        "execution_mismatch",
        f"learned_temporal_deviation_{suffix}",
    )
    ensure_parent_dir(output_prefix)
    return output_prefix


def learned_rng(episode_seed, salt=0):
    """Create deterministic RNG streams for learned temporal deviation."""
    if episode_seed is None:
        return np.random.default_rng()
    seed = (int(episode_seed) + LEARNED_TEMPORAL_OFFSET + int(salt) * 0x165667B1) % (2**32)
    return np.random.default_rng(seed)


def split_episode_specs(seed, episodes, rhos, delays):
    """Generate train/validation episode seeds and sampled temporal scenarios."""
    rng = np.random.default_rng(seed)
    specs = []
    for episode_seed in rng.integers(0, 2**31 - 1, size=episodes):
        specs.append(
            (
                int(episode_seed),
                float(rng.choice(rhos)),
                int(rng.choice(delays)),
            )
        )
    return specs


def set_training_seed(args):
    """Seed torch-side model initialization and DataLoader shuffling."""
    if int(args.seed) >= 0:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))


def print_progress(name, current, total):
    """Print progress at 10 percent intervals."""
    interval = max(total // 10, 1)
    if current % interval == 0 or current == total:
        print(f"  {name}: [{current:04d}/{total:04d}]")


def initialize_history(args):
    """Initialize observable probe/execution history."""
    c_count = int(args.num_codebook_states)
    return {
        "counts": np.zeros(c_count, dtype=float),
        "decision_tx": np.zeros(c_count, dtype=float),
        "decision_power": np.zeros(c_count, dtype=float),
        "success": np.zeros(c_count, dtype=float),
        "failed": np.zeros(c_count, dtype=float),
        "missed": np.zeros(c_count, dtype=float),
        "age": np.full(c_count, float(args.num_slots + 1), dtype=float),
        "last_selected": -1,
        "last_offset": 0.0,
        "last_success": 0.0,
        "last_failed": 0.0,
        "last_missed": 0.0,
        "last_decision_tx": 0.0,
    }


def offset_indices(args, budget, slot_idx, offset):
    """Return the IRS probe indices for one relative rotating offset."""
    return limited.grid_indices(args.num_codebook_states, budget, offset=int(slot_idx) + int(offset))


def history_features(env, history, args, channel_rho, csi_delay_slots, budget, slot_idx):
    """Build a feature vector from observable state and past probe history."""
    c_count = int(args.num_codebook_states)
    remaining_count = int(np.sum(~env.transmitted_flags))
    slots_left = max(int(args.num_slots) - int(slot_idx), 1)
    slot_progress = float(slot_idx) / float(max(args.num_slots - 1, 1))
    remaining_ratio = float(remaining_count) / float(max(args.num_nodes, 1))
    pressure = remaining_ratio / max(float(slots_left) / float(max(args.num_slots, 1)), 1e-12)
    rotating_phase = 2.0 * np.pi * float(slot_idx % c_count) / float(max(c_count, 1))
    if int(history["last_selected"]) >= 0:
        selected = np.zeros(c_count, dtype=float)
        selected[int(history["last_selected"])] = 1.0
    else:
        selected = np.zeros(c_count, dtype=float)

    scalars = np.asarray(
        [
            slot_progress,
            remaining_ratio,
            float(slots_left) / float(max(args.num_slots, 1)),
            min(float(pressure), 4.0) / 4.0,
            float(channel_rho),
            float(csi_delay_slots) / float(max(args.num_slots, 1)),
            float(budget) / float(max(c_count, 1)),
            float(np.sin(rotating_phase)),
            float(np.cos(rotating_phase)),
            float(history["last_offset"]) / float(max(c_count, 1)),
            float(history["last_success"]),
            float(history["last_failed"]),
            float(history["last_missed"]),
            float(history["last_decision_tx"]),
        ],
        dtype=np.float32,
    )
    counts = np.clip(history["counts"] / max(args.num_slots, 1), 0.0, 1.0)
    recency = 1.0 / (1.0 + np.asarray(history["age"], dtype=float))
    return np.concatenate(
        [
            scalars,
            counts.astype(np.float32),
            np.asarray(history["decision_tx"], dtype=np.float32),
            np.asarray(history["decision_power"], dtype=np.float32),
            np.asarray(history["success"], dtype=np.float32),
            np.asarray(history["failed"], dtype=np.float32),
            np.asarray(history["missed"], dtype=np.float32),
            recency.astype(np.float32),
            selected.astype(np.float32),
        ]
    ).astype(np.float32)


def offset_window_features(env, history, args, channel_rho, csi_delay_slots, budget, slot_idx, offsets):
    """Build per-offset candidate-window features from observable history."""
    c_count = int(args.num_codebook_states)
    base = history_features(env, history, args, channel_rho, csi_delay_slots, budget, slot_idx)[:14]
    counts = np.clip(np.asarray(history["counts"], dtype=float) / max(args.num_slots, 1), 0.0, 1.0)
    recency = 1.0 / (1.0 + np.asarray(history["age"], dtype=float))
    arrays = [
        np.asarray(history["decision_tx"], dtype=float),
        np.asarray(history["decision_power"], dtype=float),
        np.asarray(history["success"], dtype=float),
        np.asarray(history["failed"], dtype=float),
        np.asarray(history["missed"], dtype=float),
        counts,
        recency,
    ]
    rows = []
    last_selected = int(history["last_selected"])
    for offset in offsets:
        indices = offset_indices(args, budget, slot_idx, offset)
        center = float(np.mean(indices)) / float(max(c_count - 1, 1))
        phase = 2.0 * np.pi * center
        row = [
            *base.tolist(),
            float(offset) / float(max(c_count, 1)),
            abs(float(offset)) / float(max(c_count, 1)),
            float(center),
            float(np.sin(phase)),
            float(np.cos(phase)),
            float(last_selected in set(int(index) for index in indices)) if last_selected >= 0 else 0.0,
        ]
        for values in arrays:
            window_values = np.asarray(values, dtype=float)[indices]
            row.extend(
                [
                    float(np.mean(window_values)),
                    float(np.max(window_values)),
                    float(np.min(window_values)),
                    float(np.std(window_values)),
                ]
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def policy_features(env, history, args, channel_rho, csi_delay_slots, budget, slot_idx):
    """Build policy features for the selected model architecture."""
    if getattr(args, "feature_mode", "global") == "window":
        return offset_window_features(
            env,
            history,
            args,
            channel_rho,
            csi_delay_slots,
            budget,
            slot_idx,
            args.offsets,
        )
    return history_features(env, history, args, channel_rho, csi_delay_slots, budget, slot_idx)


def update_history(history, args, indices, decision_candidates, selected_decision, info, chosen_offset):
    """Update probe/execution history after one slot."""
    history["age"] += 1.0
    lr = float(getattr(args, "history_lr", 0.60))
    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in decision_candidates}
    for index in indices:
        candidate = candidate_by_index[int(index)]
        old_count = float(history["counts"][index])
        tx_fraction = float(candidate["tx_this_slot"]) / float(max(args.num_nodes, 1))
        power_norm = float(candidate["power_avg"]) / max(float(getattr(args, "P_max", 1.0)), 1e-12)
        if old_count <= 0.0:
            history["decision_tx"][index] = tx_fraction
            history["decision_power"][index] = power_norm
        else:
            history["decision_tx"][index] = (1.0 - lr) * history["decision_tx"][index] + lr * tx_fraction
            history["decision_power"][index] = (
                (1.0 - lr) * history["decision_power"][index] + lr * power_norm
            )
        history["counts"][index] = old_count + 1.0
        history["age"][index] = 0.0

    selected_index = int(selected_decision["irs_index"])
    success_fraction = float(info["tx_this_slot"]) / float(max(args.num_nodes, 1))
    failed_fraction = float(info["failed_this_slot"]) / float(max(args.num_nodes, 1))
    missed_fraction = float(info["missed_opportunity_this_slot"]) / float(max(args.num_nodes, 1))
    for key, value in (
        ("success", success_fraction),
        ("failed", failed_fraction),
        ("missed", missed_fraction),
    ):
        if history["counts"][selected_index] <= 1.0:
            history[key][selected_index] = value
        else:
            history[key][selected_index] = (1.0 - lr) * history[key][selected_index] + lr * value

    history["last_selected"] = selected_index
    history["last_offset"] = float(chosen_offset)
    history["last_success"] = success_fraction
    history["last_failed"] = failed_fraction
    history["last_missed"] = missed_fraction
    history["last_decision_tx"] = float(selected_decision["tx_this_slot"]) / float(max(args.num_nodes, 1))


def mask_metrics(env, args, decision, true_candidate):
    """Compute execution metrics for a decision without mutating the environment."""
    remaining = ~env.transmitted_flags
    scheduled = np.asarray(decision["valid_mask"], dtype=bool) & remaining
    true_valid = np.asarray(true_candidate["valid_mask"], dtype=bool) & remaining
    success = int(np.sum(scheduled & true_valid))
    failed = int(np.sum(scheduled & (~true_valid)))
    scheduled_power = np.asarray(decision["required_power"], dtype=float)[scheduled]
    power_avg = float(np.mean(scheduled_power)) if scheduled_power.size else 0.0
    return success, failed, power_avg


def estimated_window_decision(env, args, indices, decision_error_std, episode_seed, slot_idx, salt):
    """Build stale/estimated candidates for a window and return the selected decision."""
    rng = limited.stable_rng(
        episode_seed,
        decision_error_std,
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        len(indices),
        salt=salt + slot_idx,
    )
    candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=indices,
        error_std=decision_error_std,
        rng=rng,
    )
    return limited.best_candidate(candidates), candidates


def offset_target_scores(
    env,
    args,
    budget,
    slot_idx,
    channel_rho,
    csi_delay_slots,
    decision_error_std,
    execution_error_std,
    episode_seed,
    decision_state,
    execution_state,
    offsets,
):
    """Return hidden supervised utility scores for each candidate offset."""
    snapshot = capture_channel_state(env)
    scores = []
    tx_values = []
    for row_idx, offset in enumerate(offsets):
        indices = offset_indices(args, budget, slot_idx, offset)
        apply_channel_state(env, decision_state)
        decision, _candidates = estimated_window_decision(
            env,
            args,
            indices,
            decision_error_std,
            episode_seed,
            slot_idx,
            salt=101 + row_idx * 17,
        )
        apply_channel_state(env, execution_state)
        true_candidate = execution_candidate_for_decision(
            env,
            args,
            decision,
            execution_error_std,
            slot_idx,
        )
        success, failed, power_avg = mask_metrics(env, args, decision, true_candidate)
        score = (
            float(success) / float(max(args.num_nodes, 1))
            - float(args.target_failure_weight) * float(failed) / float(max(args.num_nodes, 1))
            - float(args.target_power_weight) * power_avg / max(float(getattr(args, "P_max", 1.0)), 1e-12)
        )
        scores.append(score)
        tx_values.append(float(success) / float(max(args.num_nodes, 1)))
    apply_channel_state(env, snapshot)
    return np.asarray(scores, dtype=np.float32), np.asarray(tx_values, dtype=np.float32)


def choose_behavior_offset(target_scores, offsets, rng, random_prob):
    """Choose a data-collection behavior offset."""
    if rng.random() < float(random_prob):
        return int(rng.choice(offsets))
    return int(offsets[int(np.argmax(target_scores))])


def dagger_beta(args, iteration):
    """Return expert-mixing probability for one DAgger iteration."""
    total = int(args.dagger_iterations)
    if total <= 1:
        return float(args.dagger_beta_start)
    fraction = float(iteration - 1) / float(max(total - 1, 1))
    return float(args.dagger_beta_start + fraction * (args.dagger_beta_end - args.dagger_beta_start))


def choose_dagger_offset(model, mean, std, feature, target_scores, args, beta, rng):
    """Choose a rollout offset using an expert/model DAgger mixture."""
    if rng.random() < float(beta):
        return int(args.offsets[int(np.argmax(target_scores))])
    scores = predict_offset_scores(model, feature, mean, std, torch.device(args.device))
    return int(args.offsets[int(np.argmax(scores))])


def execute_offset_slot(
    env,
    args,
    budget,
    slot_idx,
    decision_error_std,
    execution_error_std,
    episode_seed,
    decision_state,
    execution_state,
    offset,
):
    """Execute one slot using a chosen relative rotating offset."""
    indices = offset_indices(args, budget, slot_idx, offset)
    apply_channel_state(env, decision_state)
    decision, candidates = estimated_window_decision(
        env,
        args,
        indices,
        decision_error_std,
        episode_seed,
        slot_idx,
        salt=211,
    )
    apply_channel_state(env, execution_state)
    true_selected = execution_candidate_for_decision(
        env,
        args,
        decision,
        execution_error_std,
        slot_idx,
    )
    execution_oracle = execution_oracle_candidate(env, args, execution_error_std, slot_idx)
    info, done = limited.execute_limited_csi_slot(env, args, decision, true_selected)
    oracle_gap = max(0.0, float(execution_oracle["tx_this_slot"]) - float(info["tx_this_slot"]))
    return info, done, decision, candidates, indices, oracle_gap


def collect_dataset(args, episodes, seed, split_name):
    """Collect temporal offset features and hidden supervised targets."""
    eval_args = build_eval_args(args, episodes=episodes, num_seeds=1)
    eval_args.P_max = 1.0
    env = limited.make_env(eval_args)
    specs = split_episode_specs(seed, episodes, args.channel_rho_values, args.csi_delay_slots)
    features = []
    targets = []
    target_tx = []
    success_nodes = []

    print(f"Collecting {split_name} temporal-deviation data: episodes={episodes}, seed={seed}")
    for ep, (episode_seed, channel_rho, csi_delay_slots) in enumerate(specs, start=1):
        env.reset(seed=episode_seed)
        env._last_seed = episode_seed
        history = initialize_history(eval_args)
        behavior_rng = learned_rng(episode_seed, salt=13)
        temporal_states = build_temporal_channel_states(env, eval_args, episode_seed, channel_rho)
        total_tx = 0

        for slot_idx in range(eval_args.num_slots):
            execution_state = temporal_states[min(slot_idx, len(temporal_states) - 1)]
            decision_state = delayed_channel_state(temporal_states, slot_idx, csi_delay_slots)
            feature = policy_features(
                env,
                history,
                eval_args,
                channel_rho,
                csi_delay_slots,
                eval_args.probe_budgets[0],
                slot_idx,
            )
            score_target, tx_target = offset_target_scores(
                env,
                eval_args,
                eval_args.probe_budgets[0],
                slot_idx,
                channel_rho,
                csi_delay_slots,
                args.decision_error_std,
                args.execution_error_std,
                episode_seed,
                decision_state,
                execution_state,
                args.offsets,
            )
            features.append(feature)
            targets.append(score_target)
            target_tx.append(tx_target)

            chosen_offset = choose_behavior_offset(
                score_target,
                args.offsets,
                behavior_rng,
                args.behavior_random_prob,
            )
            info, done, decision, candidates, indices, _oracle_gap = execute_offset_slot(
                env,
                eval_args,
                eval_args.probe_budgets[0],
                slot_idx,
                args.decision_error_std,
                args.execution_error_std,
                episode_seed,
                decision_state,
                execution_state,
                chosen_offset,
            )
            total_tx = int(info["total_tx"])
            update_history(history, eval_args, indices, candidates, decision, info, chosen_offset)
            if done:
                break
        success_nodes.append(total_tx)
        print_progress(f"collect {split_name}", ep, episodes)

    print(
        f"Collected {split_name}: samples={len(targets)}, "
        f"mean behavior success={np.mean(success_nodes):.3f}/{args.num_nodes}"
    )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(target_tx, dtype=np.float32),
    )


def collect_dagger_dataset(args, episodes, seed, iteration, model, mean, std, beta):
    """Collect on-policy DAgger states and hidden supervised targets."""
    eval_args = build_eval_args(args, episodes=episodes, num_seeds=1)
    eval_args.P_max = 1.0
    eval_args.device = args.device
    env = limited.make_env(eval_args)
    specs = split_episode_specs(seed, episodes, args.channel_rho_values, args.csi_delay_slots)
    features = []
    targets = []
    target_tx = []
    success_nodes = []

    print(
        f"Collecting DAgger temporal-deviation data: iteration={iteration}, "
        f"episodes={episodes}, beta={beta:.3f}, seed={seed}"
    )
    for ep, (episode_seed, channel_rho, csi_delay_slots) in enumerate(specs, start=1):
        env.reset(seed=episode_seed)
        env._last_seed = episode_seed
        history = initialize_history(eval_args)
        behavior_rng = learned_rng(episode_seed, salt=97 + int(iteration))
        temporal_states = build_temporal_channel_states(env, eval_args, episode_seed, channel_rho)
        total_tx = 0

        for slot_idx in range(eval_args.num_slots):
            execution_state = temporal_states[min(slot_idx, len(temporal_states) - 1)]
            decision_state = delayed_channel_state(temporal_states, slot_idx, csi_delay_slots)
            feature = policy_features(
                env,
                history,
                eval_args,
                channel_rho,
                csi_delay_slots,
                eval_args.probe_budgets[0],
                slot_idx,
            )
            score_target, tx_target = offset_target_scores(
                env,
                eval_args,
                eval_args.probe_budgets[0],
                slot_idx,
                channel_rho,
                csi_delay_slots,
                args.decision_error_std,
                args.execution_error_std,
                episode_seed,
                decision_state,
                execution_state,
                args.offsets,
            )
            features.append(feature)
            targets.append(score_target)
            target_tx.append(tx_target)

            chosen_offset = choose_dagger_offset(
                model,
                mean,
                std,
                feature,
                score_target,
                eval_args,
                beta,
                behavior_rng,
            )
            info, done, decision, candidates, indices, _oracle_gap = execute_offset_slot(
                env,
                eval_args,
                eval_args.probe_budgets[0],
                slot_idx,
                args.decision_error_std,
                args.execution_error_std,
                episode_seed,
                decision_state,
                execution_state,
                chosen_offset,
            )
            total_tx = int(info["total_tx"])
            update_history(history, eval_args, indices, candidates, decision, info, chosen_offset)
            if done:
                break
        success_nodes.append(total_tx)
        print_progress(f"collect dagger {iteration}", ep, episodes)

    print(
        f"Collected DAgger iteration {iteration}: samples={len(targets)}, "
        f"mean behavior success={np.mean(success_nodes):.3f}/{args.num_nodes}"
    )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(target_tx, dtype=np.float32),
    )


def normalize_features(train_x, val_x):
    """Normalize features with train-split statistics."""
    feature_dim = int(train_x.shape[-1])
    train_flat = train_x.reshape(-1, feature_dim)
    mean = train_flat.mean(axis=0).astype(np.float32)
    std = train_flat.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    shape = (1,) * (train_x.ndim - 1) + (feature_dim,)
    mean = mean.reshape(shape)
    std = std.reshape(shape)
    return (
        np.clip((train_x - mean) / std, -10.0, 10.0).astype(np.float32),
        np.clip((val_x - mean) / std, -10.0, 10.0).astype(np.float32),
        mean,
        std,
    )


class OffsetSelector(nn.Module):
    """MLP that predicts hidden temporal-deviation utility for each offset."""

    def __init__(self, input_dim, output_dim, hidden_size, hidden_layers):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_size))
            layers.append(nn.ReLU())
            dim = hidden_size
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_input):
        """Return one score per offset."""
        return self.net(x_input)


class WindowOffsetSelector(nn.Module):
    """Shared scorer that ranks each candidate offset window."""

    def __init__(self, input_dim, hidden_size, hidden_layers):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_size))
            layers.append(nn.ReLU())
            dim = hidden_size
        layers.append(nn.Linear(dim, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(self, x_input):
        """Return one score per candidate offset window."""
        batch_size, offset_count, feature_dim = x_input.shape
        flat = x_input.reshape(batch_size * offset_count, feature_dim)
        return self.scorer(flat).reshape(batch_size, offset_count)


def train_model(args, train_x, train_y, val_x, val_y):
    """Train the offset selector."""
    train_x_norm, val_x_norm, mean, std = normalize_features(train_x, val_x)
    device = torch.device(args.device)
    if getattr(args, "feature_mode", "global") == "window":
        model = WindowOffsetSelector(
            input_dim=train_x.shape[-1],
            hidden_size=args.hidden_size,
            hidden_layers=args.hidden_layers,
        ).to(device)
    else:
        model = OffsetSelector(
            input_dim=train_x.shape[1],
            output_dim=train_y.shape[1],
            hidden_size=args.hidden_size,
            hidden_layers=args.hidden_layers,
        ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_x_tensor = torch.from_numpy(val_x_norm).to(device)
    val_y_tensor = torch.from_numpy(val_y).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    history = []
    best_state = None
    best_val_loss = float("inf")

    print("Training temporal deviation selector...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x_tensor), val_y_tensor).item())
        train_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 10, 1) == 0:
            print(f"  epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, history


def predict_offset_scores(model, feature, mean, std, device):
    """Predict offset scores for one feature vector."""
    feature_batch = feature.reshape((1,) + feature.shape).astype(np.float32)
    feature_norm = np.clip((feature_batch - mean) / std, -10.0, 10.0)
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(feature_norm.astype(np.float32)).to(device)
        return model(tensor).detach().cpu().numpy()[0]


def choose_offset_with_gate(scores, offsets, gate_margin):
    """Choose the predicted offset, falling back to rotating if confidence is low."""
    best_idx = int(np.argmax(scores))
    if float(gate_margin) <= 0.0:
        return int(offsets[best_idx])
    rotating_idx = list(offsets).index(0)
    if best_idx == rotating_idx:
        return 0
    margin = float(scores[best_idx] - scores[rotating_idx])
    if margin >= float(gate_margin):
        return int(offsets[best_idx])
    return 0


def validation_metrics(predictions, target_scores, target_tx):
    """Compute validation offset-selection diagnostics."""
    chosen = np.argmax(predictions, axis=1)
    best_score = np.max(target_scores, axis=1)
    chosen_score = target_scores[np.arange(len(chosen)), chosen]
    best_tx = np.max(target_tx, axis=1)
    chosen_tx = target_tx[np.arange(len(chosen)), chosen]
    return [
        {
            "samples": int(len(chosen)),
            "offset_hit_rate": float(np.mean(chosen_score >= best_score - 1e-7) * 100.0),
            "target_score_gap_mean": float(np.mean(best_score - chosen_score)),
            "target_tx_gap_mean": float(np.mean(best_tx - chosen_tx)),
        }
    ]


def learned_policy_name(args, gate_margin=0.0):
    """Return the learned policy label for output tables."""
    is_dagger = int(getattr(args, "dagger_iterations", 0)) > 0
    is_window = getattr(args, "feature_mode", "global") == "window"
    is_gated = float(gate_margin) > 0.0
    if is_dagger and is_window and is_gated:
        return POLICY_DAGGER_GATED_WINDOW_TEMPORAL_DEVIATION
    if is_dagger and is_gated:
        return POLICY_DAGGER_GATED_TEMPORAL_DEVIATION
    if is_window and is_gated:
        return POLICY_GATED_WINDOW_TEMPORAL_DEVIATION
    if is_gated:
        return POLICY_GATED_TEMPORAL_DEVIATION
    if is_dagger and is_window:
        return POLICY_DAGGER_WINDOW_TEMPORAL_DEVIATION
    if is_dagger:
        return POLICY_DAGGER_TEMPORAL_DEVIATION
    if is_window:
        return POLICY_WINDOW_TEMPORAL_DEVIATION
    return POLICY_LEARNED_TEMPORAL_DEVIATION


def learned_result_name(args, budget, gate_margin=0.0):
    """Return the display name for learned temporal deviation."""
    if float(gate_margin) > 0.0:
        return f"{learned_policy_name(args, gate_margin)} m={float(gate_margin):g} B={int(budget)}"
    return f"{learned_policy_name(args, gate_margin)} B={int(budget)}"


def evaluate_learned_policy(
    episode_seeds,
    args,
    channel_rho,
    csi_delay_slots,
    budget,
    model,
    mean,
    std,
    gate_margin=0.0,
):
    """Evaluate learned temporal deviation for one scenario and run seed."""
    env = limited.make_env(args)
    device = torch.device(getattr(args, "device", "cpu"))
    success_nodes = []
    avg_power = []
    rewards = []
    slots_used = []
    total_energy = []
    scheduled_nodes = []
    failed_nodes = []
    missed_opportunities = []
    true_opportunities = []
    failure_slots = []
    preview_calls_per_slot = []
    oracle_tx_gap_mean = []
    effective_risk_weights = []

    print(
        f"Running {learned_result_name(args, budget, gate_margin)} model=temporal_ar1 "
        f"rho={channel_rho:g} delay={int(csi_delay_slots)}..."
    )
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        env.reset(seed=episode_seed)
        env._last_seed = episode_seed
        history = initialize_history(args)
        temporal_states = build_temporal_channel_states(env, args, episode_seed, channel_rho)
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
            execution_state = temporal_states[min(slot_idx, len(temporal_states) - 1)]
            decision_state = delayed_channel_state(temporal_states, slot_idx, csi_delay_slots)
            feature = policy_features(env, history, args, channel_rho, csi_delay_slots, budget, slot_idx)
            scores = predict_offset_scores(model, feature, mean, std, device)
            chosen_offset = choose_offset_with_gate(scores, args.offsets, gate_margin)
            info, done, decision, candidates, indices, oracle_gap = execute_offset_slot(
                env,
                args,
                budget,
                slot_idx,
                args.decision_error_std,
                args.execution_error_std,
                episode_seed,
                decision_state,
                execution_state,
                chosen_offset,
            )
            total_tx = int(info["total_tx"])
            episode_reward += float(info["reward"])
            episode_slots = int(info.get("slots_used", slot_idx + 1))
            episode_energy += float(info["attempted_energy"])
            episode_scheduled += int(info["scheduled_this_slot"])
            episode_failed += int(info["failed_this_slot"])
            episode_missed += int(info["missed_opportunity_this_slot"])
            episode_true_opportunities += int(info["true_opportunity_this_slot"])
            episode_failure_slots += int(info["failed_this_slot"] > 0)
            episode_preview_calls.append(len(indices))
            episode_oracle_gaps.append(oracle_gap)
            episode_effective_risk_weights.append(0.0)
            if info["scheduled_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))
            update_history(history, args, indices, candidates, decision, info, chosen_offset)
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
        failure_slots.append(float(episode_failure_slots) / max(float(episode_slots), 1.0))
        preview_calls_per_slot.append(float(sum(episode_preview_calls)) / max(len(episode_preview_calls), 1))
        oracle_tx_gap_mean.append(float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0)
        effective_risk_weights.append(
            float(np.mean(episode_effective_risk_weights)) if episode_effective_risk_weights else 0.0
        )
        limited.print_progress(
            learned_policy_name(args, gate_margin),
            0.0,
            budget,
            ep,
            args.episodes,
            success_nodes,
            args.num_nodes,
        )

    return {
        "name": learned_result_name(args, budget, gate_margin),
        "policy": learned_policy_name(args, gate_margin),
        "decision_error_std": float(args.decision_error_std),
        "execution_error_std": float(args.execution_error_std),
        "mismatch_model": mismatch.MISMATCH_TEMPORAL_AR1,
        "channel_rho": float(channel_rho),
        "csi_delay_slots": int(csi_delay_slots),
        "probe_budget": int(budget),
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
        "success_nodes": np.asarray(success_nodes, dtype=float),
        "avg_power": np.asarray(avg_power, dtype=float),
        "episode_reward": np.asarray(rewards, dtype=float),
        "slots_used": np.asarray(slots_used, dtype=float),
        "total_energy": np.asarray(total_energy, dtype=float),
        "scheduled_nodes": np.asarray(scheduled_nodes, dtype=float),
        "failed_nodes": np.asarray(failed_nodes, dtype=float),
        "missed_opportunities": np.asarray(missed_opportunities, dtype=float),
        "true_opportunities": np.asarray(true_opportunities, dtype=float),
        "failure_slots": np.asarray(failure_slots, dtype=float),
        "decision_preview_calls_per_slot": np.asarray(preview_calls_per_slot, dtype=float),
        "oracle_tx_gap_mean": np.asarray(oracle_tx_gap_mean, dtype=float),
        "effective_risk_weight": np.asarray(effective_risk_weights, dtype=float),
    }


def evaluate_suite(args, model, mean, std):
    """Evaluate learned temporal deviation and diagnostic baselines."""
    eval_args = build_eval_args(args)
    eval_args.P_max = 1.0
    eval_args.device = args.device
    eval_args.offsets = list(args.offsets)
    eval_args.decision_error_std = float(args.decision_error_std)
    eval_args.execution_error_std = float(args.execution_error_std)
    eval_args.target_power_weight = float(args.target_power_weight)
    eval_args.target_failure_weight = float(args.target_failure_weight)
    eval_args.history_lr = float(args.history_lr)
    run_seeds = make_run_seeds(eval_args)
    episode_seed_sets = [make_episode_seeds(eval_args, run_seed) for run_seed in run_seeds]
    all_rows = []

    for channel_rho in eval_args.channel_rho_values:
        for csi_delay_slots in eval_args.csi_delay_slots:
            seed_result_sets = []
            for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
                print(
                    f"Eval rho={channel_rho:g} delay={int(csi_delay_slots)} "
                    f"run [{run_idx}/{len(run_seeds)}], seed={run_seeds[run_idx - 1]}"
                )
                seed_results = [
                    mismatch.evaluate_policy(
                        episode_seeds,
                        eval_args,
                        args.decision_error_std,
                        args.execution_error_std,
                        mismatch.MISMATCH_TEMPORAL_AR1,
                        channel_rho,
                        csi_delay_slots,
                        policy_name=mismatch.POLICY_EXECUTION_ORACLE,
                        budget=eval_args.num_codebook_states,
                    ),
                    mismatch.evaluate_policy(
                        episode_seeds,
                        eval_args,
                        args.decision_error_std,
                        args.execution_error_std,
                        mismatch.MISMATCH_TEMPORAL_AR1,
                        channel_rho,
                        csi_delay_slots,
                        policy_name=limited.POLICY_EXACT_GREEDY,
                        budget=eval_args.num_codebook_states,
                    ),
                ]
                for budget in eval_args.probe_budgets:
                    seed_results.append(
                        mismatch.evaluate_policy(
                            episode_seeds,
                            eval_args,
                            args.decision_error_std,
                            args.execution_error_std,
                            mismatch.MISMATCH_TEMPORAL_AR1,
                            channel_rho,
                            csi_delay_slots,
                            policy_name=limited.POLICY_ROTATING_GRID,
                            budget=budget,
                        )
                    )
                    for gate_margin in eval_args.gate_margin_thresholds:
                        seed_results.append(
                            evaluate_learned_policy(
                                episode_seeds,
                                eval_args,
                                channel_rho,
                                csi_delay_slots,
                                budget,
                                model,
                                mean,
                                std,
                                gate_margin=gate_margin,
                            )
                        )
                    seed_results.append(
                        mismatch.evaluate_policy(
                            episode_seeds,
                            eval_args,
                            args.decision_error_std,
                            args.execution_error_std,
                            mismatch.MISMATCH_TEMPORAL_AR1,
                            channel_rho,
                            csi_delay_slots,
                            policy_name=mismatch.POLICY_TEMPORAL_DEVIATION_ORACLE_GRID,
                            budget=budget,
                        )
                    )
                seed_result_sets.append(seed_results)
            all_rows.extend(mismatch.summarize_results(eval_args, mismatch.aggregate_seed_results(seed_result_sets)))
    return all_rows


def write_train_history(path, rows):
    """Write train/validation loss history."""
    ensure_parent_dir(path)
    fieldnames = ["epoch", "train_loss", "val_loss"]
    if any("iteration" in row for row in rows):
        fieldnames = ["iteration"] + fieldnames
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def write_validation_metrics(path, rows):
    """Write validation diagnostics."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["samples", "offset_hit_rate", "target_score_gap_mean", "target_tx_gap_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_checkpoint(path, model, mean, std, args):
    """Save model and normalization state."""
    ensure_parent_dir(path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_mean": mean,
            "obs_std": std,
            "offsets": list(args.offsets),
            "input_dim": int(mean.shape[-1]),
            "feature_mode": args.feature_mode,
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "channel_rho_values": list(args.channel_rho_values),
            "csi_delay_slots": list(args.csi_delay_slots),
            "gate_margin_thresholds": list(args.gate_margin_thresholds),
            "dagger_iterations": int(getattr(args, "dagger_iterations", 0)),
            "dagger_episodes": int(getattr(args, "dagger_episodes", 0)),
        },
        path,
    )
    print(f"Saved: {path}")


def tag_train_history(rows, iteration):
    """Add DAgger iteration metadata to training rows."""
    return [dict(row, iteration=int(iteration)) for row in rows]


def print_best_rows(rows):
    """Print compact learned-vs-baseline summary."""
    print("=" * 144)
    print("Learned Temporal Deviation Summary")
    print("=" * 144)
    print(
        f"{'Rho':>5} {'Delay':>5} {'Best non-oracle':<38} {'Success':>9} "
        f"{'Perfect%':>9} {'Slots':>7} {'Gap':>7}"
    )
    for channel_rho in sorted({row["channel_rho"] for row in rows}):
        for csi_delay_slots in sorted({row["csi_delay_slots"] for row in rows if row["channel_rho"] == channel_rho}):
            subset = [
                row
                for row in rows
                if row["channel_rho"] == channel_rho and row["csi_delay_slots"] == csi_delay_slots
            ]
            non_oracle = [row for row in subset if row["policy"] != mismatch.POLICY_EXECUTION_ORACLE]
            best = max(
                non_oracle,
                key=lambda row: (
                    row["success_mean"],
                    row["perfect_rate"],
                    -row["oracle_tx_gap_mean"],
                    -row["slots_mean"],
                ),
            )
            print(
                f"{channel_rho:>5.2f} {int(csi_delay_slots):>5} {best['policy']:<38} "
                f"{best['success_mean']:>9.3f} {best['perfect_rate']:>8.2f}% "
                f"{best['slots_mean']:>7.3f} {best['oracle_tx_gap_mean']:>7.3f}"
            )


def main():
    """Train and evaluate the temporal deviation selector."""
    args = parse_args()
    validate_args(args)
    set_training_seed(args)
    output_prefix = resolve_output_prefix(args)

    print("=" * 96)
    print(
        f"Learned temporal deviation: train_episodes={args.train_episodes}, "
        f"eval_episodes={args.eval_episodes}, rhos={args.channel_rho_values}, "
        f"delays={args.csi_delay_slots}, budgets={args.probe_budgets}, "
        f"feature_mode={args.feature_mode}, gate_margins={args.gate_margin_thresholds}"
    )
    if args.dagger_iterations > 0:
        print(
            f"DAgger: iterations={args.dagger_iterations}, episodes={args.dagger_episodes}, "
            f"beta={args.dagger_beta_start:g}->{args.dagger_beta_end:g}"
        )
    print(f"Offsets={args.offsets}, output_prefix={output_prefix}")
    print("=" * 96)

    train_x, train_y, train_tx = collect_dataset(args, args.train_episodes, args.seed + 17, "train")
    val_x, val_y, val_tx = collect_dataset(args, args.val_episodes, args.seed + 31, "val")
    model, mean, std, history = train_model(args, train_x, train_y, val_x, val_y)
    all_history = tag_train_history(history, iteration=0) if args.dagger_iterations > 0 else list(history)

    for iteration in range(1, args.dagger_iterations + 1):
        beta = dagger_beta(args, iteration)
        dagger_x, dagger_y, dagger_tx = collect_dagger_dataset(
            args,
            args.dagger_episodes,
            args.seed + 1009 + 37 * iteration,
            iteration,
            model,
            mean,
            std,
            beta,
        )
        train_x = np.concatenate([train_x, dagger_x], axis=0)
        train_y = np.concatenate([train_y, dagger_y], axis=0)
        train_tx = np.concatenate([train_tx, dagger_tx], axis=0)
        print(
            f"Retraining after DAgger iteration {iteration}: "
            f"total_samples={len(train_y)}, target_tx_mean={np.mean(train_tx):.4f}"
        )
        model, mean, std, iteration_history = train_model(args, train_x, train_y, val_x, val_y)
        all_history.extend(tag_train_history(iteration_history, iteration=iteration))

    write_train_history(f"{output_prefix}_train_history.csv", all_history)
    save_checkpoint(f"{output_prefix}_model.pt", model, mean, std, args)

    val_x_norm = np.clip((val_x - mean) / std, -10.0, 10.0).astype(np.float32)
    with torch.no_grad():
        predictions = model(torch.from_numpy(val_x_norm).to(torch.device(args.device))).cpu().numpy()
    write_validation_metrics(
        f"{output_prefix}_val_offsets.csv",
        validation_metrics(predictions, val_y, val_tx),
    )

    rows = evaluate_suite(args, model, mean, std)
    print_best_rows(rows)
    mismatch.print_summary(rows)
    mismatch.write_csv(f"{output_prefix}.csv", rows)


if __name__ == "__main__":
    main()
