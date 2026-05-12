"""
Train and evaluate a feedback-conditioned IRS probing selector.

Training may use offline oracle labels, but evaluation stays in the strict
bandit-feedback setting: the policy sees only the base environment observation,
past noisy aggregate probe feedback, and its own probing history. It never sees
full CSI, node-level masks, or full codebook features at decision time.
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


POLICY_LEARNED_FEEDBACK = "Learned Feedback Probe"
LEARNED_OFFSET = 0x94D049BB


def parse_csv_items(value):
    """Parse a comma-separated list of non-empty strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    """Parse training and evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Train a feedback-conditioned selector for aggregate-feedback IRS probing."
    )
    parser.add_argument("--scenario", choices=sorted(stress.SCENARIO_PRESETS), default="short_slots")
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--val-episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-eval-seeds", type=int, default=5)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--probe-budgets", default="1,2,4")
    parser.add_argument("--feedback-noise-std-values", default="0.2")
    parser.add_argument("--train-feedback-noise-std", type=float, default=0.2)
    parser.add_argument("--behavior-policy", choices=["rotating", "random", "mixed"], default="mixed")
    parser.add_argument("--behavior-probe-budget", type=int, default=1)
    parser.add_argument("--mixed-random-prob", type=float, default=0.5)
    parser.add_argument("--baseline-policies", default="no_irs,random_irs,oracle,full_feedback")
    parser.add_argument("--probe-policies", default="rotating,ucb,thompson")
    parser.add_argument("--slot-cost", type=float, default=0.10)
    parser.add_argument("--probe-cost", type=float, default=0.005)
    parser.add_argument("--power-feedback-noise-std", type=float, default=0.0)
    parser.add_argument("--feedback-power-weight", type=float, default=0.05)
    parser.add_argument("--history-lr", type=float, default=0.60)
    parser.add_argument("--ucb-coeff", type=float, default=0.25)
    parser.add_argument("--thompson-std", type=float, default=0.20)
    parser.add_argument("--bandit-lr", type=float, default=0.60)
    parser.add_argument("--bandit-prior", type=float, default=0.0)
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
    """Validate training, feedback, and scenario parameters."""
    for name in (
        "train_episodes",
        "val_episodes",
        "eval_episodes",
        "num_eval_seeds",
        "behavior_probe_budget",
        "hidden_size",
        "hidden_layers",
        "epochs",
        "batch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.train_feedback_noise_std < 0.0:
        raise ValueError("--train-feedback-noise-std must be non-negative")
    if not 0.0 <= args.history_lr <= 1.0:
        raise ValueError("--history-lr must be in [0, 1]")
    if not 0.0 <= args.mixed_random_prob <= 1.0:
        raise ValueError("--mixed-random-prob must be in [0, 1]")
    if args.slot_cost < 0.0 or args.probe_cost < 0.0:
        raise ValueError("--slot-cost and --probe-cost must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")

    args.probe_budgets = sorted(
        {int(value) for value in bandit.parse_int_list(args.probe_budgets)}
    )
    args.feedback_noise_std_values = bandit.parse_float_list(args.feedback_noise_std_values)
    if not args.probe_budgets:
        raise ValueError("--probe-budgets must contain at least one value")
    if not args.feedback_noise_std_values:
        raise ValueError("--feedback-noise-std-values must contain at least one value")

    args.baseline_policies = parse_csv_items(args.baseline_policies)
    unknown_baselines = [name for name in args.baseline_policies if name not in stress.BASELINE_POLICIES]
    if unknown_baselines:
        raise ValueError(f"Unknown baseline policies: {unknown_baselines}")

    args.probe_policies = parse_csv_items(args.probe_policies)
    unknown_probe_policies = [name for name in args.probe_policies if name not in stress.PROBE_POLICIES]
    if unknown_probe_policies:
        raise ValueError(f"Unknown probe policies: {unknown_probe_policies}")


def build_bandit_args(args):
    """Build the namespace expected by the aggregate-feedback evaluator."""
    config = stress.scenario_config(args.scenario)
    bandit_args = argparse.Namespace(
        episodes=args.eval_episodes,
        seed=args.seed,
        num_seeds=args.num_eval_seeds,
        seed_stride=args.seed_stride,
        probe_budgets=",".join(str(value) for value in args.probe_budgets),
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
        output_prefix=None,
        no_plots=args.no_plots,
    )
    bandit.validate_args(bandit_args)
    return bandit_args


def resolve_output_prefix(args, bandit_args):
    """Resolve output prefix for model, CSV, and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    budget_label = "-".join(format_float_for_suffix(value) for value in args.probe_budgets)
    noise_label = "-".join(format_float_for_suffix(value) for value in args.feedback_noise_std_values)
    train_noise_label = format_float_for_suffix(args.train_feedback_noise_std)
    suffix = (
        f"{args.scenario}_train{args.train_episodes}_eval{args.eval_episodes}_"
        f"runs{args.num_eval_seeds}_seed{args.seed}_c{bandit_args.num_codebook_states}_"
        f"b{budget_label}_trainnoise{train_noise_label}_evalnoise{noise_label}_"
        f"{args.behavior_policy}_bb{args.behavior_probe_budget}"
    )
    output_prefix = os.path.join("results", "bandit_feedback", f"learned_feedback_probe_{suffix}")
    ensure_parent_dir(output_prefix)
    return output_prefix


def split_seeds(seed, episodes):
    """Generate deterministic episode seeds for a dataset split."""
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**31 - 1, size=episodes)]


def learned_rng(episode_seed, feedback_noise_std, budget, salt=0):
    """Create deterministic RNG streams for learned probing evaluation."""
    if episode_seed is None:
        return np.random.default_rng()
    noise_tag = int(round(float(feedback_noise_std) * 1_000_000))
    seed = (
        int(episode_seed)
        + LEARNED_OFFSET
        + int(budget) * 0x9E3779B1
        + noise_tag * 0x85EBCA6B
        + int(salt) * 0x165667B1
    ) % (2**32)
    return np.random.default_rng(seed)


def initialize_history(args):
    """Initialize observable aggregate-feedback history."""
    c_count = int(args.num_codebook_states)
    return {
        "counts": np.zeros(c_count, dtype=float),
        "mean_score": np.zeros(c_count, dtype=float),
        "mean_tx": np.zeros(c_count, dtype=float),
        "mean_power": np.zeros(c_count, dtype=float),
        "age": np.full(c_count, float(args.num_slots + 1), dtype=float),
        "last_selected": -1,
        "last_score": 0.0,
        "last_tx": 0.0,
        "last_power": 0.0,
    }


def update_history(history, feedbacks, args):
    """Update observable aggregate-feedback history after a probing slot."""
    history["age"] += 1.0
    lr = float(getattr(args, "history_lr", 0.60))
    for feedback in feedbacks:
        index = int(feedback["irs_index"])
        old_count = float(history["counts"][index])
        if old_count <= 0.0:
            history["mean_score"][index] = float(feedback["observed_score"])
            history["mean_tx"][index] = float(feedback["observed_tx_fraction"])
            history["mean_power"][index] = float(feedback["observed_power"])
        else:
            history["mean_score"][index] = (
                (1.0 - lr) * history["mean_score"][index]
                + lr * float(feedback["observed_score"])
            )
            history["mean_tx"][index] = (
                (1.0 - lr) * history["mean_tx"][index]
                + lr * float(feedback["observed_tx_fraction"])
            )
            history["mean_power"][index] = (
                (1.0 - lr) * history["mean_power"][index]
                + lr * float(feedback["observed_power"])
            )
        history["counts"][index] = old_count + 1.0
        history["age"][index] = 0.0
        history["last_selected"] = index
        history["last_score"] = float(feedback["observed_score"])
        history["last_tx"] = float(feedback["observed_tx_fraction"])
        history["last_power"] = float(feedback["observed_power"])


def history_features(obs, history, args):
    """Build model features from base observation and observable feedback history."""
    c_count = int(args.num_codebook_states)
    base = np.asarray(obs[:7], dtype=np.float32)
    counts = np.clip(history["counts"] / max(args.num_slots, 1), 0.0, 1.0)
    recency = 1.0 / (1.0 + np.asarray(history["age"], dtype=float))
    selected = np.zeros(c_count, dtype=float)
    previous_index = int(history["last_selected"])
    if previous_index >= 0:
        selected[previous_index] = 1.0
        previous_norm = (
            0.0 if c_count <= 1 else 2.0 * float(previous_index) / float(c_count - 1) - 1.0
        )
        angle = 2.0 * np.pi * float(previous_index) / float(c_count)
        previous_sin = float(np.sin(angle))
        previous_cos = float(np.cos(angle))
        previous_known = 1.0
    else:
        previous_norm = -1.0
        previous_sin = 0.0
        previous_cos = 0.0
        previous_known = 0.0

    scalars = np.asarray(
        [
            previous_known,
            previous_norm,
            previous_sin,
            previous_cos,
            float(history["last_score"]),
            float(history["last_tx"]),
            float(history["last_power"]),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            base,
            counts.astype(np.float32),
            np.asarray(history["mean_score"], dtype=np.float32),
            np.asarray(history["mean_tx"], dtype=np.float32),
            np.asarray(history["mean_power"], dtype=np.float32),
            recency.astype(np.float32),
            selected.astype(np.float32),
            scalars,
        ]
    ).astype(np.float32)


def target_scores(candidates, args):
    """Build supervised targets from hidden oracle candidate metrics."""
    scores = np.zeros(args.num_codebook_states, dtype=np.float32)
    tx = np.zeros(args.num_codebook_states, dtype=np.float32)
    for candidate in candidates:
        index = int(candidate["irs_index"])
        tx_fraction = float(candidate["tx_this_slot"]) / float(max(args.num_nodes, 1))
        power = float(candidate["power_avg"])
        scores[index] = tx_fraction - float(args.feedback_power_weight) * power
        tx[index] = tx_fraction
    return scores, tx


def behavior_indices(args, bandit_args, budget, slot_idx, rng):
    """Choose behavior probe indices used only for collecting training histories."""
    c_count = int(bandit_args.num_codebook_states)
    budget = min(int(budget), c_count)
    if budget >= c_count:
        return list(range(c_count))
    if args.behavior_policy == "random":
        return [int(index) for index in rng.choice(c_count, size=budget, replace=False)]
    if args.behavior_policy == "rotating":
        return bandit.grid_indices(c_count, budget, offset=slot_idx)
    if rng.random() < float(args.mixed_random_prob):
        return [int(index) for index in rng.choice(c_count, size=budget, replace=False)]
    return bandit.grid_indices(c_count, budget, offset=slot_idx)


def collect_dataset(args, bandit_args, episodes, seed, split_name):
    """
    Collect offline supervised data.

    The feature vector contains only observable history. Full candidate previews
    are used only to create training targets and are not exposed at evaluation.
    """
    env = bandit.make_env(bandit_args)
    base_action = bandit.make_base_action(bandit_args)
    episode_seeds = split_seeds(seed, episodes)
    features = []
    targets = []
    target_tx = []

    print(f"Collecting {split_name} feedback data: episodes={episodes}, seed={seed}")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        obs, _info = env.reset(seed=episode_seed)
        history = initialize_history(bandit_args)
        rng = learned_rng(episode_seed, args.train_feedback_noise_std, args.behavior_probe_budget, salt=7)
        feedback_rng = learned_rng(episode_seed, args.train_feedback_noise_std, args.behavior_probe_budget, salt=11)

        for slot_idx in range(bandit_args.num_slots):
            candidates = [
                bandit.preview_codebook_candidate(env, bandit_args, index)
                for index in range(bandit_args.num_codebook_states)
            ]
            score_target, tx_target = target_scores(candidates, bandit_args)
            features.append(history_features(obs, history, bandit_args))
            targets.append(score_target)
            target_tx.append(tx_target)

            indices = behavior_indices(args, bandit_args, args.behavior_probe_budget, slot_idx, rng)
            probed = [candidates[index] for index in indices]
            feedbacks = [
                bandit.observe_probe_feedback(candidate, bandit_args, args.train_feedback_noise_std, feedback_rng)
                for candidate in probed
            ]
            update_history(history, feedbacks, bandit_args)
            selected = bandit.select_from_feedback(probed, feedbacks)

            action = base_action.copy()
            action[2] = codebook_index_to_action(
                int(selected["irs_index"]),
                bandit_args.num_codebook_states,
            )
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

        print_progress(f"collect {split_name}", ep, episodes)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(target_tx, dtype=np.float32),
    )


def print_progress(name, current, total):
    """Print progress at 10 percent intervals."""
    interval = max(total // 10, 1)
    if current % interval == 0 or current == total:
        print(f"  {name}: [{current:04d}/{total:04d}]")


def normalize_features(train_x, val_x):
    """Normalize features using train-split statistics."""
    mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return (
        np.clip((train_x - mean) / std, -10.0, 10.0).astype(np.float32),
        np.clip((val_x - mean) / std, -10.0, 10.0).astype(np.float32),
        mean,
        std,
    )


class FeedbackSelector(nn.Module):
    """MLP that predicts codebook aggregate value from feedback history."""

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
        """Predict per-codebook aggregate scores."""
        return self.net(x_input)


def train_model(args, train_x, train_y, val_x, val_y):
    """Train the feedback-conditioned selector."""
    train_x_norm, val_x_norm, mean, std = normalize_features(train_x, val_x)
    device = torch.device(args.device)
    model = FeedbackSelector(
        input_dim=train_x.shape[1],
        output_dim=train_y.shape[1],
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
    ).to(device)

    train_ds = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_tensor = torch.from_numpy(val_x_norm).to(device)
    val_target = torch.from_numpy(val_y).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    history = []
    best_state = None
    best_val_loss = float("inf")

    print("Training feedback-conditioned selector...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_tensor), val_target).item())
        train_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"  epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, history


def predict_scores(model, feature, mean, std, device):
    """Predict score vector for one feature row."""
    feature_norm = np.clip(
        (feature.reshape(1, -1).astype(np.float32) - mean) / std,
        -10.0,
        10.0,
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(feature_norm).to(device)
        return model(tensor).detach().cpu().numpy()[0]


def validation_topk_metrics(predictions, target_tx, budgets, num_nodes):
    """Compute top-k oracle-hit and missed-tx metrics on validation states."""
    oracle_tx = np.max(target_tx, axis=1)
    rows = []
    for budget in budgets:
        budget = min(int(budget), target_tx.shape[1])
        top_indices = np.argpartition(-predictions, kth=budget - 1, axis=1)[:, :budget]
        selected_tx = np.take_along_axis(target_tx, top_indices, axis=1)
        best_selected_tx = np.max(selected_tx, axis=1)
        rows.append(
            {
                "probe_budget": budget,
                "val_oracle_hit_rate": float(
                    np.mean(np.isclose(best_selected_tx, oracle_tx, rtol=0.0, atol=1e-7)) * 100.0
                ),
                "val_oracle_tx_gap_mean": float(
                    np.mean((oracle_tx - best_selected_tx) * num_nodes)
                ),
            }
        )
    return rows


def top_indices(scores, budget, args, rng):
    """Select top-scoring codebook indices with tiny deterministic jitter."""
    c_count = int(args.num_codebook_states)
    budget = min(int(budget), c_count)
    if budget >= c_count:
        return list(range(c_count))
    jitter = rng.uniform(0.0, 1e-9, size=c_count)
    order = np.argsort(-(np.asarray(scores, dtype=float) + jitter))
    return [int(index) for index in order[:budget]]


def evaluate_learned_policy(episode_seeds, args, feedback_noise_std, budget, model, mean, std, base_action):
    """Evaluate learned feedback probing for one run seed."""
    env = bandit.make_env(args)
    device = torch.device(getattr(args, "device", "cpu"))
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

    print(f"Running {POLICY_LEARNED_FEEDBACK} noise={feedback_noise_std:g} B={budget}...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        obs, _info = env.reset(seed=episode_seed)
        rng = learned_rng(episode_seed, feedback_noise_std, budget)
        history = initialize_history(args)
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

        for _slot_idx in range(args.num_slots):
            oracle = bandit.full_oracle_candidate(env, args)
            feature = history_features(obs, history, args)
            scores = predict_scores(model, feature, mean, std, device)
            indices = top_indices(scores, budget, args, rng)
            candidates = [bandit.preview_codebook_candidate(env, args, index) for index in indices]
            feedbacks = [
                bandit.observe_probe_feedback(candidate, args, feedback_noise_std, rng)
                for candidate in candidates
            ]
            update_history(history, feedbacks, args)
            selected = bandit.select_from_feedback(candidates, feedbacks)
            selected_index = int(selected["irs_index"])

            episode_probe_calls.append(len(indices))
            episode_oracle_matches.append(float(selected_index == int(oracle["irs_index"])))
            episode_oracle_gaps.append(
                max(0.0, float(oracle["tx_this_slot"]) - float(selected["tx_this_slot"]))
            )
            for feedback in feedbacks:
                episode_observed_scores.append(float(feedback["observed_score"]))
                episode_observed_tx_fractions.append(float(feedback["observed_tx_fraction"]))

            action = base_action.copy()
            action[2] = codebook_index_to_action(selected_index, args.num_codebook_states)
            obs, reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            episode_reward += float(reward)
            episode_slots = int(info.get("slots_used", len(episode_probe_calls)))
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
        bandit.print_progress(
            POLICY_LEARNED_FEEDBACK,
            feedback_noise_std,
            budget,
            ep,
            args.episodes,
            success_nodes,
            args.num_nodes,
        )

    return {
        "name": POLICY_LEARNED_FEEDBACK,
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


def evaluate_suite(args, bandit_args, model, mean, std, feedback_noise_std):
    """Evaluate learned and selected baseline policies for one noise level."""
    base_action = bandit.make_base_action(bandit_args)
    run_seeds = make_run_seeds(bandit_args)
    episode_seed_sets = [make_episode_seeds(bandit_args, run_seed) for run_seed in run_seeds]
    baseline_policy_names = [stress.BASELINE_POLICIES[name] for name in args.baseline_policies]
    probe_policy_names = [stress.PROBE_POLICIES[name] for name in args.probe_policies]
    seed_result_sets = []

    for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
        print(f"Eval run [{run_idx}/{len(run_seeds)}], seed={run_seeds[run_idx - 1]}")
        seed_results = []
        for policy_name in baseline_policy_names:
            budget = bandit_args.num_codebook_states if policy_name in {
                bandit.POLICY_ORACLE_FULL,
                bandit.POLICY_FULL_FEEDBACK,
            } else 0
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
            seed_results.append(
                evaluate_learned_policy(
                    episode_seeds,
                    bandit_args,
                    feedback_noise_std,
                    budget,
                    model,
                    mean,
                    std,
                    base_action,
                )
            )
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
        seed_result_sets.append(seed_results)

    return bandit.summarize_results(bandit_args, bandit.aggregate_seed_results(seed_result_sets))


def write_train_history(path, rows):
    """Write training loss history."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def write_validation_metrics(path, rows):
    """Write validation top-k metrics."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["probe_budget", "val_oracle_hit_rate", "val_oracle_tx_gap_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_checkpoint(path, model, mean, std, args, bandit_args):
    """Save model and normalization state."""
    ensure_parent_dir(path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_mean": mean,
            "obs_std": std,
            "scenario": args.scenario,
            "num_codebook_states": bandit_args.num_codebook_states,
            "input_dim": int(mean.shape[1]),
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "behavior_policy": args.behavior_policy,
            "behavior_probe_budget": args.behavior_probe_budget,
            "train_feedback_noise_std": args.train_feedback_noise_std,
        },
        path,
    )
    print(f"Saved: {path}")


def policy_label(row):
    """Return compact plot labels."""
    policy = row["policy"]
    if policy in bandit.PROBE_POLICIES or policy == POLICY_LEARNED_FEEDBACK:
        return f"{policy} B={int(row['probe_budget'])}"
    return policy


def plot_results(rows, output_prefix):
    """Plot utility and perfect coverage vs feedback noise."""
    labels = []
    for row in rows:
        label = policy_label(row)
        if label not in labels:
            labels.append(label)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(idx % 20) for idx, label in enumerate(labels)}
    for label in labels:
        label_rows = sorted(
            [row for row in rows if policy_label(row) == label],
            key=lambda item: item["feedback_noise_std"],
        )
        x_values = [row["feedback_noise_std"] for row in label_rows]
        axes[0].plot(
            x_values,
            [row["utility_mean"] for row in label_rows],
            marker="o",
            linewidth=1.5,
            label=label,
            color=colors[label],
        )
        axes[1].plot(
            x_values,
            [row["success_mean"] for row in label_rows],
            marker="o",
            linewidth=1.5,
            label=label,
            color=colors[label],
        )
        axes[2].plot(
            x_values,
            [row["perfect_rate"] for row in label_rows],
            marker="o",
            linewidth=1.5,
            label=label,
            color=colors[label],
        )

    axes[0].set_title("Utility vs Feedback Noise")
    axes[0].set_ylabel("Node-Equivalent Utility")
    axes[1].set_title("Success vs Feedback Noise")
    axes[1].set_ylabel("Successful Nodes")
    axes[2].set_title("Perfect Coverage vs Feedback Noise")
    axes[2].set_ylabel("Perfect Episodes (%)")
    for ax in axes:
        ax.set_xlabel("Feedback Noise Std")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = f"{output_prefix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def print_best_rows(rows):
    """Print best non-oracle rows for quick inspection."""
    print("=" * 132)
    print("Learned Feedback Probe Summary")
    print("=" * 132)
    print(
        f"{'Noise':>6} {'Best non-oracle':<34} {'Utility':>9} {'Success':>9} "
        f"{'Perfect%':>9} {'Slots':>7} {'Probes':>8}"
    )
    for noise in sorted({row["feedback_noise_std"] for row in rows}):
        subset = [row for row in rows if row["feedback_noise_std"] == noise]
        non_oracle = [row for row in subset if row["policy"] != bandit.POLICY_ORACLE_FULL]
        best = max(non_oracle, key=lambda row: row["utility_mean"])
        print(
            f"{noise:>6.3f} {policy_label(best):<34} {best['utility_mean']:>9.3f} "
            f"{best['success_mean']:>9.3f} {best['perfect_rate']:>8.2f}% "
            f"{best['slots_mean']:>7.3f} {best['total_probe_calls_mean']:>8.2f}"
        )


def main():
    """Train and evaluate the learned feedback-conditioned selector."""
    args = parse_args()
    validate_args(args)
    bandit_args = build_bandit_args(args)
    bandit_args.device = args.device
    bandit_args.history_lr = args.history_lr
    output_prefix = resolve_output_prefix(args, bandit_args)

    print("=" * 96)
    print(
        f"Learned feedback probing: scenario={args.scenario}, train_episodes={args.train_episodes}, "
        f"eval_episodes={args.eval_episodes}, eval_noise={bandit_args.feedback_noise_std_values}, "
        f"budgets={bandit_args.probe_budgets}"
    )
    print(
        f"Behavior={args.behavior_policy} B={args.behavior_probe_budget}, "
        f"train_feedback_noise={args.train_feedback_noise_std}, output_prefix={output_prefix}"
    )
    print("=" * 96)

    train_x, train_y, _train_tx = collect_dataset(
        args,
        bandit_args,
        args.train_episodes,
        args.seed + 17,
        "train",
    )
    val_x, val_y, val_tx = collect_dataset(
        args,
        bandit_args,
        args.val_episodes,
        args.seed + 31,
        "val",
    )
    model, mean, std, train_history = train_model(args, train_x, train_y, val_x, val_y)
    write_train_history(f"{output_prefix}_train_history.csv", train_history)
    save_checkpoint(f"{output_prefix}_model.pt", model, mean, std, args, bandit_args)

    val_x_norm = np.clip((val_x - mean) / std, -10.0, 10.0).astype(np.float32)
    with torch.no_grad():
        predictions = model(torch.from_numpy(val_x_norm).to(torch.device(args.device))).cpu().numpy()
    val_rows = validation_topk_metrics(predictions, val_tx, bandit_args.probe_budgets, bandit_args.num_nodes)
    write_validation_metrics(f"{output_prefix}_val_topk.csv", val_rows)

    all_rows = []
    for feedback_noise_std in bandit_args.feedback_noise_std_values:
        rows = evaluate_suite(args, bandit_args, model, mean, std, feedback_noise_std)
        all_rows.extend(stress.attach_stress_metadata(rows, args, args.scenario, bandit_args))

    print_best_rows(all_rows)
    stress.write_csv(f"{output_prefix}.csv", all_rows)
    if not args.no_plots:
        plot_results(all_rows, output_prefix)


if __name__ == "__main__":
    main()
