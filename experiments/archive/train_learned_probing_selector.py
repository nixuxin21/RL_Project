"""
Train and evaluate a learned partial probing selector.

The model does not observe full codebook quality features. It sees only the
base 7-dimensional environment observation plus a compact encoding of the
previous selected IRS index, predicts the C codebook tx-count fractions, and
probes only the top-B predicted candidates at evaluation time.
"""

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_partial_probing_sweep import (
    POLICY_RANDOM,
    POLICY_ROTATING_GRID,
    aggregate_seed_results,
    best_candidate,
    candidate_key,
    evaluate_greedy_upper_bound,
    evaluate_probe_policy,
    full_greedy_candidate,
    make_base_action,
    make_env,
    preview_indices,
    print_summary,
    select_probe_indices,
    stable_probe_rng,
    summarize_results,
    write_csv,
)
from evaluate_policy_comparison import (
    codebook_index_to_action,
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
    update_energy,
)


POLICY_LEARNED = "Learned Probe"


def parse_int_list(value):
    """Parse a comma-separated integer list such as '1,2,4,8'."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    """Parse learned probing training and evaluation parameters."""
    parser = argparse.ArgumentParser(
        description="Train a supervised low-state IRS probing selector and compare against Rotating Grid."
    )
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--val-episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-eval-seeds", type=int, default=5)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--probe-budgets", default="1,2,4,8")
    parser.add_argument("--behavior-policy", choices=["greedy", "rotating-grid", "random"], default="rotating-grid")
    parser.add_argument("--behavior-probe-budget", type=int, default=4)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args):
    """Validate sizes, budgets, and training hyperparameters."""
    positive_ints = {
        "--train-episodes": args.train_episodes,
        "--val-episodes": args.val_episodes,
        "--eval-episodes": args.eval_episodes,
        "--num-eval-seeds": args.num_eval_seeds,
        "--num-nodes": args.num_nodes,
        "--num-slots": args.num_slots,
        "--num-irs-elements": args.num_irs_elements,
        "--num-codebook-states": args.num_codebook_states,
        "--behavior-probe-budget": args.behavior_probe_budget,
        "--hidden-size": args.hidden_size,
        "--hidden-layers": args.hidden_layers,
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    budgets = parse_int_list(args.probe_budgets)
    if not budgets:
        raise ValueError("--probe-budgets must contain at least one value")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("--probe-budgets must contain positive integers")
    args.probe_budgets = sorted({min(int(budget), args.num_codebook_states) for budget in budgets})
    args.behavior_probe_budget = min(args.behavior_probe_budget, args.num_codebook_states)


def resolve_output_prefix(args):
    """Resolve shared output prefix for model, CSVs, and plots."""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    budget_label = "-".join(format_float_for_suffix(budget) for budget in args.probe_budgets)
    behavior = args.behavior_policy.replace("-", "")
    output_prefix = os.path.join(
        "results",
        "learned_probing",
        f"learned_probing_train{args.train_episodes}_eval{args.eval_episodes}_"
        f"seed{args.seed}_{behavior}_b{args.behavior_probe_budget}_evalb{budget_label}",
    )
    ensure_parent_dir(output_prefix)
    return output_prefix


def make_split_seeds(seed, episodes):
    """Generate deterministic episode seeds for a dataset split."""
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**31 - 1, size=episodes)]


def state_features(obs, previous_index, args):
    """
    Build low-dimensional model features.

    The model receives no per-codebook quality features. Previous IRS is encoded
    both linearly and circularly so the network can learn simple probing schedules.
    """
    base = obs[:7].astype(np.float32)
    if previous_index is None:
        previous_known = 0.0
        previous_norm = -1.0
        previous_sin = 0.0
        previous_cos = 0.0
    else:
        previous_known = 1.0
        previous_norm = (
            0.0
            if args.num_codebook_states <= 1
            else 2.0 * float(previous_index) / float(args.num_codebook_states - 1) - 1.0
        )
        angle = 2.0 * np.pi * float(previous_index) / float(args.num_codebook_states)
        previous_sin = float(np.sin(angle))
        previous_cos = float(np.cos(angle))
    return np.concatenate(
        [
            base,
            np.asarray([previous_known, previous_norm, previous_sin, previous_cos], dtype=np.float32),
        ]
    ).astype(np.float32)


def full_candidate_ranking(env, args):
    """Preview all codebooks and return candidates sorted by the Greedy key."""
    candidates = [
        env.preview_codebook_index(index, args.g_th, args.alpha_th)
        for index in range(args.num_codebook_states)
    ]
    return sorted(candidates, key=candidate_key, reverse=True)


def behavior_policy_name(args):
    """Return the partial probing policy name used for behavior collection."""
    if args.behavior_policy == "rotating-grid":
        return POLICY_ROTATING_GRID
    if args.behavior_policy == "random":
        return POLICY_RANDOM
    return None


def choose_behavior_candidate(env, args, ranking, slot_idx, previous_index, rng):
    """Choose the action used to advance dataset collection."""
    if args.behavior_policy == "greedy":
        return ranking[0]

    policy_name = behavior_policy_name(args)
    if policy_name is None:
        raise ValueError(f"Unknown behavior policy: {args.behavior_policy}")

    indices = select_probe_indices(
        policy_name,
        args,
        args.behavior_probe_budget,
        slot_idx,
        {"previous_index": previous_index},
        rng,
    )
    return best_candidate(preview_indices(env, args, indices))


def collect_dataset(args, episodes, seed, split_name):
    """
    Collect low-state observations and full codebook tx-count fraction targets.

    Targets are the C-dimensional normalized tx-counts produced by exact preview.
    They are used for supervised distillation, but they are not available to the
    learned probing policy at evaluation time.
    """
    env = make_env(args)
    base_action = make_base_action(args)
    states = []
    targets = []
    best_labels = []
    episode_seeds = make_split_seeds(seed, episodes)

    print(f"Collecting {split_name} learned-probing data: episodes={episodes}, seed={seed}")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        obs, _info = env.reset(seed=episode_seed)
        previous_index = None
        policy_name = behavior_policy_name(args)
        behavior_rng = (
            stable_probe_rng(episode_seed, policy_name, args.behavior_probe_budget)
            if policy_name is not None
            else None
        )
        for slot_idx in range(args.num_slots):
            ranking = full_candidate_ranking(env, args)
            target = np.zeros(args.num_codebook_states, dtype=np.float32)
            for candidate in ranking:
                target[int(candidate["irs_index"])] = float(candidate["tx_this_slot"]) / args.num_nodes

            states.append(state_features(obs, previous_index, args))
            targets.append(target)
            best_labels.append(int(ranking[0]["irs_index"]))

            behavior = choose_behavior_candidate(env, args, ranking, slot_idx, previous_index, behavior_rng)
            action = base_action.copy()
            action[2] = codebook_index_to_action(int(behavior["irs_index"]), args.num_codebook_states)
            obs, _reward, terminated, truncated, info = env.step(action)
            previous_index = int(info["irs_index"])
            if terminated or truncated:
                break

        print_progress(f"collect {split_name}", ep, episodes)

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(best_labels, dtype=np.int64),
    )


def print_progress(name, current, total):
    """Print progress at 10% intervals."""
    interval = max(total // 10, 1)
    if current % interval == 0 or current == total:
        print(f"  {name}: [{current:04d}/{total:04d}]")


def normalize_features(train_x, val_x):
    """Normalize state features with train-split statistics."""
    mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return ((train_x - mean) / std).astype(np.float32), ((val_x - mean) / std).astype(np.float32), mean, std


class ProbingRegressor(nn.Module):
    """MLP that predicts normalized tx-count quality for all codebook indices."""

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

    def forward(self, obs):
        """Predict C codebook quality scores."""
        return self.net(obs)


def topk_target_metrics(predictions, targets, budgets, num_nodes):
    """
    Compute validation metrics for selecting top-k predicted candidates.

    `oracle_hit_rate` is true if at least one predicted candidate reaches the
    max target tx-count among all codebooks for that state.
    """
    rows = []
    oracle_values = np.max(targets, axis=1)
    for budget in budgets:
        top_indices = np.argpartition(-predictions, kth=budget - 1, axis=1)[:, :budget]
        selected_values = np.take_along_axis(targets, top_indices, axis=1)
        best_selected = np.max(selected_values, axis=1)
        hit_rate = float(np.mean(np.isclose(best_selected, oracle_values, rtol=0.0, atol=1e-7)) * 100.0)
        gap = float(np.mean((oracle_values - best_selected) * num_nodes))
        rows.append(
            {
                "probe_budget": int(budget),
                "val_oracle_hit_rate": hit_rate,
                "val_oracle_tx_gap_mean": gap,
            }
        )
    return rows


def train_model(args, train_x, train_y, val_x, val_y):
    """Train the tx-count regressor."""
    train_x_norm, val_x_norm, mean, std = normalize_features(train_x, val_x)
    device = torch.device(args.device)
    model = ProbingRegressor(
        input_dim=train_x_norm.shape[1],
        output_dim=args.num_codebook_states,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
    ).to(device)

    train_ds = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y))
    val_tensor = torch.from_numpy(val_x_norm).to(device)
    val_target = torch.from_numpy(val_y).to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    history = []
    best_state = None
    best_val_loss = float("inf")

    print("Training learned probing regressor...")
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
            val_pred = model(val_tensor)
            val_loss = float(loss_fn(val_pred, val_target).item())
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
    """Predict C codebook scores for one low-state feature vector."""
    feature_norm = ((feature.reshape(1, -1).astype(np.float32) - mean) / std).astype(np.float32)
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(feature_norm).to(device)
        scores = model(tensor).detach().cpu().numpy()[0]
    return scores


def evaluate_learned_probe_policy(episode_seeds, args, budget, model, mean, std, base_action):
    """Evaluate the learned probing policy at one preview budget."""
    env = make_env(args)
    device = torch.device(args.device)
    success_nodes = []
    avg_power = []
    rewards = []
    tx_per_slot = []
    slots_used = []
    total_energy = []
    preview_calls_per_slot = []
    candidate_count_mean = []
    oracle_match_rate = []
    oracle_tx_gap_mean = []

    print(f"Running {POLICY_LEARNED} (B={budget})...")
    for ep, episode_seed in enumerate(episode_seeds, start=1):
        obs, _info = env.reset(seed=episode_seed)
        previous_index = None
        episode_power = []
        episode_reward = 0.0
        episode_tx = []
        episode_slots = args.num_slots
        episode_energy = 0.0
        episode_candidate_counts = []
        episode_oracle_matches = []
        episode_oracle_gaps = []
        total_tx = 0

        for _slot_idx in range(args.num_slots):
            oracle = full_greedy_candidate(env, args)
            feature = state_features(obs, previous_index, args)
            scores = predict_scores(model, feature, mean, std, device)
            budget_count = min(int(budget), args.num_codebook_states)
            top_indices = np.argsort(-scores)[:budget_count]
            candidates = preview_indices(env, args, top_indices)
            selected = best_candidate(candidates)
            selected_index = int(selected["irs_index"])

            episode_candidate_counts.append(len(candidates))
            episode_oracle_matches.append(float(selected_index == int(oracle["irs_index"])))
            episode_oracle_gaps.append(
                max(0.0, float(oracle["tx_this_slot"]) - float(selected["tx_this_slot"]))
            )

            action = base_action.copy()
            action[2] = codebook_index_to_action(selected_index, args.num_codebook_states)
            obs, reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            episode_tx.append(int(info["tx_this_slot"]))
            episode_reward += float(reward)
            episode_slots = int(info.get("slots_used", len(episode_tx)))
            episode_energy = update_energy(episode_energy, info)
            previous_index = selected_index

            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            if terminated or truncated:
                break

        success_nodes.append(total_tx)
        avg_power.append(float(np.mean(episode_power)) if episode_power else 0.0)
        rewards.append(float(episode_reward))
        tx_per_slot.append(episode_tx)
        slots_used.append(int(episode_slots))
        total_energy.append(float(episode_energy))
        active_slots = max(len(episode_candidate_counts), 1)
        preview_calls_per_slot.append(float(sum(episode_candidate_counts)) / active_slots)
        candidate_count_mean.append(float(np.mean(episode_candidate_counts)) if episode_candidate_counts else 0.0)
        oracle_match_rate.append(float(np.mean(episode_oracle_matches)) if episode_oracle_matches else 0.0)
        oracle_tx_gap_mean.append(float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0)

        print_eval_progress(POLICY_LEARNED, budget, ep, args.eval_episodes, success_nodes, args.num_nodes)

    return {
        "name": POLICY_LEARNED,
        "probe_budget": int(budget),
        "success_nodes": np.asarray(success_nodes, dtype=float),
        "avg_power": np.asarray(avg_power, dtype=float),
        "episode_reward": np.asarray(rewards, dtype=float),
        "tx_per_slot": tx_per_slot,
        "slots_used": np.asarray(slots_used, dtype=float),
        "total_energy": np.asarray(total_energy, dtype=float),
        "decision_preview_calls_per_slot": np.asarray(preview_calls_per_slot, dtype=float),
        "candidate_count_mean": np.asarray(candidate_count_mean, dtype=float),
        "oracle_match_rate": np.asarray(oracle_match_rate, dtype=float),
        "oracle_tx_gap_mean": np.asarray(oracle_tx_gap_mean, dtype=float),
    }


def print_eval_progress(name, budget, ep, episodes, success_nodes, num_nodes):
    """Print evaluation progress at 10% intervals."""
    interval = max(episodes // 10, 1)
    if ep % interval == 0 or ep == episodes:
        recent = np.mean(success_nodes[-interval:])
        print(f"  {name} B={budget}: [{ep:04d}/{episodes:04d}] recent success {recent:.2f}/{num_nodes}")


def write_train_history(path, history):
    """Write training loss history."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved: {path}")


def write_validation_metrics(path, rows):
    """Write validation top-k target metrics."""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["probe_budget", "val_oracle_hit_rate", "val_oracle_tx_gap_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_checkpoint(path, model, mean, std, args):
    """Save learned probing regressor and normalization statistics."""
    ensure_parent_dir(path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_mean": mean,
            "obs_std": std,
            "input_dim": int(mean.shape[1]),
            "num_codebook_states": args.num_codebook_states,
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "behavior_policy": args.behavior_policy,
            "behavior_probe_budget": args.behavior_probe_budget,
        },
        path,
    )
    print(f"Saved: {path}")


def plot_eval_results(rows, args, output_prefix):
    """Plot learned probing against Rotating Grid and Random probing."""
    policies = []
    for row in rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.get_cmap("tab10")
    colors = {policy: cmap(idx % 10) for idx, policy in enumerate(policies)}
    for policy in policies:
        policy_rows = sorted(
            [row for row in rows if row["policy"] == policy],
            key=lambda row: row["probe_budget"],
        )
        x = [row["probe_budget"] for row in policy_rows]
        axes[0].plot(
            x,
            [row["perfect_rate"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[1].plot(
            x,
            [row["slots_mean"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[2].plot(
            x,
            [row["oracle_tx_gap_mean"] for row in policy_rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )

    axes[0].set_title("Perfect Coverage vs Probe Budget")
    axes[0].set_ylabel("Perfect Episodes (%)")
    axes[0].set_ylim(0.0, 103.0)
    axes[1].set_title("Latency vs Probe Budget")
    axes[1].set_ylabel("Slots Used")
    axes[1].set_ylim(0.0, args.num_slots + 1)
    axes[2].set_title("Per-Slot Oracle Tx Gap")
    axes[2].set_ylabel("Missed Tx Count")
    for ax in axes:
        ax.set_xlabel("Preview Budget B")
        ax.set_xticks(sorted({row["probe_budget"] for row in rows}))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = f"{output_prefix}_eval.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def run_eval_suite(args, model, mean, std, output_prefix):
    """Evaluate learned probing and key non-learning baselines."""
    base_action = make_base_action(args)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.episodes = args.eval_episodes
    eval_args.num_seeds = args.num_eval_seeds
    run_seeds = make_run_seeds(eval_args)
    episode_seed_sets = [make_episode_seeds(eval_args, run_seed) for run_seed in run_seeds]

    rows = []
    for budget in args.probe_budgets:
        learned_seed_results = []
        rotating_seed_results = []
        random_seed_results = []
        for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
            print(f"Eval budget B={budget}, run [{run_idx}/{len(run_seeds)}], seed={run_seeds[run_idx - 1]}")
            learned_seed_results.append(
                [evaluate_learned_probe_policy(episode_seeds, eval_args, budget, model, mean, std, base_action)]
            )
            rotating_seed_results.append(
                [evaluate_probe_policy(episode_seeds, eval_args, budget, POLICY_ROTATING_GRID, base_action)]
            )
            random_seed_results.append(
                [evaluate_probe_policy(episode_seeds, eval_args, budget, POLICY_RANDOM, base_action)]
            )
        rows.extend(summarize_results(eval_args, aggregate_seed_results(learned_seed_results)))
        rows.extend(summarize_results(eval_args, aggregate_seed_results(rotating_seed_results)))
        rows.extend(summarize_results(eval_args, aggregate_seed_results(random_seed_results)))

    greedy_seed_results = []
    for run_idx, episode_seeds in enumerate(episode_seed_sets, start=1):
        print(f"Eval full Greedy, run [{run_idx}/{len(run_seeds)}], seed={run_seeds[run_idx - 1]}")
        greedy_seed_results.append([evaluate_greedy_upper_bound(eval_args, episode_seeds, base_action)])
    rows.extend(summarize_results(eval_args, aggregate_seed_results(greedy_seed_results)))

    print_summary(rows)
    write_csv(f"{output_prefix}_eval_summary.csv", rows)
    if not args.no_plots:
        plot_eval_results(rows, eval_args, output_prefix)
    return rows


def main():
    """Train and evaluate learned probing."""
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)

    print("=" * 96)
    print(
        f"Learned probing: train={args.train_episodes}, val={args.val_episodes}, "
        f"eval={args.eval_episodes}, budgets={args.probe_budgets}"
    )
    print(
        f"Behavior policy={args.behavior_policy}, behavior_probe_budget={args.behavior_probe_budget}, "
        f"output_prefix={output_prefix}"
    )
    print("=" * 96)

    train_x, train_y, _train_labels = collect_dataset(args, args.train_episodes, args.seed, "train")
    val_x, val_y, _val_labels = collect_dataset(args, args.val_episodes, args.seed + 1, "val")
    model, mean, std, history = train_model(args, train_x, train_y, val_x, val_y)

    device = torch.device(args.device)
    val_x_norm = ((val_x.astype(np.float32) - mean) / std).astype(np.float32)
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(val_x_norm).to(device)).detach().cpu().numpy()
    val_rows = topk_target_metrics(val_pred, val_y, args.probe_budgets, args.num_nodes)

    write_train_history(f"{output_prefix}_train_history.csv", history)
    write_validation_metrics(f"{output_prefix}_val_topk.csv", val_rows)
    save_checkpoint(f"{output_prefix}_regressor.pt", model, mean, std, args)
    run_eval_suite(args, model, mean, std, output_prefix)


if __name__ == "__main__":
    main()
