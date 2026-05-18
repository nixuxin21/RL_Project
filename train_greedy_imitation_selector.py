"""训练监督学习模型模仿贪心 IRS 索引，并评估 imitation 基线的闭环表现。"""

import argparse
import csv
import os
from collections import Counter, defaultdict

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from test_env import MSAirCompEnv


# 评估阶段使用的策略名，集中定义以保证 CSV、图例、控制台输出一致。
POLICY_IMITATION = "Greedy Imitation"
POLICY_FEATURE_ARGMAX = "Feature Argmax"
POLICY_GREEDY = "Greedy IRS"


def parse_args():
    """解析 imitation 数据量、模型结构、训练超参数和输出路径。"""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a supervised IRS selector from greedy IRS labels."
    )
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--val-episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument(
        "--codebook-feature-noise-std",
        type=float,
        default=0.0,
        help=(
            "Gaussian noise std added to normalized codebook quality features "
            "during train/val and default evaluation."
        ),
    )
    parser.add_argument(
        "--eval-noise-std-values",
        default=None,
        help=(
            "Optional comma-separated feature-noise std values for evaluation. "
            "When omitted, evaluate only --codebook-feature-noise-std."
        ),
    )
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


def parse_float_list(value):
    """把形如 `0,0.05,0.1` 的字符串解析为浮点列表。"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def format_float_for_suffix(value):
    """把浮点参数转换成适合文件名的短字符串。"""
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def validate_args(args):
    """校验所有必须为正的整数参数。"""
    positive_ints = {
        "--train-episodes": args.train_episodes,
        "--val-episodes": args.val_episodes,
        "--eval-episodes": args.eval_episodes,
        "--num-nodes": args.num_nodes,
        "--num-slots": args.num_slots,
        "--num-irs-elements": args.num_irs_elements,
        "--num-codebook-states": args.num_codebook_states,
        "--hidden-size": args.hidden_size,
        "--hidden-layers": args.hidden_layers,
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")
    if args.codebook_feature_noise_std < 0:
        raise ValueError("--codebook-feature-noise-std must be non-negative")
    if args.eval_noise_std_values is None:
        args.eval_noise_std_values = [args.codebook_feature_noise_std]
    else:
        args.eval_noise_std_values = parse_float_list(args.eval_noise_std_values)
        if not args.eval_noise_std_values:
            raise ValueError("--eval-noise-std-values must contain at least one value")
        if any(value < 0 for value in args.eval_noise_std_values):
            raise ValueError("--eval-noise-std-values must be non-negative")


def ensure_parent_dir(path):
    """确保输出文件所在目录存在。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_output_prefix(args):
    """生成训练和评估产物共享的输出前缀。"""
    if args.output_prefix is not None:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix
    output_prefix = os.path.join(
        "results",
        "imitation",
        f"greedy_imitation_train{args.train_episodes}_"
        f"eval{args.eval_episodes}_seed{args.seed}",
    )
    if args.codebook_feature_noise_std > 0:
        output_prefix += f"_featnoise{format_float_for_suffix(args.codebook_feature_noise_std)}"
    ensure_parent_dir(output_prefix)
    return output_prefix


def make_env(args, codebook_feature_noise_std=None):
    """
    创建带 codebook features 的环境。

    imitation 模型和 Feature Argmax 都依赖观测尾部 C 维码本质量特征，
    因此这里固定 `include_codebook_features=True`。
    """
    noise_std = (
        args.codebook_feature_noise_std
        if codebook_feature_noise_std is None
        else codebook_feature_noise_std
    )
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode="codebook",
        include_codebook_features=True,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
        codebook_feature_noise_std=noise_std,
    )


def physical_to_action(value, low, scale):
    """将真实物理参数反向映射到环境动作范围 [-1, 1]。"""
    action_value = (value - low) / scale - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def codebook_index_to_action(index, num_codebook_states):
    """将离散 IRS 码本索引映射到环境第三维连续动作。"""
    if num_codebook_states <= 1:
        return 0.0
    action_value = 2.0 * index / (num_codebook_states - 1) - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def make_action(irs_index, args):
    """构造固定 `g_th/alpha_th` 且指定 IRS index 的三维环境动作。"""
    return np.array(
        [
            physical_to_action(args.g_th, low=0.001, scale=0.05),
            physical_to_action(args.alpha_th, low=0.05, scale=0.05),
            codebook_index_to_action(irs_index, args.num_codebook_states),
        ],
        dtype=np.float32,
    )


def make_episode_seeds(seed, episodes):
    """生成可复现的 episode 级随机种子列表。"""
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**31 - 1, size=episodes)]


def greedy_candidate(env, args):
    """
    计算当前状态下的 Greedy IRS 标签。

    排序规则和其他脚本保持一致：先最大化本槽可调度节点数；
    若并列，再选择平均功率更低的码本；最后用剩余节点平均信道增益稳定排序。
    """
    candidates = [
        env.preview_codebook_index(codebook_index, args.g_th, args.alpha_th)
        for codebook_index in range(args.num_codebook_states)
    ]

    def candidate_key(candidate):
        """贪心标签排序键：先比较调度节点数，再用低功率和剩余增益打破并列。"""
        tx_count = int(candidate["tx_this_slot"])
        power_avg = float(candidate["power_avg"])
        mean_gain = float(candidate["mean_gain_remaining"])
        power_tiebreak = -power_avg if tx_count > 0 else 0.0
        return tx_count, power_tiebreak, mean_gain

    return max(candidates, key=candidate_key)


def collect_greedy_dataset(args, episodes, seed, split_name, codebook_feature_noise_std=None):
    """
    收集 Greedy 标签数据。

    每个样本是某个时隙开始时的观测 `obs`，标签是 Greedy IRS 选择的 `irs_index`。
    收集数据时环境实际执行 Greedy 动作，因此后续状态分布也是 Greedy 轨迹下的状态分布。
    """
    noise_std = (
        args.codebook_feature_noise_std
        if codebook_feature_noise_std is None
        else codebook_feature_noise_std
    )
    env = make_env(args, codebook_feature_noise_std=noise_std)
    episode_seeds = make_episode_seeds(seed, episodes)
    observations = []
    labels = []
    metadata = []
    success_nodes = []

    for episode_idx, episode_seed in enumerate(episode_seeds, start=1):
        obs, _info = env.reset(seed=episode_seed)
        total_tx = 0
        for slot in range(1, args.num_slots + 1):
            candidate = greedy_candidate(env, args)
            label = int(candidate["irs_index"])
            observations.append(obs.astype(np.float32).copy())
            labels.append(label)
            metadata.append(
                {
                    "split": split_name,
                    "episode_idx": episode_idx,
                    "episode_seed": episode_seed,
                    "slot": slot,
                    "label": label,
                    "oracle_tx_this_slot": int(candidate["tx_this_slot"]),
                }
            )
            obs, _reward, terminated, truncated, info = env.step(make_action(label, args))
            total_tx = int(info["total_tx"])
            if terminated or truncated:
                break
        success_nodes.append(total_tx)

    features = np.asarray(observations, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    print(
        f"Collected {split_name}: episodes={episodes}, samples={len(targets)}, "
        f"feature_noise_std={noise_std:g}, "
        f"mean greedy success={np.mean(success_nodes):.3f}/{args.num_nodes}"
    )
    return features, targets, metadata


class ImitationSelector(nn.Module):
    """
    简单 MLP IRS 分类器。

    输入维度等于环境观测维度 `7 + C`，输出维度等于码本数量 C。
    这里没有使用 RNN/Transformer，目的是验证一个低复杂度监督模型能否模仿 Greedy。
    """

    def __init__(self, input_dim, num_classes, hidden_size, hidden_layers):
        """按命令行指定的隐藏层数量和宽度搭建全连接网络。"""
        super().__init__()
        layers = []
        dim = input_dim
        for _idx in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_size))
            layers.append(nn.ReLU())
            dim = hidden_size
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        """前向传播，返回每个 IRS index 的分类 logits。"""
        return self.net(obs)


def normalize_features(train_x, val_x):
    """
    使用训练集统计量标准化 train/val 特征。

    返回的 `mean/std` 会一起保存到 checkpoint，评估时必须用同一组统计量。
    """
    mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def accuracy_from_logits(logits, targets):
    """根据分类 logits 计算 top-1 标签准确率。"""
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


def evaluate_loss_acc(model, loader, criterion, device):
    """在给定 DataLoader 上计算平均交叉熵和分类准确率。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * len(batch_y)
            total_correct += int((torch.argmax(logits, dim=1) == batch_y).sum().item())
            total_count += int(len(batch_y))
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def train_classifier(args, train_x, train_y, val_x, val_y):
    """
    训练监督 IRS selector。

    训练目标是 Greedy 的 IRS index 标签准确率。注意：标签准确率不是最终通信性能的唯一指标，
    因为多个 IRS index 可能在本时隙调度节点数上等价。
    """
    train_x_norm, val_x_norm, obs_mean, obs_std = normalize_features(train_x, val_x)
    device = torch.device(args.device)
    model = ImitationSelector(
        input_dim=train_x.shape[1],
        num_classes=args.num_codebook_states,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
    ).to(device)

    train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y))
    val_dataset = TensorDataset(torch.from_numpy(val_x_norm), torch.from_numpy(val_y))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # 保存验证准确率最高的模型参数，避免最后一个 epoch 过拟合或偶然退化。
    best_state = None
    best_val_acc = -1.0
    history = []
    print("Training supervised imitation selector...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_y)
            total_correct += int((torch.argmax(logits, dim=1) == batch_y).sum().item())
            total_count += int(len(batch_y))

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 10, 1) == 0:
            print(
                f"  epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, obs_mean, obs_std, history


def predict_index(model, obs, obs_mean, obs_std, args):
    """对单个环境观测执行标准化和模型推理，返回预测 IRS index。"""
    model.eval()
    obs_norm = ((obs.reshape(1, -1).astype(np.float32) - obs_mean) / obs_std).astype(np.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(obs_norm).to(device))
    return int(torch.argmax(logits, dim=1).item())


def feature_argmax_index(obs, args):
    """无需训练的规则 baseline：选择 codebook feature 最大的索引。"""
    features = obs[7 : 7 + args.num_codebook_states]
    return int(np.argmax(features))


def evaluate_policy(
    args,
    policy_name,
    episode_seeds,
    model=None,
    obs_mean=None,
    obs_std=None,
    codebook_feature_noise_std=None,
):
    """
    在独立评估 seeds 上评估 imitation、Feature Argmax 或 Greedy。

    每个 step 都会同时计算 Greedy oracle，用于记录当前策略是否选中了同一个 index，
    以及相对 oracle 少调度了多少节点。
    """
    noise_std = (
        args.codebook_feature_noise_std
        if codebook_feature_noise_std is None
        else codebook_feature_noise_std
    )
    env = make_env(args, codebook_feature_noise_std=noise_std)
    episode_rows = []
    step_rows = []

    for episode_idx, episode_seed in enumerate(episode_seeds, start=1):
        obs, _info = env.reset(seed=episode_seed)
        total_tx = 0
        episode_reward = 0.0
        episode_power = []
        episode_energy = 0.0
        slots_used = args.num_slots

        for slot in range(1, args.num_slots + 1):
            oracle = greedy_candidate(env, args)
            oracle_index = int(oracle["irs_index"])
            # 三个策略共享同一套环境执行逻辑，只在 IRS index 选择方式上不同。
            if policy_name == POLICY_GREEDY:
                irs_index = oracle_index
            elif policy_name == POLICY_FEATURE_ARGMAX:
                irs_index = feature_argmax_index(obs, args)
            else:
                irs_index = predict_index(model, obs, obs_mean, obs_std, args)

            obs, reward, terminated, truncated, info = env.step(make_action(irs_index, args))
            total_tx = int(info["total_tx"])
            slots_used = int(info.get("slots_used", slot))
            episode_reward += float(reward)
            episode_energy += float(info["power_avg"]) * int(info["tx_this_slot"])
            if info["tx_this_slot"] > 0:
                episode_power.append(float(info["power_avg"]))

            step_rows.append(
                {
                    "policy": policy_name,
                    "noise_std": noise_std,
                    "episode_idx": episode_idx,
                    "episode_seed": episode_seed,
                    "slot": slot,
                    "irs_index": int(info["irs_index"]),
                    "oracle_irs_index": oracle_index,
                    "irs_matches_oracle": int(int(info["irs_index"]) == oracle_index),
                    "tx_this_slot": int(info["tx_this_slot"]),
                    "oracle_tx_this_slot": int(oracle["tx_this_slot"]),
                    "oracle_tx_gap": int(oracle["tx_this_slot"]) - int(info["tx_this_slot"]),
                    "total_tx": total_tx,
                    "power_avg": float(info["power_avg"]),
                    "reward": float(reward),
                }
            )

            if terminated or truncated:
                break

        episode_rows.append(
            {
                "policy": policy_name,
                "noise_std": noise_std,
                "episode_idx": episode_idx,
                "episode_seed": episode_seed,
                "success_nodes": total_tx,
                "perfect": int(total_tx == args.num_nodes),
                "missed_nodes": args.num_nodes - total_tx,
                "slots_used": slots_used,
                "episode_reward": episode_reward,
                "avg_power": float(np.mean(episode_power)) if episode_power else 0.0,
                "total_energy": episode_energy,
            }
        )

    return episode_rows, step_rows


def mean(values):
    """空列表安全均值。"""
    return float(np.mean(values)) if values else 0.0


def std(values):
    """空列表安全标准差。"""
    return float(np.std(values)) if values else 0.0


def summarize_eval(episode_rows, step_rows):
    """
    计算评估 summary 表。

    重点指标包括成功节点数、完美覆盖率、完成时隙、能耗、
    与 Greedy oracle 的 index 匹配率和调度 gap。
    """
    episodes_by_policy = defaultdict(list)
    steps_by_policy = defaultdict(list)
    for row in episode_rows:
        episodes_by_policy[row["policy"]].append(row)
    for row in step_rows:
        steps_by_policy[row["policy"]].append(row)

    summary_rows = []
    for policy in [POLICY_IMITATION, POLICY_FEATURE_ARGMAX, POLICY_GREEDY]:
        episodes = episodes_by_policy[policy]
        steps = steps_by_policy[policy]
        success = [row["success_nodes"] for row in episodes]
        slots = [row["slots_used"] for row in episodes]
        powers = [row["avg_power"] for row in episodes]
        rewards = [row["episode_reward"] for row in episodes]
        energies = [row["total_energy"] for row in episodes]
        match_values = [row["irs_matches_oracle"] for row in steps]
        gap_values = [row["oracle_tx_gap"] for row in steps]
        indices = [row["irs_index"] for row in steps]
        dominant_index = ""
        dominant_rate = 0.0
        if indices:
            dominant_index, count = Counter(indices).most_common(1)[0]
            dominant_rate = count / len(indices)

        summary_rows.append(
            {
                "policy": policy,
                "noise_std": mean([row["noise_std"] for row in episodes]),
                "episodes": len(episodes),
                "steps": len(steps),
                "success_mean": mean(success),
                "success_std": std(success),
                "perfect_rate": mean([row["perfect"] for row in episodes]) * 100.0,
                "slots_mean": mean(slots),
                "slots_std": std(slots),
                "avg_power_mean": mean(powers),
                "episode_reward_mean": mean(rewards),
                "total_energy_mean": mean(energies),
                "irs_oracle_match_rate": mean(match_values) * 100.0,
                "oracle_tx_gap_mean": mean(gap_values),
                "dominant_irs_index": dominant_index,
                "dominant_irs_rate": dominant_rate,
            }
        )
    return summary_rows


def compute_slot_stats(episode_rows, step_rows, args):
    """
    计算逐时隙平均新增调度数和累计成功数。

    对提前完成的 episode，后续时隙补 0 个新增调度节点，并保留最终累计成功数。
    """
    episodes_by_policy = defaultdict(list)
    steps_by_key = defaultdict(dict)
    for row in episode_rows:
        episodes_by_policy[row["policy"]].append(row)
    for row in step_rows:
        steps_by_key[(row["policy"], row["episode_idx"])][row["slot"]] = row

    rows = []
    for policy in [POLICY_IMITATION, POLICY_FEATURE_ARGMAX, POLICY_GREEDY]:
        episodes = episodes_by_policy[policy]
        for slot in range(1, args.num_slots + 1):
            tx_values = []
            total_values = []
            active_count = 0
            match_values = []
            gap_values = []
            for episode in episodes:
                step = steps_by_key[(policy, episode["episode_idx"])].get(slot)
                if step is None:
                    tx_values.append(0)
                    total_values.append(episode["success_nodes"])
                    continue
                active_count += 1
                tx_values.append(step["tx_this_slot"])
                total_values.append(step["total_tx"])
                match_values.append(step["irs_matches_oracle"])
                gap_values.append(step["oracle_tx_gap"])
            rows.append(
                {
                    "policy": policy,
                    "noise_std": mean([episode["noise_std"] for episode in episodes]),
                    "slot": slot,
                    "episode_count": len(episodes),
                    "active_episode_count": active_count,
                    "tx_mean_padded": mean(tx_values),
                    "total_tx_mean_padded": mean(total_values),
                    "irs_oracle_match_rate_active": mean(match_values) * 100.0,
                    "oracle_tx_gap_mean_active": mean(gap_values),
                }
            )
    return rows


def write_csv(path, rows, fieldnames):
    """按固定字段顺序写 CSV，保证后续报告读取稳定。"""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_checkpoint(path, model, obs_mean, obs_std, args):
    """
    保存 imitation 模型和标准化统计量。

    这里保存的是 PyTorch state_dict，而不是 SB3 模型；
    加载时需要按 `input_dim/hidden_size/hidden_layers` 重建同样的 MLP。
    """
    ensure_parent_dir(path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "input_dim": int(obs_mean.shape[1]),
            "num_codebook_states": args.num_codebook_states,
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "g_th": args.g_th,
            "alpha_th": args.alpha_th,
            "codebook_feature_noise_std": args.codebook_feature_noise_std,
        },
        path,
    )
    print(f"Saved: {path}")


def validation_predictions(model, val_x, val_y, obs_mean, obs_std, args):
    """批量生成验证集预测，用于计算标签准确率和混淆矩阵。"""
    model.eval()
    x_norm = ((val_x.astype(np.float32) - obs_mean) / obs_std).astype(np.float32)
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_norm), 4096):
            batch = torch.from_numpy(x_norm[start : start + 4096]).to(device)
            logits = model(batch)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    preds = np.asarray(preds, dtype=np.int64)
    accuracy = float(np.mean(preds == val_y)) if len(val_y) else 0.0
    return preds, accuracy


def plot_confusion(path, labels, preds, args):
    """
    绘制验证集混淆矩阵。

    行表示 Greedy 标签，列表示模型预测；按行归一化后可观察哪些码本标签容易混淆。
    """
    matrix = np.zeros((args.num_codebook_states, args.num_codebook_states), dtype=np.int64)
    for label, pred in zip(labels, preds):
        matrix[int(label), int(pred)] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="Prediction Rate")
    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted IRS Index")
    ax.set_ylabel("Greedy Label")
    ax.set_xticks(range(args.num_codebook_states))
    ax.set_yticks(range(args.num_codebook_states))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_latency(path, episode_rows, args):
    """绘制三种策略的完成时延分布。"""
    by_policy = defaultdict(list)
    for row in episode_rows:
        by_policy[row["policy"]].append(row["slots_used"])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, args.num_slots + 1)
    for policy in [POLICY_IMITATION, POLICY_FEATURE_ARGMAX, POLICY_GREEDY]:
        values = by_policy[policy]
        counts = Counter(values)
        rates = [counts[slot] / len(values) if values else 0.0 for slot in x]
        ax.plot(x, rates, marker="o", linewidth=1.8, label=policy)
    ax.set_title("Completion Latency Distribution")
    ax.set_xlabel("Slots Used")
    ax.set_ylabel("Episode Rate")
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_slot_curves(path, slot_rows, args):
    """绘制逐时隙新增调度数和累计成功节点数曲线。"""
    rows_by_policy = defaultdict(list)
    for row in slot_rows:
        rows_by_policy[row["policy"]].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for policy in [POLICY_IMITATION, POLICY_FEATURE_ARGMAX, POLICY_GREEDY]:
        rows = sorted(rows_by_policy[policy], key=lambda row: row["slot"])
        slots = [row["slot"] for row in rows]
        tx_means = [row["tx_mean_padded"] for row in rows]
        total_means = [row["total_tx_mean_padded"] for row in rows]
        axes[0].plot(slots, tx_means, marker="o", linewidth=1.8, label=policy)
        axes[1].plot(slots, total_means, marker="o", linewidth=1.8, label=policy)

    axes[0].set_title("Mean Scheduled Nodes per Slot")
    axes[0].set_xlabel("Slot")
    axes[0].set_ylabel("Scheduled Nodes")
    axes[1].set_title("Mean Cumulative Success Nodes")
    axes[1].set_xlabel("Slot")
    axes[1].set_ylabel("Cumulative Success Nodes")
    axes[1].axhline(args.num_nodes, color="#444444", linestyle="--", linewidth=1.0)
    for ax in axes:
        ax.set_xticks(range(1, args.num_slots + 1))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def print_eval_summary(rows):
    """在控制台打印压缩版 imitation 评估结果。"""
    print("=" * 116)
    print("Greedy Imitation Evaluation")
    print("=" * 116)
    print(
        f"{'Noise':>7} {'Policy':<18} {'Success':>9} {'Perfect%':>9} {'Slots':>8} "
        f"{'Match%':>9} {'Gap':>7} {'Dominant IRS':>13}"
    )
    for row in rows:
        print(
            f"{row['noise_std']:>7.3f} {row['policy']:<18} {row['success_mean']:>9.3f} "
            f"{row['perfect_rate']:>8.2f}% {row['slots_mean']:>8.3f} "
            f"{row['irs_oracle_match_rate']:>8.2f}% {row['oracle_tx_gap_mean']:>7.3f} "
            f"{row['dominant_irs_index']:>13}"
        )


def plot_eval_noise_sweep(path, summary_rows):
    """绘制 noise-aware imitation 在不同 feature noise 下的鲁棒性曲线。"""
    policies = []
    for row in summary_rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.get_cmap("tab10")
    colors = {policy: cmap(idx % 10) for idx, policy in enumerate(policies)}

    for policy in policies:
        rows = sorted(
            [row for row in summary_rows if row["policy"] == policy],
            key=lambda row: row["noise_std"],
        )
        x = [row["noise_std"] for row in rows]
        axes[0].plot(
            x,
            [row["success_mean"] for row in rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[1].plot(
            x,
            [row["perfect_rate"] for row in rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
        axes[2].plot(
            x,
            [row["slots_mean"] for row in rows],
            marker="o",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )

    axes[0].set_title("Success Nodes vs Feature Noise")
    axes[0].set_ylabel("Success Nodes")
    axes[1].set_title("Perfect Coverage vs Feature Noise")
    axes[1].set_ylabel("Perfect Coverage (%)")
    axes[1].set_ylim(0.0, 105.0)
    axes[2].set_title("Latency vs Feature Noise")
    axes[2].set_ylabel("Slots Used")

    for ax in axes:
        ax.set_xlabel("Codebook Feature Noise Std")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    """
    脚本入口。

    依次完成数据收集、监督训练、验证集预测、模型保存、三策略评估、CSV/图像输出。
    """
    args = parse_args()
    validate_args(args)
    output_prefix = resolve_output_prefix(args)
    print("=" * 96)
    print(
        f"Greedy imitation training: train_episodes={args.train_episodes}, "
        f"val_episodes={args.val_episodes}, eval_episodes={args.eval_episodes}, seed={args.seed}"
    )
    print(
        f"Train/val feature noise std={args.codebook_feature_noise_std}, "
        f"eval noise std values={args.eval_noise_std_values}"
    )
    print(f"Output prefix: {output_prefix}")
    print("=" * 96)

    train_x, train_y, _train_meta = collect_greedy_dataset(
        args,
        args.train_episodes,
        args.seed,
        "train",
        codebook_feature_noise_std=args.codebook_feature_noise_std,
    )
    val_x, val_y, _val_meta = collect_greedy_dataset(
        args,
        args.val_episodes,
        args.seed + 1,
        "val",
        codebook_feature_noise_std=args.codebook_feature_noise_std,
    )
    model, obs_mean, obs_std, history = train_classifier(args, train_x, train_y, val_x, val_y)
    val_preds, val_acc = validation_predictions(model, val_x, val_y, obs_mean, obs_std, args)
    print(f"Best-model validation label accuracy: {val_acc:.4f}")

    save_checkpoint(f"{output_prefix}_classifier.pt", model, obs_mean, obs_std, args)
    write_csv(
        f"{output_prefix}_train_history.csv",
        history,
        ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
    )

    eval_seeds = make_episode_seeds(args.seed + 2, args.eval_episodes)
    episode_rows = []
    step_rows = []
    summary_rows = []
    slot_rows = []
    for eval_noise_std in args.eval_noise_std_values:
        print("=" * 96)
        print(f"Evaluating feature noise std={eval_noise_std:g}")
        print("=" * 96)
        noise_episode_rows = []
        noise_step_rows = []
        for policy_name in [POLICY_IMITATION, POLICY_FEATURE_ARGMAX, POLICY_GREEDY]:
            print(f"Evaluating {policy_name}...")
            policy_episode_rows, policy_step_rows = evaluate_policy(
                args,
                policy_name,
                eval_seeds,
                model=model,
                obs_mean=obs_mean,
                obs_std=obs_std,
                codebook_feature_noise_std=eval_noise_std,
            )
            noise_episode_rows.extend(policy_episode_rows)
            noise_step_rows.extend(policy_step_rows)
        episode_rows.extend(noise_episode_rows)
        step_rows.extend(noise_step_rows)
        summary_rows.extend(summarize_eval(noise_episode_rows, noise_step_rows))
        slot_rows.extend(compute_slot_stats(noise_episode_rows, noise_step_rows, args))

    print_eval_summary(summary_rows)

    write_csv(
        f"{output_prefix}_eval_summary.csv",
        summary_rows,
        [
            "policy",
            "noise_std",
            "episodes",
            "steps",
            "success_mean",
            "success_std",
            "perfect_rate",
            "slots_mean",
            "slots_std",
            "avg_power_mean",
            "episode_reward_mean",
            "total_energy_mean",
            "irs_oracle_match_rate",
            "oracle_tx_gap_mean",
            "dominant_irs_index",
            "dominant_irs_rate",
        ],
    )
    write_csv(
        f"{output_prefix}_eval_episodes.csv",
        episode_rows,
        [
            "policy",
            "noise_std",
            "episode_idx",
            "episode_seed",
            "success_nodes",
            "perfect",
            "missed_nodes",
            "slots_used",
            "episode_reward",
            "avg_power",
            "total_energy",
        ],
    )
    write_csv(
        f"{output_prefix}_eval_slot_stats.csv",
        slot_rows,
        [
            "policy",
            "noise_std",
            "slot",
            "episode_count",
            "active_episode_count",
            "tx_mean_padded",
            "total_tx_mean_padded",
            "irs_oracle_match_rate_active",
            "oracle_tx_gap_mean_active",
        ],
    )
    write_csv(
        f"{output_prefix}_eval_steps.csv",
        step_rows,
        [
            "policy",
            "noise_std",
            "episode_idx",
            "episode_seed",
            "slot",
            "irs_index",
            "oracle_irs_index",
            "irs_matches_oracle",
            "tx_this_slot",
            "oracle_tx_this_slot",
            "oracle_tx_gap",
            "total_tx",
            "power_avg",
            "reward",
        ],
    )

    if not args.no_plots:
        plot_confusion(f"{output_prefix}_confusion.png", val_y, val_preds, args)
        if len(args.eval_noise_std_values) == 1:
            plot_latency(f"{output_prefix}_latency.png", episode_rows, args)
            plot_slot_curves(f"{output_prefix}_slot_curves.png", slot_rows, args)
        else:
            plot_eval_noise_sweep(f"{output_prefix}_eval_noise_sweep.png", summary_rows)


if __name__ == "__main__":
    main()
