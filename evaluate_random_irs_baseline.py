"""
随机 IRS 相位 baseline 评估脚本。

该 baseline 固定 `g_th/alpha_th`，但每个时隙让 IRS 相位随机生成，
用于回答一个基础问题：性能提升究竟来自“动态改变 IRS”这个机制本身，
还是来自策略真的选到了更好的码本/波束。

和 `evaluate_policy_comparison.py` 中的 Random IRS 一样，这里使用 `irs_phase_mode="random"`，
但本脚本更轻量，适合快速单独生成随机 IRS 的图和统计结果。
"""

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from test_env import MSAirCompEnv


def physical_to_action(value, low, scale):
    """将真实物理参数反向映射到环境动作范围 [-1, 1]。"""
    action_value = (value - low) / scale - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def parse_args():
    """解析随机 IRS baseline 的 episode 数、环境规模、固定传输参数和输出路径。"""
    parser = argparse.ArgumentParser(
        description="Evaluate a random IRS phase baseline for the MS-AirComp environment."
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of Monte Carlo episodes.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed. Use -1 for unseeded runs.")
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001, help="Fixed channel admission threshold.")
    parser.add_argument("--alpha-th", type=float, default=0.05, help="Fixed AirComp target amplitude.")
    parser.add_argument(
        "--output",
        default=os.path.join("results", "policy_comparison", "random_irs_baseline_results.png"),
        help="Path for the saved baseline plot.",
    )
    parser.add_argument("--show", action="store_true", help="Show the Matplotlib window after saving.")
    return parser.parse_args()


def validate_args(args):
    """校验解析后的命令行参数，尽早拒绝非法规模、预算或概率配置。"""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    for name in ("num_nodes", "num_slots", "num_irs_elements"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")
    if args.g_th <= 0.0:
        raise ValueError("--g-th must be positive")
    if args.alpha_th <= 0.0:
        raise ValueError("--alpha-th must be positive")


def run_random_irs_baseline(args):
    """
    执行随机 IRS baseline 的 Monte Carlo 评估。

    每个 episode 会重置信道；每个 step 中环境内部都会重新生成随机 IRS 相位。
    动作中的第三维在 `irs_phase_mode="random"` 下不会被当作码本索引使用，
    这里只保留 0.0 占位，保证动作维度和原环境一致。
    """
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode="random",
    )

    # 前两维动作固定为用户给定的物理参数，第三维在 random 模式下只是占位。
    g_action = physical_to_action(args.g_th, low=0.001, scale=0.05)
    alpha_action = physical_to_action(args.alpha_th, low=0.05, scale=0.05)
    action = np.array([g_action, alpha_action, 0.0], dtype=np.float32)

    # `seed=-1` 表示完全不固定随机性；否则先生成 episode 级随机种子，
    # 让不同 episode 信道不同，同时整个实验仍可复现。
    seed = None if args.seed < 0 else args.seed
    seed_rng = np.random.default_rng(seed)

    success_nodes_history = []
    avg_mse_history = []
    avg_power_history = []
    tx_per_slot_history = []
    noise_var = 1e-9

    print("=" * 64)
    print(f"Random IRS Phase Baseline ({args.episodes} episodes)")
    print("=" * 64)
    print(f"g_th={args.g_th:.6f}, alpha_th={args.alpha_th:.6f}, seed={seed}")

    for ep in range(args.episodes):
        episode_seed = None if seed is None else int(seed_rng.integers(0, 2**31 - 1))
        env.reset(seed=episode_seed)

        mse_list = []
        power_list = []
        tx_this_episode = []
        total_tx = 0

        for _slot in range(args.num_slots):
            _obs, _reward, terminated, truncated, info = env.step(action)
            total_tx = int(info["total_tx"])
            tx_this_episode.append(int(info["tx_this_slot"]))

            mse_list.append(noise_var / (args.alpha_th**2))
            if info["tx_this_slot"] > 0:
                power_list.append(float(info["power_avg"]))

            if terminated or truncated:
                break

        success_nodes_history.append(total_tx)
        avg_mse_history.append(float(np.mean(mse_list)))
        avg_power_history.append(float(np.mean(power_list)) if power_list else 0.0)
        tx_per_slot_history.append(tx_this_episode)

        if (ep + 1) % max(args.episodes // 10, 1) == 0:
            recent_success = np.mean(success_nodes_history[-max(args.episodes // 10, 1):])
            print(
                f"Progress [{ep + 1:04d}/{args.episodes:04d}] | "
                f"recent success: {recent_success:.2f}/{args.num_nodes}"
            )

    summarize_results(success_nodes_history, avg_mse_history, avg_power_history, args)
    output_parent = os.path.dirname(os.path.abspath(args.output))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    plot_results(success_nodes_history, avg_power_history, args)

    return {
        "success_nodes": success_nodes_history,
        "avg_mse": avg_mse_history,
        "avg_power": avg_power_history,
        "tx_per_slot": tx_per_slot_history,
    }


def summarize_results(success_nodes_history, avg_mse_history, avg_power_history, args):
    """把 episode 级统计量压缩为控制台可读的总体指标。"""
    success = np.array(success_nodes_history)
    avg_mse = np.array(avg_mse_history)
    avg_power = np.array(avg_power_history)

    print("=" * 64)
    print("Random IRS Phase Baseline Summary")
    print("=" * 64)
    print(f"Average success nodes : {np.mean(success):.2f} +/- {np.std(success):.2f} / {args.num_nodes}")
    print(f"Perfect coverage rate : {np.mean(success == args.num_nodes) * 100:.1f}%")
    print(f"Average theoretical MSE: {np.mean(avg_mse):.4e}")
    print(f"Average transmit power : {np.mean(avg_power):.4e} W")


def plot_results(success_nodes_history, avg_power_history, args):
    """
    生成随机 IRS baseline 的三联图。

    左图显示 episode 级成功节点数波动，中图显示成功节点数 CDF，
    右图显示平均功率分布，用于观察随机相位是否带来较大的能耗方差。
    """
    success = np.array(success_nodes_history)
    avg_power = np.array(avg_power_history)
    sorted_success = np.sort(success)
    success_cdf = np.arange(1, len(sorted_success) + 1) / len(sorted_success)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(np.arange(1, len(success) + 1), success, color="#1f77b4", linewidth=1.2)
    axes[0].axhline(args.num_nodes, color="#d62728", linestyle="--", linewidth=1.0)
    axes[0].set_title("Random IRS Baseline: Success Nodes")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Success Nodes")
    axes[0].set_ylim(0, args.num_nodes + 2)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(sorted_success, success_cdf, color="#2ca02c", linewidth=1.5)
    axes[1].set_title("Success Nodes CDF")
    axes[1].set_xlabel("Success Nodes")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_xlim(max(0, np.min(success) - 1), args.num_nodes + 1)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].hist(avg_power, bins=30, color="#9467bd", alpha=0.85)
    axes[2].set_title("Average Power Distribution")
    axes[2].set_xlabel("Average Power (W)")
    axes[2].set_ylabel("Episodes")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"Plot saved to: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    cli_args = parse_args()
    validate_args(cli_args)
    run_random_irs_baseline(cli_args)
