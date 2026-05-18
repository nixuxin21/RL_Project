"""训练 IRS-only SAC selector，固定传输参数，只让 agent 学习码本索引。"""

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize

from test_env import MSAirCompEnv


def physical_to_action(value, low, scale):
    """
    将真实物理值映射回 Gym/SB3 使用的 [-1, 1] 动作空间。

    环境内部的 `_decode_action` 是从动作到物理值；这里做反向映射，
    用于把固定的 `g_th` 和 `alpha_th` 填回三维动作。
    """
    action_value = (value - low) / scale - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


class FixedTransmissionActionWrapper(gym.ActionWrapper):
    """
    将动作空间从 [g_th, alpha_th, irs] 压缩为 [irs]。
    这样训练出的 SAC 只负责选择 IRS 码本，g_th/alpha_th 使用固定实验参数。
    """

    def __init__(self, env, g_th=0.001, alpha_th=0.05):
        """处理init相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
        super().__init__(env)
        self.g_action = physical_to_action(g_th, low=0.001, scale=0.05)
        self.alpha_action = physical_to_action(alpha_th, low=0.05, scale=0.05)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        """
        把一维 IRS 动作扩展回环境需要的三维动作。

        SAC 实际只输出 `action[0]` 这一维 IRS 选择信号；
        wrapper 会自动在前两维填入固定传输参数。
        """
        return np.array([self.g_action, self.alpha_action, float(action[0])], dtype=np.float32)


class TrackSuccessCallback(BaseCallback):
    """训练时把 episode 结束时的成功节点数写入 TensorBoard。"""

    def _on_step(self) -> bool:
        """从每个子环境的 `info` 中提取最终 `total_tx` 指标。"""
        dones = self.locals.get("dones", [])
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            if "total_tx" in info and env_idx < len(dones) and dones[env_idx]:
                self.logger.record("custom_metrics/final_total_tx_nodes", info["total_tx"])
        return True


def parse_args():
    """解析训练参数，并根据是否启用 codebook features 自动调整输出文件名。"""
    parser = argparse.ArgumentParser(description="Train a codebook-aware SAC IRS selector.")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="./rl_logs")
    parser.add_argument("--model-dir", default="./rl_models")
    parser.add_argument("--model-name", default="sac_codebook_aware_irs_selector")
    parser.add_argument("--stats-name", default="vec_normalize_codebook_aware_irs_selector.pkl")
    parser.add_argument("--checkpoint-prefix", default="ms_aircomp_codebook_aware_irs_selector")
    parser.add_argument(
        "--disable-codebook-features",
        action="store_true",
        help="Train the same IRS-only selector without the C-dimensional codebook quality features.",
    )
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args()

    if args.disable_codebook_features:
        # 无码本特征消融使用单独的模型/统计文件名，避免覆盖码本感知模型。
        if args.model_name == "sac_codebook_aware_irs_selector":
            args.model_name = "sac_irs_selector_no_codebook_features"
        if args.stats_name == "vec_normalize_codebook_aware_irs_selector.pkl":
            args.stats_name = "vec_normalize_irs_selector_no_codebook_features.pkl"
        if args.checkpoint_prefix == "ms_aircomp_codebook_aware_irs_selector":
            args.checkpoint_prefix = "ms_aircomp_irs_selector_no_codebook_features"

    return args


def validate_args(args):
    """校验解析后的命令行参数，尽早拒绝非法规模、预算或概率配置。"""
    for name in (
        "total_timesteps",
        "num_envs",
        "num_nodes",
        "num_slots",
        "num_irs_elements",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")
    if args.g_th <= 0.0:
        raise ValueError("--g-th must be positive")
    if args.alpha_th <= 0.0:
        raise ValueError("--alpha-th must be positive")


def make_single_env(args):
    """
    创建单个 IRS-only 训练环境。

    这里先构建原始 `MSAirCompEnv`，再包一层 `FixedTransmissionActionWrapper`，
    使 SB3 看到的动作空间只有 IRS 一维。
    """
    env = MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
        irs_phase_mode="codebook",
        include_codebook_features=not args.disable_codebook_features,
        codebook_feature_g_th=args.g_th,
        codebook_feature_alpha_th=args.alpha_th,
    )
    return FixedTransmissionActionWrapper(env, g_th=args.g_th, alpha_th=args.alpha_th)


def build_vec_env(args):
    """
    构建向量化训练环境。

    当 `num_envs > 1` 时使用多进程 `SubprocVecEnv` 加速采样；
    单环境时用 `DummyVecEnv`，便于调试。随后统一做 4 帧堆叠和归一化。
    """
    env_fns = [lambda args=args: make_single_env(args) for _ in range(args.num_envs)]
    vec_cls = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    env = vec_cls(env_fns)
    env = VecFrameStack(env, n_stack=4)
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def main():
    """训练 IRS-only SAC selector，并保存模型权重和 VecNormalize 统计文件。"""
    args = parse_args()
    validate_args(args)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    env = build_vec_env(args)
    checkpoint_callback = CheckpointCallback(
        # 多环境训练时总步数是所有环境的合计值，因此检查点保存间隔需要按环境数量缩放。
        save_freq=max(10000 // max(args.num_envs, 1), 1),
        save_path=args.model_dir,
        name_prefix=args.checkpoint_prefix,
    )
    callback_list = CallbackList([checkpoint_callback, TrackSuccessCallback()])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        ent_coef="auto",
        tensorboard_log=args.log_dir,
        verbose=1,
        seed=args.seed,
    )

    feature_mode = "without codebook features" if args.disable_codebook_features else "with codebook features"
    print(f">>> Training IRS selector {feature_mode}")
    print(f">>> obs_dim={env.observation_space.shape[0]}, action_dim={env.action_space.shape[0]}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback_list,
        progress_bar=not args.no_progress_bar,
    )

    model.save(os.path.join(args.model_dir, args.model_name))
    env.save(os.path.join(args.model_dir, args.stats_name))
    env.close()
    print(">>> Training complete")


if __name__ == "__main__":
    main()
