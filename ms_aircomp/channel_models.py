"""封装执行信道错配所需的信道快照、信道漂移、temporal AR(1) trace 和 stale CSI 工具。"""

import numpy as np

__all__ = [
    "apply_channel_state",
    "ar1_predict_channel_state",
    "build_temporal_channel_states",
    "build_temporal_channel_trace",
    "capture_channel_state",
    "complex_normal",
    "delayed_channel_state",
    "drift_channels",
    "execution_rng",
    "temporal_rng",
    "temporal_uncertainty_std",
]


def execution_rng(
    episode_seed,
    execution_error_std,
    slot_idx,
    no_irs=False,
    candidate_index=None,
):
    """根据 episode seed、执行误差、时隙和候选索引生成确定性随机数流，保证执行漂移可复现。"""
    if episode_seed is None:
        return np.random.default_rng()
    error_tag = int(round(float(execution_error_std) * 1_000_000))
    mode_tag = 0x4CF5AD43 if no_irs else 0
    index_tag = 0 if candidate_index is None else int(candidate_index) + 2
    seed = (
        int(episode_seed)
        + 0xD1B54A32
        + error_tag * 0x9E3779B1
        + int(slot_idx) * 0x85EBCA6B
        + mode_tag
        + index_tag * 0x27D4EB2F
    ) % (2**32)
    return np.random.default_rng(seed)


def drift_channels(h_total, execution_error_std, rng):
    """在等效信道上加入按信道尺度归一化的复高斯扰动，用来模拟执行阶段信道漂移。"""
    clean = np.asarray(h_total, dtype=np.complex128)
    if float(execution_error_std) <= 0.0:
        return clean.copy()
    rms = np.sqrt(np.mean(np.abs(clean) ** 2, axis=1, keepdims=True))
    noise = (rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)) / np.sqrt(2.0)
    return clean + float(execution_error_std) * np.maximum(rms, 1e-12) * noise


def capture_channel_state(env):
    """复制环境当前的物理信道快照，后续可以在 stale/current 信道之间切换。"""
    return {
        "h_d": np.asarray(env.h_d, dtype=np.complex128).copy(),
        "h_r": np.asarray(env.h_r, dtype=np.complex128).copy(),
        "h_bs_r": np.asarray(env.h_bs_r, dtype=np.complex128).copy(),
    }


def apply_channel_state(env, state):
    """把保存的物理信道快照写回环境，并同步更新观测中使用的大尺度统计量。"""
    env.h_d = np.asarray(state["h_d"], dtype=np.complex128).copy()
    env.h_r = np.asarray(state["h_r"], dtype=np.complex128).copy()
    env.h_bs_r = np.asarray(state["h_bs_r"], dtype=np.complex128).copy()
    env.avg_large_scale = float(np.mean(np.abs(env.h_d) ** 2))


def temporal_rng(episode_seed, channel_rho):
    """根据 episode seed 和 AR(1) 相关系数生成时序信道专用随机数流。"""
    if episode_seed is None:
        return np.random.default_rng()
    rho_tag = int(round(float(channel_rho) * 1_000_000))
    seed = (int(episode_seed) + 0xA511E9B3 + rho_tag * 0x9E3779B1) % (2**32)
    return np.random.default_rng(seed)


def complex_normal(rng, shape, scale=1.0):
    """生成复高斯噪声矩阵，实部和虚部按相同尺度独立采样。"""
    return float(scale) * (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)


def next_temporal_channel_state(prev, rho, innovation_weight, rng):
    """根据 AR(1) 递推公式生成下一时刻信道状态，保持信道相关性和创新噪声。"""
    return {
        "h_d": rho * prev["h_d"]
        + innovation_weight * complex_normal(rng, prev["h_d"].shape, scale=0.1),
        "h_r": rho * prev["h_r"]
        + innovation_weight * complex_normal(rng, prev["h_r"].shape),
        "h_bs_r": rho * prev["h_bs_r"]
        + innovation_weight * complex_normal(rng, prev["h_bs_r"].shape),
    }


def build_temporal_channel_trace(env, args, episode_seed, channel_rho, prehistory_slots=0):
    """构造包含 prehistory 和执行时隙的时序信道轨迹，用于 stale CSI 延迟场景。"""
    rho = float(channel_rho)
    innovation_weight = np.sqrt(max(0.0, 1.0 - rho**2))
    rng = temporal_rng(episode_seed, rho)
    prehistory_slots = max(0, int(prehistory_slots))
    total_slots = prehistory_slots + int(args.num_slots)
    states = [capture_channel_state(env)]
    for _slot_idx in range(1, total_slots):
        prev = states[-1]
        states.append(next_temporal_channel_state(prev, rho, innovation_weight, rng))
    history_states = states[:prehistory_slots]
    execution_states = states[prehistory_slots:]
    return history_states, execution_states


def build_temporal_channel_states(env, args, episode_seed, channel_rho):
    """构造不含 prehistory 的执行时序信道状态列表，供普通 temporal 场景使用。"""
    _history_states, execution_states = build_temporal_channel_trace(
        env,
        args,
        episode_seed,
        channel_rho,
        prehistory_slots=0,
    )
    return execution_states


def delayed_channel_state(states, slot_idx, csi_delay_slots, history_states=None):
    """按 CSI delay 取出策略可见的 stale 信道；delay 超过当前时隙时从 prehistory 中补足。"""
    slot_idx = int(slot_idx)
    delay = max(0, int(csi_delay_slots))
    if delay > slot_idx:
        history_states = [] if history_states is None else list(history_states)
        history_offset = delay - slot_idx
        if history_offset <= len(history_states):
            return history_states[-history_offset]
    delayed_idx = max(0, slot_idx - delay)
    delayed_idx = min(delayed_idx, len(states) - 1)
    return states[delayed_idx]


def ar1_predict_channel_state(delayed_state, channel_rho, csi_delay_slots):
    """用 AR(1) 相关系数把 delayed CSI 预测到当前时隙，作为预测型 stale CSI。"""
    delay = max(int(csi_delay_slots), 0)
    rho = float(channel_rho)
    direct_factor = rho**delay
    return {
        "h_d": direct_factor * delayed_state["h_d"],
        "h_r": direct_factor * delayed_state["h_r"],
        "h_bs_r": direct_factor * delayed_state["h_bs_r"],
    }


def temporal_uncertainty_std(channel_rho, csi_delay_slots, use_ar1_prediction=False):
    """根据相关系数和 delay 估计 stale CSI 不确定性，供风险和 temporal 策略使用。"""
    delay = max(int(csi_delay_slots), 0)
    if delay == 0:
        return 0.0
    rho_delay = float(channel_rho) ** delay
    if use_ar1_prediction:
        return float(np.sqrt(max(0.0, 1.0 - rho_delay**2)))
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - rho_delay))))
