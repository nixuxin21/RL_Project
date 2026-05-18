"""Channel-state and mismatch helpers for MS-AirComp experiments."""

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
    """Create a policy-independent RNG for execution channel drift.

    `candidate_index` makes execution drift stable for a specific IRS candidate
    regardless of whether it is built alone or inside a larger candidate batch.
    """
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
    """Apply per-slot execution drift to equivalent channels."""
    clean = np.asarray(h_total, dtype=np.complex128)
    if float(execution_error_std) <= 0.0:
        return clean.copy()
    rms = np.sqrt(np.mean(np.abs(clean) ** 2, axis=1, keepdims=True))
    noise = (rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)) / np.sqrt(2.0)
    return clean + float(execution_error_std) * np.maximum(rms, 1e-12) * noise


def capture_channel_state(env):
    """Copy the current physical channels from the environment."""
    return {
        "h_d": np.asarray(env.h_d, dtype=np.complex128).copy(),
        "h_r": np.asarray(env.h_r, dtype=np.complex128).copy(),
        "h_bs_r": np.asarray(env.h_bs_r, dtype=np.complex128).copy(),
    }


def apply_channel_state(env, state):
    """Apply a copied physical-channel state to the environment."""
    env.h_d = np.asarray(state["h_d"], dtype=np.complex128).copy()
    env.h_r = np.asarray(state["h_r"], dtype=np.complex128).copy()
    env.h_bs_r = np.asarray(state["h_bs_r"], dtype=np.complex128).copy()
    env.avg_large_scale = float(np.mean(np.abs(env.h_d) ** 2))


def temporal_rng(episode_seed, channel_rho):
    """Create a policy-independent RNG for temporal channel evolution."""
    if episode_seed is None:
        return np.random.default_rng()
    rho_tag = int(round(float(channel_rho) * 1_000_000))
    seed = (int(episode_seed) + 0xA511E9B3 + rho_tag * 0x9E3779B1) % (2**32)
    return np.random.default_rng(seed)


def complex_normal(rng, shape, scale=1.0):
    """Return circular complex Gaussian samples with the requested scale."""
    return float(scale) * (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)


def next_temporal_channel_state(prev, rho, innovation_weight, rng):
    """Generate the next AR(1) physical-channel state."""
    return {
        "h_d": rho * prev["h_d"]
        + innovation_weight * complex_normal(rng, prev["h_d"].shape, scale=0.1),
        "h_r": rho * prev["h_r"]
        + innovation_weight * complex_normal(rng, prev["h_r"].shape),
        "h_bs_r": rho * prev["h_bs_r"]
        + innovation_weight * complex_normal(rng, prev["h_bs_r"].shape),
    }


def build_temporal_channel_trace(env, args, episode_seed, channel_rho, prehistory_slots=0):
    """Generate AR(1) history states and execution states for an episode."""
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
    """Generate one AR(1) physical-channel state per execution slot."""
    _history_states, execution_states = build_temporal_channel_trace(
        env,
        args,
        episode_seed,
        channel_rho,
        prehistory_slots=0,
    )
    return execution_states


def delayed_channel_state(states, slot_idx, csi_delay_slots, history_states=None):
    """Return the stale CSI state visible to the decision policy."""
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
    """Predict the current channel mean from delayed AR(1) CSI."""
    delay = max(int(csi_delay_slots), 0)
    rho = float(channel_rho)
    direct_factor = rho**delay
    return {
        "h_d": direct_factor * delayed_state["h_d"],
        "h_r": direct_factor * delayed_state["h_r"],
        "h_bs_r": direct_factor * delayed_state["h_bs_r"],
    }


def temporal_uncertainty_std(channel_rho, csi_delay_slots, use_ar1_prediction=False):
    """Return relative CSI uncertainty induced by temporal delay."""
    delay = max(int(csi_delay_slots), 0)
    if delay == 0:
        return 0.0
    rho_delay = float(channel_rho) ** delay
    if use_ar1_prediction:
        return float(np.sqrt(max(0.0, 1.0 - rho_delay**2)))
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - rho_delay))))
