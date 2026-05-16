"""Small shared utilities for MS-AirComp experiment scripts.

This module is intentionally conservative: it contains only stateless helpers
that are already duplicated across scripts. Keeping these helpers here makes
new experiment scripts easier to read without changing any policy logic.
"""

import os

import numpy as np

__all__ = [
    "action_to_alpha",
    "codebook_index_to_action",
    "ensure_parent_dir",
    "format_float_for_suffix",
    "make_episode_seed_list",
    "make_episode_seeds",
    "make_run_seeds",
    "physical_to_action",
    "update_energy",
]


def ensure_parent_dir(path):
    """Create the parent directory for an output path when needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def physical_to_action(value, low, scale):
    """Map a physical scalar value back into the environment action range."""
    action_value = (value - low) / scale - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def codebook_index_to_action(index, num_codebook_states):
    """Map a discrete IRS codebook index into the continuous action slot."""
    if num_codebook_states <= 1:
        return 0.0
    action_value = 2.0 * index / (num_codebook_states - 1) - 1.0
    return float(np.clip(action_value, -1.0, 1.0))


def action_to_alpha(action):
    """Decode the AirComp target amplitude from a full environment action."""
    return float(0.05 + (float(action[1]) + 1.0) * 0.05)


def format_float_for_suffix(value):
    """Convert a numeric parameter to a compact filename-safe suffix."""
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def make_run_seeds(args):
    """Generate deterministic outer run seeds from common CLI arguments."""
    if args.num_seeds <= 1:
        return [None if args.seed < 0 else args.seed]

    if args.seed < 0:
        rng = np.random.default_rng()
        return [int(seed) for seed in rng.integers(0, 2**31 - 1, size=args.num_seeds)]

    return [args.seed + idx * args.seed_stride for idx in range(args.num_seeds)]


def make_episode_seeds(args, run_seed):
    """Generate episode-level seeds for one outer run seed."""
    seed = None if run_seed is None else int(run_seed)
    return make_episode_seed_list(seed, args.episodes)


def make_episode_seed_list(seed, episodes):
    """Generate a deterministic episode seed list from a scalar seed."""
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, 2**31 - 1, size=episodes)]


def update_energy(episode_energy, info):
    """Accumulate the project's per-episode energy proxy."""
    return episode_energy + float(info["power_avg"]) * int(info["tx_this_slot"])
