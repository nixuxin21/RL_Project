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
    "validate_common_experiment_args",
    "validate_nonempty_values",
    "validate_nonnegative_values",
    "validate_positive_values",
    "validate_probe_budget_values",
    "validate_probability_values",
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


def _option_name(attribute_name):
    """Convert an argparse attribute name into its CLI option spelling."""
    return f"--{attribute_name.replace('_', '-')}"


def validate_common_experiment_args(args):
    """Validate common MS-AirComp dimensions and seed controls."""
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if hasattr(args, "seed_stride") and args.seed_stride <= 0:
        raise ValueError("--seed-stride must be positive")

    for name in ("num_nodes", "num_slots", "num_irs_elements", "num_codebook_states"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{_option_name(name)} must be positive")
    if args.num_codebook_states <= 1:
        raise ValueError("--num-codebook-states must be greater than 1")
    if args.g_th <= 0.0:
        raise ValueError("--g-th must be positive")
    if args.alpha_th <= 0.0:
        raise ValueError("--alpha-th must be positive")


def validate_nonempty_values(values, option_name):
    """Require a parsed comma-list option to contain at least one value."""
    values = list(values)
    if not values:
        raise ValueError(f"{option_name} must not be empty")
    return values


def _validate_finite_values(values, option_name):
    """Require a parsed numeric option list to contain only finite values."""
    values = validate_nonempty_values(values, option_name)
    if any(not np.isfinite(float(value)) for value in values):
        raise ValueError(f"{option_name} must contain finite values")
    return values


def validate_nonnegative_values(values, option_name):
    """Require parsed numeric values to be finite and non-negative."""
    values = _validate_finite_values(values, option_name)
    if any(float(value) < 0.0 for value in values):
        raise ValueError(f"{option_name} must be non-negative")
    return values


def validate_positive_values(values, option_name):
    """Require parsed numeric values to be finite and positive."""
    values = _validate_finite_values(values, option_name)
    if any(float(value) <= 0.0 for value in values):
        raise ValueError(f"{option_name} must be positive")
    return values


def validate_probability_values(values, option_name):
    """Require parsed numeric values to be finite probabilities."""
    values = _validate_finite_values(values, option_name)
    if any(float(value) < 0.0 or float(value) > 1.0 for value in values):
        raise ValueError(f"{option_name} must be in [0, 1]")
    return values


def validate_probe_budget_values(values, num_codebook_states, option_name):
    """Validate and normalize one or more codebook probe budgets."""
    budgets = [int(value) for value in validate_nonempty_values(values, option_name)]
    if any(value <= 0 for value in budgets):
        raise ValueError(f"{option_name} must contain positive integers")
    if any(value > num_codebook_states for value in budgets):
        raise ValueError(f"{option_name} must not exceed --num-codebook-states")
    return sorted(set(budgets))


def update_energy(episode_energy, info):
    """Accumulate the project's per-episode energy proxy."""
    return episode_energy + float(info["power_avg"]) * int(info["tx_this_slot"])
