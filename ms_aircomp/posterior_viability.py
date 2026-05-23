"""Posterior viability model and count-conditioned invitation refinement.

The deployable posterior model treats each remaining node's execution-channel
viability as a Bernoulli variable inferred from stale effective-channel gain.
For temporal AR(1) CSI, the reusable matrix estimator below supports:

* ``analytic``: a conditional-moment complex-Gaussian approximation for each
  effective device/IRS channel.
* ``monte_carlo``: physical-channel samples drawn from the AR(1) conditional
  distribution and then evaluated by the simulator's feasibility threshold.

The exact cascaded IRS channel contains products of conditional complex
Gaussians, so the analytic mode is an approximation; Monte Carlo is the
correct baseline for that nonlinear channel law.
"""

import math

import numpy as np

from ms_aircomp.channel_models import temporal_uncertainty_std

__all__ = [
    "POSTERIOR_INVITATION_RULE_MEAN_TOPK",
    "POSTERIOR_INVITATION_RULE_TOP_Y",
    "POSTERIOR_INVITATION_RULE_THRESHOLD",
    "POSTERIOR_TARGET_POLICY_CUMULATIVE_PROBABILITY",
    "POSTERIOR_TARGET_POLICY_EXPECTED_COUNT",
    "POSTERIOR_TARGET_POLICY_FAILURE_PENALTY",
    "POSTERIOR_TARGET_POLICY_FIXED_Y",
    "POSTERIOR_TARGET_POLICY_MISSED_PENALTY",
    "POSTERIOR_MEAN_MODE_AR1_PREDICT",
    "POSTERIOR_MEAN_MODE_STALE",
    "POSTERIOR_MODE_ANALYTIC",
    "POSTERIOR_MODE_MONTE_CARLO",
    "VALID_POSTERIOR_INVITATION_RULES",
    "VALID_POSTERIOR_MEAN_MODES",
    "VALID_POSTERIOR_MODES",
    "VALID_POSTERIOR_TARGET_POLICIES",
    "apply_count_conditioned_invitation_refinement",
    "attach_posterior_viability",
    "bernoulli_entropy",
    "count_conditioned_marginals",
    "count_conditioned_posterior_details",
    "poisson_binomial_pmf",
    "posterior_guided_sparse_indices",
    "posterior_invitation_target_count",
    "posterior_probability_coverage_stats",
    "posterior_summary_from_matrix",
    "posterior_viability_matrix",
    "posterior_viability_probabilities",
    "validate_posterior_invitation_rule",
    "validate_posterior_mean_mode",
    "validate_posterior_mode",
    "validate_posterior_target_policy",
]

POSTERIOR_MODE_ANALYTIC = "analytic"
POSTERIOR_MODE_MONTE_CARLO = "monte_carlo"
VALID_POSTERIOR_MODES = (
    POSTERIOR_MODE_ANALYTIC,
    POSTERIOR_MODE_MONTE_CARLO,
)

POSTERIOR_MEAN_MODE_AR1_PREDICT = "ar1_predict"
POSTERIOR_MEAN_MODE_STALE = "stale"
VALID_POSTERIOR_MEAN_MODES = (
    POSTERIOR_MEAN_MODE_AR1_PREDICT,
    POSTERIOR_MEAN_MODE_STALE,
)

POSTERIOR_INVITATION_RULE_MEAN_TOPK = "posterior_mean_topk"
POSTERIOR_INVITATION_RULE_TOP_Y = "posterior_top_y"
POSTERIOR_INVITATION_RULE_THRESHOLD = "probability_threshold"
VALID_POSTERIOR_INVITATION_RULES = (
    POSTERIOR_INVITATION_RULE_MEAN_TOPK,
    POSTERIOR_INVITATION_RULE_TOP_Y,
    POSTERIOR_INVITATION_RULE_THRESHOLD,
)

POSTERIOR_TARGET_POLICY_FIXED_Y = "invite_fixed_cardinality_y"
POSTERIOR_TARGET_POLICY_EXPECTED_COUNT = "invite_top_expected_count"
POSTERIOR_TARGET_POLICY_CUMULATIVE_PROBABILITY = "invite_until_cumulative_probability"
POSTERIOR_TARGET_POLICY_FAILURE_PENALTY = "invite_with_failure_penalty"
POSTERIOR_TARGET_POLICY_MISSED_PENALTY = "invite_with_missed_penalty"
VALID_POSTERIOR_TARGET_POLICIES = (
    POSTERIOR_TARGET_POLICY_FIXED_Y,
    POSTERIOR_TARGET_POLICY_EXPECTED_COUNT,
    POSTERIOR_TARGET_POLICY_CUMULATIVE_PROBABILITY,
    POSTERIOR_TARGET_POLICY_FAILURE_PENALTY,
    POSTERIOR_TARGET_POLICY_MISSED_PENALTY,
)


def validate_posterior_mode(posterior_mode):
    """Normalize and validate the posterior viability estimator mode."""
    mode = str(posterior_mode).strip()
    if mode not in VALID_POSTERIOR_MODES:
        valid = ", ".join(VALID_POSTERIOR_MODES)
        raise ValueError(f"unknown posterior mode {mode!r}; expected one of: {valid}")
    return mode


def validate_posterior_mean_mode(mean_mode):
    """Normalize and validate the stale-to-current posterior mean approximation."""
    mode = str(mean_mode).strip()
    if mode not in VALID_POSTERIOR_MEAN_MODES:
        valid = ", ".join(VALID_POSTERIOR_MEAN_MODES)
        raise ValueError(f"unknown posterior mean mode {mode!r}; expected one of: {valid}")
    return mode


def validate_posterior_invitation_rule(invitation_rule):
    """Normalize and validate the rule that converts posterior marginals to an invite mask."""
    rule = str(invitation_rule).strip()
    if rule not in VALID_POSTERIOR_INVITATION_RULES:
        valid = ", ".join(VALID_POSTERIOR_INVITATION_RULES)
        raise ValueError(f"unknown posterior invitation rule {rule!r}; expected one of: {valid}")
    return rule


def validate_posterior_target_policy(target_policy):
    """Normalize and validate how posterior marginals choose invitation count."""
    policy = str(target_policy).strip()
    if policy not in VALID_POSTERIOR_TARGET_POLICIES:
        valid = ", ".join(VALID_POSTERIOR_TARGET_POLICIES)
        raise ValueError(f"unknown posterior target policy {policy!r}; expected one of: {valid}")
    return policy


def bernoulli_entropy(probabilities):
    """Return Bernoulli entropy in nats for an array of success probabilities."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    return -probabilities * np.log(probabilities) - (1.0 - probabilities) * np.log(
        1.0 - probabilities
    )


def _posterior_mean_factor(channel_rho, csi_delay_slots, mean_mode):
    mean_mode = validate_posterior_mean_mode(mean_mode)
    if mean_mode == POSTERIOR_MEAN_MODE_AR1_PREDICT:
        return float(channel_rho) ** max(int(csi_delay_slots), 0)
    return 1.0


def _posterior_temporal_std(channel_rho, csi_delay_slots, mean_mode):
    mean_mode = validate_posterior_mean_mode(mean_mode)
    return temporal_uncertainty_std(
        channel_rho,
        csi_delay_slots,
        use_ar1_prediction=mean_mode == POSTERIOR_MEAN_MODE_AR1_PREDICT,
    )


def _clip_probabilities(probabilities, clip_eps):
    probabilities = np.asarray(probabilities, dtype=float)
    eps = float(clip_eps)
    if eps <= 0.0:
        return np.clip(probabilities, 0.0, 1.0)
    if eps >= 0.5:
        raise ValueError("posterior_clip_eps must be smaller than 0.5")
    return np.clip(probabilities, eps, 1.0 - eps)


def _complex_normal(rng, shape, scale=1.0):
    return float(scale) * (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ) / np.sqrt(2.0)


def _noncentral_exponential_sf_scalar(mean_power, noise_power, threshold):
    """Approximate P(|CN(mu, sigma^2)|^2 >= threshold) without SciPy.

    ``noise_power`` is E|noise|^2. For moderate noncentrality this uses the
    Poisson mixture of central chi-square tails. For very large noncentrality,
    a moment-matched normal tail avoids underflow while preserving monotonicity.
    """
    mean_power = max(float(mean_power), 0.0)
    noise_power = max(float(noise_power), 0.0)
    threshold = max(float(threshold), 0.0)
    if noise_power <= 1e-14:
        return 1.0 if mean_power >= threshold else 0.0

    noncentrality = mean_power / noise_power
    scaled_threshold = threshold / noise_power
    if noncentrality > 200.0:
        mean = mean_power + noise_power
        variance = noise_power**2 + 2.0 * noise_power * mean_power
        z = (threshold - mean) / max(math.sqrt(variance), 1e-14)
        return float(0.5 * math.erfc(z / math.sqrt(2.0)))

    poisson_weight = math.exp(-noncentrality)
    gamma_term = math.exp(-scaled_threshold)
    central_tail = gamma_term
    survival = poisson_weight * central_tail
    max_terms = int(max(64, math.ceil(noncentrality + 12.0 * math.sqrt(noncentrality + 1.0) + 16.0)))
    for term_idx in range(1, max_terms):
        poisson_weight *= noncentrality / float(term_idx)
        gamma_term *= scaled_threshold / float(term_idx)
        central_tail = min(1.0, central_tail + gamma_term)
        increment = poisson_weight * central_tail
        survival += increment
        if term_idx > noncentrality and abs(increment) < 1e-12:
            break
    return float(np.clip(survival, 0.0, 1.0))


def _complex_gaussian_gain_survival(mean_power, noise_power, threshold):
    mean_power = np.asarray(mean_power, dtype=float)
    noise_power = np.asarray(noise_power, dtype=float)
    probabilities = np.empty_like(mean_power, dtype=float)
    for index in np.ndindex(mean_power.shape):
        probabilities[index] = _noncentral_exponential_sf_scalar(
            mean_power[index],
            noise_power[index],
            threshold,
        )
    return probabilities


def _conditional_effective_moments(stale_state, codebook, channel_rho, csi_delay_slots):
    """Return conditional mean and variance of effective channels for all states.

    The simulator evolves physical links with AR(1). The direct link uses the
    same 0.1 scale as environment reset and temporal innovations; IRS/device
    and IRS/BS links use unit Rayleigh scale. The product-form cascaded link is
    approximated as complex Gaussian after matching conditional mean/variance.
    """
    codebook = np.asarray(codebook, dtype=np.complex128)
    delay = max(int(csi_delay_slots), 0)
    rho_delay = float(channel_rho) ** delay
    innovation_power = max(0.0, 1.0 - rho_delay**2)
    direct_noise_power = (0.1**2) * innovation_power
    reflected_noise_power = innovation_power
    bs_noise_power = innovation_power

    h_d_stale = np.asarray(stale_state["h_d"], dtype=np.complex128)
    h_r_stale = np.asarray(stale_state["h_r"], dtype=np.complex128)
    h_bs_stale = np.asarray(stale_state["h_bs_r"], dtype=np.complex128)

    h_d_mean = rho_delay * h_d_stale
    h_r_mean = rho_delay * h_r_stale
    h_bs_mean = rho_delay * h_bs_stale
    product_mean = h_r_mean * h_bs_mean[np.newaxis, :]
    product_second = (
        np.abs(h_r_mean) ** 2 + reflected_noise_power
    ) * (
        np.abs(h_bs_mean[np.newaxis, :]) ** 2 + bs_noise_power
    )
    product_variance = np.maximum(product_second - np.abs(product_mean) ** 2, 0.0)

    cascade_scale = 0.05 / math.sqrt(float(h_r_stale.shape[1]))
    cascade_mean = (product_mean @ codebook.T).T * cascade_scale
    cascade_variance = (product_variance @ (np.abs(codebook) ** 2).T).T * cascade_scale**2
    effective_mean = h_d_mean[np.newaxis, :] + cascade_mean
    effective_variance = direct_noise_power + cascade_variance
    return effective_mean, np.maximum(effective_variance, 0.0)


def _posterior_matrix_rng(episode_seed, slot_idx, seed_offset):
    if episode_seed is None:
        return np.random.default_rng()
    seed = (
        int(episode_seed)
        + int(seed_offset)
        + 0x6A09E667
        + int(slot_idx) * 0x9E3779B1
    ) % (2**32)
    return np.random.default_rng(seed)


def _monte_carlo_viability_matrix(
    stale_state,
    codebook,
    threshold,
    channel_rho,
    csi_delay_slots,
    posterior_num_samples,
    rng,
):
    """Sample current physical channels conditional on stale CSI and test feasibility."""
    codebook = np.asarray(codebook, dtype=np.complex128)
    delay = max(int(csi_delay_slots), 0)
    rho_delay = float(channel_rho) ** delay
    innovation_std = math.sqrt(max(0.0, 1.0 - rho_delay**2))
    h_d_stale = np.asarray(stale_state["h_d"], dtype=np.complex128)
    h_r_stale = np.asarray(stale_state["h_r"], dtype=np.complex128)
    h_bs_stale = np.asarray(stale_state["h_bs_r"], dtype=np.complex128)
    counts = np.zeros((h_d_stale.size, codebook.shape[0]), dtype=float)
    samples = int(posterior_num_samples)
    if samples <= 0:
        raise ValueError("posterior_num_samples must be positive")

    for _sample_idx in range(samples):
        h_d = rho_delay * h_d_stale + _complex_normal(
            rng,
            h_d_stale.shape,
            scale=0.1 * innovation_std,
        )
        h_r = rho_delay * h_r_stale + _complex_normal(
            rng,
            h_r_stale.shape,
            scale=innovation_std,
        )
        h_bs = rho_delay * h_bs_stale + _complex_normal(
            rng,
            h_bs_stale.shape,
            scale=innovation_std,
        )
        cascade = ((h_r * h_bs[np.newaxis, :]) @ codebook.T).T
        cascade *= 0.05 / math.sqrt(float(h_r_stale.shape[1]))
        h_total = h_d[np.newaxis, :] + cascade
        counts += (np.abs(h_total).T ** 2 >= float(threshold)).astype(float)
    return counts / float(samples)


def posterior_viability_matrix(
    *,
    stale_state,
    codebook,
    args,
    p_max,
    channel_rho,
    csi_delay_slots,
    posterior_mode=POSTERIOR_MODE_ANALYTIC,
    posterior_num_samples=256,
    posterior_clip_eps=1e-6,
    posterior_seed_offset=0,
    episode_seed=None,
    slot_idx=0,
    rng=None,
):
    """Compute p[k,c] = P(device k feasible on IRS state c | stale CSI).

    The feasibility event matches ``limited.build_candidate`` under the
    default execution margins: ``|h_cur[k,c]|^2 >= max(g_th, alpha_th^2/P_max)``.
    Returned matrix shape is ``(K, C)``.
    """
    mode = validate_posterior_mode(posterior_mode)
    threshold = max(float(args.g_th), (float(args.alpha_th) ** 2) / max(float(p_max), 1e-12))
    if mode == POSTERIOR_MODE_ANALYTIC:
        effective_mean, effective_variance = _conditional_effective_moments(
            stale_state,
            codebook,
            channel_rho,
            csi_delay_slots,
        )
        probabilities = _complex_gaussian_gain_survival(
            np.abs(effective_mean).T ** 2,
            effective_variance.T,
            threshold,
        )
    elif mode == POSTERIOR_MODE_MONTE_CARLO:
        rng = _posterior_matrix_rng(episode_seed, slot_idx, posterior_seed_offset) if rng is None else rng
        probabilities = _monte_carlo_viability_matrix(
            stale_state,
            codebook,
            threshold,
            channel_rho,
            csi_delay_slots,
            posterior_num_samples,
            rng,
        )
    else:
        raise AssertionError(f"unhandled posterior mode: {mode}")
    return _clip_probabilities(probabilities, posterior_clip_eps)


def posterior_summary_from_matrix(
    probabilities,
    remaining_mask,
    *,
    posterior_mode,
    posterior_num_samples,
    posterior_clip_eps,
    posterior_seed_offset,
):
    """Build JSON-friendly posterior viability diagnostics from a K x C matrix."""
    probabilities = np.asarray(probabilities, dtype=float)
    remaining_mask = np.asarray(remaining_mask, dtype=bool)
    if remaining_mask.shape[0] != probabilities.shape[0]:
        raise ValueError("remaining_mask length must match the posterior matrix K dimension")
    remaining_probabilities = probabilities[remaining_mask]
    flat = remaining_probabilities.reshape(-1) if remaining_probabilities.size else np.asarray([], dtype=float)
    quantiles = (
        np.quantile(flat, [0.05, 0.25, 0.5, 0.75, 0.95])
        if flat.size
        else np.zeros(5, dtype=float)
    )
    expected_by_state = (
        np.sum(remaining_probabilities, axis=0)
        if remaining_probabilities.size
        else np.zeros(probabilities.shape[1], dtype=float)
    )
    return {
        "posterior_mode": validate_posterior_mode(posterior_mode),
        "posterior_num_samples": int(posterior_num_samples),
        "posterior_clip_eps": float(posterior_clip_eps),
        "posterior_seed_offset": int(posterior_seed_offset),
        "posterior_probability_shape": [int(probabilities.shape[0]), int(probabilities.shape[1])],
        "posterior_mean_p": float(np.mean(flat)) if flat.size else 0.0,
        "posterior_p_quantiles": {
            "q05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q95": float(quantiles[4]),
        },
        "posterior_expected_feasible_count_per_irs_state": [
            float(value) for value in expected_by_state
        ],
    }


def posterior_viability_probabilities(
    candidate,
    args,
    p_max,
    channel_rho,
    csi_delay_slots,
    sample_count=64,
    uncertainty_scale=1.0,
    mean_mode=POSTERIOR_MEAN_MODE_AR1_PREDICT,
    posterior_mode=POSTERIOR_MODE_MONTE_CARLO,
    posterior_clip_eps=0.0,
    rng=None,
):
    """Estimate per-node current viability probabilities from stale CSI.

    A node is viable when ``|H_current|^2 >= max(g_th, alpha_th^2 / P_max)``.
    With zero uncertainty this reduces to a deterministic stale/predicted
    viability mask. Otherwise, a fixed-seed Monte Carlo estimate is used for
    the Rice-like magnitude probability; the policy supplies the RNG so runs
    remain reproducible and no hidden current CSI is used.
    """
    h_gain = np.asarray(candidate["h_gain"], dtype=float)
    mean_abs = np.sqrt(np.maximum(h_gain, 0.0)) * _posterior_mean_factor(
        channel_rho,
        csi_delay_slots,
        mean_mode,
    )
    gain_threshold = max(float(args.g_th), (float(args.alpha_th) ** 2) / max(float(p_max), 1e-12))
    amp_threshold = float(np.sqrt(max(gain_threshold, 0.0)))
    remaining_mask = np.asarray(candidate["valid_mask"], dtype=bool) | (
        np.asarray(candidate.get("success_reliability", np.zeros_like(h_gain)), dtype=float) >= 0.0
    )
    reference = mean_abs[remaining_mask] if np.any(remaining_mask) else mean_abs
    reference_rms = float(np.sqrt(np.mean(reference**2))) if reference.size else 0.0
    temporal_std = _posterior_temporal_std(channel_rho, csi_delay_slots, mean_mode)
    error_scale = float(uncertainty_scale) * float(temporal_std) * max(reference_rms, 1e-12)

    posterior_mode = validate_posterior_mode(posterior_mode)
    if error_scale <= 1e-12:
        probabilities = (mean_abs >= amp_threshold).astype(float)
        return (
            _clip_probabilities(probabilities, posterior_clip_eps),
            float(error_scale),
            float(temporal_std),
        )
    if posterior_mode == POSTERIOR_MODE_ANALYTIC:
        probabilities = _complex_gaussian_gain_survival(
            mean_abs**2,
            np.full_like(mean_abs, error_scale**2, dtype=float),
            gain_threshold,
        )
        return (
            _clip_probabilities(probabilities, posterior_clip_eps),
            float(error_scale),
            float(temporal_std),
        )
    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive in monte_carlo posterior mode")
    if rng is None:
        rng = np.random.default_rng()

    samples = int(sample_count)
    noise = (
        rng.normal(size=(samples, mean_abs.size))
        + 1j * rng.normal(size=(samples, mean_abs.size))
    ) / np.sqrt(2.0)
    h_samples = mean_abs[np.newaxis, :] + error_scale * noise
    probabilities = np.mean(np.abs(h_samples) >= amp_threshold, axis=0)
    return (
        _clip_probabilities(probabilities.astype(float), posterior_clip_eps),
        float(error_scale),
        float(temporal_std),
    )


def attach_posterior_viability(
    candidate,
    args,
    env,
    channel_rho,
    csi_delay_slots,
    sample_count,
    uncertainty_scale,
    mean_mode,
    rng,
    posterior_mode=POSTERIOR_MODE_MONTE_CARLO,
    posterior_clip_eps=0.0,
):
    """Return a candidate annotated with posterior viability diagnostics."""
    adjusted = dict(candidate)
    probabilities, error_scale, temporal_std = posterior_viability_probabilities(
        candidate,
        args,
        env.P_max,
        channel_rho,
        csi_delay_slots,
        sample_count=sample_count,
        uncertainty_scale=uncertainty_scale,
        mean_mode=mean_mode,
        posterior_mode=posterior_mode,
        posterior_clip_eps=posterior_clip_eps,
        rng=rng,
    )
    remaining = ~env.transmitted_flags
    remaining_probabilities = probabilities[remaining]
    entropy = bernoulli_entropy(remaining_probabilities) if remaining_probabilities.size else []
    adjusted["posterior_viability_prob"] = probabilities
    adjusted["posterior_prior_expected_count"] = float(np.sum(remaining_probabilities))
    adjusted["posterior_prior_entropy_mean"] = float(np.mean(entropy)) if len(entropy) else 0.0
    adjusted["posterior_effective_error_scale"] = float(error_scale)
    adjusted["posterior_temporal_uncertainty_std"] = float(temporal_std)
    return adjusted


def posterior_probability_coverage_stats(candidates_by_index, selected_indices, num_nodes):
    """Compute probability-coverage diagnostics for selected IRS probe indices."""
    covered_probability = np.zeros(int(num_nodes), dtype=float)
    marginal_fractions = []
    overlap_fractions = []
    for index in selected_indices:
        candidate = candidates_by_index[int(index)]
        probabilities = np.asarray(candidate["posterior_viability_prob"], dtype=float)
        marginal = float(np.sum((1.0 - covered_probability) * probabilities))
        overlap = float(np.sum(covered_probability * probabilities))
        expected = max(float(np.sum(probabilities)), 1e-12)
        marginal_fractions.append(marginal / max(float(num_nodes), 1.0))
        overlap_fractions.append(overlap / expected)
        covered_probability = 1.0 - (1.0 - covered_probability) * (1.0 - probabilities)
    return (
        float(np.mean(marginal_fractions)) if marginal_fractions else 0.0,
        float(np.mean(overlap_fractions)) if overlap_fractions else 0.0,
    )


def posterior_guided_sparse_indices(
    seed_candidates,
    args,
    budget,
    topk_fraction,
    coverage_weight,
    power_weight,
    uncertainty_weight,
):
    """Select IRS probes by posterior expected viability and marginal coverage."""
    ranked_candidates = sorted(
        seed_candidates,
        key=lambda candidate: (
            float(candidate["posterior_prior_expected_count"]),
            -float(candidate["power_avg"]) if int(candidate["tx_this_slot"]) > 0 else 0.0,
            float(candidate["mean_gain_remaining"]),
            -int(candidate["irs_index"]),
        ),
        reverse=True,
    )
    if not ranked_candidates:
        return [], 0, 0.0, 0.0

    budget = min(int(budget), len(ranked_candidates), int(args.num_codebook_states))
    anchor_count = min(budget, max(1, int(np.ceil(float(topk_fraction) * budget))))
    selected = list(ranked_candidates[:anchor_count])
    selected_indices = [int(candidate["irs_index"]) for candidate in selected]
    selected_set = set(selected_indices)
    covered_probability = np.zeros(int(args.num_nodes), dtype=float)
    for candidate in selected:
        probabilities = np.asarray(candidate["posterior_viability_prob"], dtype=float)
        covered_probability = 1.0 - (1.0 - covered_probability) * (1.0 - probabilities)

    remaining = [
        candidate
        for candidate in ranked_candidates[anchor_count:]
        if int(candidate["irs_index"]) not in selected_set
    ]
    while len(selected_indices) < budget and remaining:
        def posterior_key(candidate):
            probabilities = np.asarray(candidate["posterior_viability_prob"], dtype=float)
            expected_fraction = float(np.sum(probabilities)) / max(float(args.num_nodes), 1.0)
            marginal_fraction = float(
                np.sum((1.0 - covered_probability) * probabilities)
            ) / max(float(args.num_nodes), 1.0)
            entropy_mean = float(candidate.get("posterior_prior_entropy_mean", 0.0))
            power_penalty = float(power_weight) * float(candidate["power_avg"])
            score = (
                expected_fraction
                + float(coverage_weight) * marginal_fraction
                + float(uncertainty_weight) * entropy_mean
                - power_penalty
            )
            return (
                score,
                marginal_fraction,
                expected_fraction,
                entropy_mean,
                -float(candidate["power_avg"]) if int(candidate["tx_this_slot"]) > 0 else 0.0,
                -int(candidate["irs_index"]),
            )

        best_candidate = max(remaining, key=posterior_key)
        remaining = [
            candidate
            for candidate in remaining
            if int(candidate["irs_index"]) != int(best_candidate["irs_index"])
        ]
        selected.append(best_candidate)
        selected_indices.append(int(best_candidate["irs_index"]))
        selected_set.add(int(best_candidate["irs_index"]))
        probabilities = np.asarray(best_candidate["posterior_viability_prob"], dtype=float)
        covered_probability = 1.0 - (1.0 - covered_probability) * (1.0 - probabilities)

    candidates_by_index = {int(candidate["irs_index"]): candidate for candidate in seed_candidates}
    marginal_mean, overlap_mean = posterior_probability_coverage_stats(
        candidates_by_index,
        selected_indices,
        args.num_nodes,
    )
    return selected_indices, anchor_count, marginal_mean, overlap_mean


def poisson_binomial_pmf(probabilities):
    """Compute the Poisson-binomial count PMF for independent Bernoulli priors."""
    probabilities = np.clip(np.asarray(probabilities, dtype=np.longdouble), 0.0, 1.0)
    pmf = np.zeros(probabilities.size + 1, dtype=np.longdouble)
    pmf[0] = 1.0
    for idx, probability in enumerate(probabilities, start=1):
        previous = pmf.copy()
        pmf[: idx + 1] = 0.0
        pmf[:idx] += previous[:idx] * (1.0 - probability)
        pmf[1 : idx + 1] += previous[:idx] * probability
    total = float(np.sum(pmf))
    if total > 0.0:
        pmf /= total
    return np.asarray(pmf, dtype=float)


def _count_likelihood(count_count, observed_count, noise_model=None):
    counts = np.arange(int(count_count) + 1, dtype=float)
    observed = float(observed_count)
    if noise_model is None:
        noise_std = 0.0
    elif isinstance(noise_model, dict):
        model_type = str(noise_model.get("type", "gaussian")).strip()
        if model_type == "exact":
            noise_std = 0.0
        elif model_type in {"gaussian", "discrete_gaussian"}:
            noise_std = float(noise_model.get("std", 0.0))
        else:
            raise ValueError(f"unsupported count noise model: {model_type}")
    else:
        noise_std = float(noise_model)
    if float(noise_std) <= 1e-12:
        rounded = int(np.clip(round(observed), 0, int(count_count)))
        likelihood = np.zeros(int(count_count) + 1, dtype=float)
        likelihood[rounded] = 1.0
        return likelihood
    normalized = (counts - observed) / max(float(noise_std), 1e-12)
    normalized = np.clip(normalized, -40.0, 40.0)
    likelihood = np.exp(-0.5 * normalized**2)
    return likelihood


def _prefix_suffix_count_pmfs(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.longdouble), 0.0, 1.0)
    count_count = int(probabilities.size)
    prefix = np.zeros((count_count + 1, count_count + 1), dtype=np.longdouble)
    suffix = np.zeros((count_count + 1, count_count + 1), dtype=np.longdouble)
    prefix[0, 0] = 1.0
    for idx, probability in enumerate(probabilities):
        prefix[idx + 1, : idx + 2] = 0.0
        prefix[idx + 1, : idx + 1] += prefix[idx, : idx + 1] * (1.0 - probability)
        prefix[idx + 1, 1 : idx + 2] += prefix[idx, : idx + 1] * probability
    suffix[count_count, 0] = 1.0
    for idx in range(count_count - 1, -1, -1):
        tail_size = count_count - idx
        probability = probabilities[idx]
        suffix[idx, : tail_size + 1] = 0.0
        suffix[idx, :tail_size] += suffix[idx + 1, :tail_size] * (1.0 - probability)
        suffix[idx, 1 : tail_size + 1] += suffix[idx + 1, :tail_size] * probability
    return probabilities, prefix, suffix


def _conditioned_count_details(probabilities, observed_count, noise_model=None):
    probabilities, prefix, suffix = _prefix_suffix_count_pmfs(probabilities)
    count_count = int(probabilities.size)
    likelihood = np.asarray(
        _count_likelihood(count_count, observed_count, noise_model),
        dtype=np.longdouble,
    )
    pmf = prefix[count_count, : count_count + 1].copy()
    evidence = np.sum(pmf * likelihood)
    if evidence <= np.finfo(float).tiny:
        target = int(np.clip(round(float(observed_count)), 0, count_count))
        fallback = np.zeros(count_count, dtype=float)
        if target > 0:
            ranked = np.argsort(np.asarray(probabilities, dtype=float))[::-1]
            fallback[ranked[:target]] = 1.0
        return {
            "marginals": fallback,
            "count_pmf": np.asarray(pmf, dtype=float),
            "count_posterior": np.asarray(pmf, dtype=float),
            "posterior_count_mean": float(np.sum(np.arange(count_count + 1) * pmf)),
            "evidence": 0.0,
            "observed_count": float(observed_count),
            "noise_model": noise_model,
        }

    count_posterior = (pmf * likelihood) / evidence
    marginals = np.zeros(count_count, dtype=np.longdouble)
    for node_idx, probability in enumerate(probabilities):
        numerator = np.longdouble(0.0)
        for total_count in range(1, count_count + 1):
            left_min = max(0, total_count - 1 - (count_count - node_idx - 1))
            left_max = min(node_idx, total_count - 1)
            without_probability = np.longdouble(0.0)
            for left_count in range(left_min, left_max + 1):
                right_count = total_count - 1 - left_count
                without_probability += (
                    prefix[node_idx, left_count]
                    * suffix[node_idx + 1, right_count]
                )
            numerator += probability * without_probability * likelihood[total_count]
        marginals[node_idx] = numerator / evidence
    return {
        "marginals": np.clip(np.asarray(marginals, dtype=float), 0.0, 1.0),
        "count_pmf": np.asarray(pmf, dtype=float),
        "count_posterior": np.asarray(count_posterior, dtype=float),
        "posterior_count_mean": float(np.sum(np.arange(count_count + 1) * count_posterior)),
        "evidence": float(evidence),
        "observed_count": float(observed_count),
        "noise_model": noise_model,
    }


def count_conditioned_marginals(probabilities, y, noise_model=None):
    """Return q_k = P(z_k = 1 | aggregate count feedback, priors).

    The aggregate count feedback is modeled as a noisy observation of the
    Poisson-binomial count. With ``noise_model=None``, this is exact count
    conditioning:

    ``q_k = p_k P(sum_{j != k} z_j = y - 1) / P(sum_j z_j = y)``.
    """
    return _conditioned_count_details(probabilities, y, noise_model)["marginals"]


def count_conditioned_posterior_details(probabilities, y, noise_model=None):
    """Return count-conditioned marginals plus PMF/evidence diagnostics."""
    return _conditioned_count_details(probabilities, y, noise_model)


def posterior_invitation_target_count(
    posterior_marginals,
    observed_count,
    target_policy,
    cumulative_probability_target=1.0,
    lambda_fail=1.0,
    lambda_miss=1.0,
):
    """Choose how many top posterior-marginal devices to invite."""
    target_policy = validate_posterior_target_policy(target_policy)
    marginals = np.clip(np.asarray(posterior_marginals, dtype=float), 0.0, 1.0)
    count_count = int(marginals.size)
    if count_count == 0:
        return 0
    rounded_y = int(np.clip(round(float(observed_count)), 0, count_count))
    expected_count = float(np.sum(marginals))
    if target_policy == POSTERIOR_TARGET_POLICY_FIXED_Y:
        return rounded_y
    if target_policy == POSTERIOR_TARGET_POLICY_EXPECTED_COUNT:
        return int(np.clip(round(expected_count), 0, count_count))

    ranked = np.sort(marginals)[::-1]
    cumulative = np.concatenate(([0.0], np.cumsum(ranked)))
    if target_policy == POSTERIOR_TARGET_POLICY_CUMULATIVE_PROBABILITY:
        target = float(cumulative_probability_target) * expected_count
        return int(np.clip(np.searchsorted(cumulative, target, side="left"), 0, count_count))

    invite_counts = np.arange(count_count + 1, dtype=float)
    expected_success = cumulative
    expected_failures = invite_counts - expected_success
    expected_missed = expected_count - expected_success
    if target_policy == POSTERIOR_TARGET_POLICY_FAILURE_PENALTY:
        objective = float(lambda_fail) * expected_failures + float(lambda_miss) * expected_missed
    elif target_policy == POSTERIOR_TARGET_POLICY_MISSED_PENALTY:
        objective = float(lambda_fail) * expected_failures + float(lambda_miss) * expected_missed
    else:
        raise AssertionError(f"unhandled posterior target policy: {target_policy}")
    return int(np.argmin(objective))


def _probability_summary(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "q05": 0.0, "q50": 0.0, "q95": 0.0}
    quantiles = np.quantile(values, [0.05, 0.5, 0.95])
    return {
        "mean": float(np.mean(values)),
        "q05": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q95": float(quantiles[2]),
    }


def apply_count_conditioned_invitation_refinement(
    candidate,
    args,
    env,
    feedback,
    probabilities,
    strength,
    count_noise_std_scale,
    invitation_rule,
    invitation_threshold,
    cardinality_policy=POSTERIOR_TARGET_POLICY_FIXED_Y,
    cumulative_probability_target=1.0,
    lambda_fail=1.0,
    lambda_miss=1.0,
):
    """Refine the confirmed IRS invitation mask using Bayesian count feedback."""
    invitation_rule = validate_posterior_invitation_rule(invitation_rule)
    adjusted = dict(candidate)
    stale_mask = np.asarray(candidate["valid_mask"], dtype=bool).copy()
    remaining_mask = ~env.transmitted_flags
    stale_remaining_mask = stale_mask & remaining_mask
    remaining_indices = np.flatnonzero(remaining_mask)
    remaining_count = int(remaining_indices.size)
    stale_count = int(np.sum(stale_remaining_mask))
    full_probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    remaining_probabilities = full_probabilities[remaining_indices]
    observed_count = float(feedback["observed_tx_fraction"]) * float(args.num_nodes)
    observed_count = float(np.clip(observed_count, 0.0, float(remaining_count)))
    count_noise_std = (
        float(args.confirmation_feedback_noise_std)
        * float(args.num_nodes)
        * float(count_noise_std_scale)
    )
    noise_model = (
        None
        if count_noise_std <= 1e-12
        else {"type": "gaussian", "std": float(count_noise_std)}
    )
    conditioned = count_conditioned_posterior_details(
        remaining_probabilities,
        observed_count,
        noise_model,
    )
    posterior_marginals = np.zeros_like(full_probabilities)
    posterior_marginals[remaining_indices] = conditioned["marginals"]
    posterior_count_mean = float(conditioned["posterior_count_mean"])

    if float(strength) <= 0.0:
        corrected_mask = stale_remaining_mask
        requested_target_count = stale_count
    elif invitation_rule in {
        POSTERIOR_INVITATION_RULE_MEAN_TOPK,
        POSTERIOR_INVITATION_RULE_TOP_Y,
    }:
        requested_target_count = int(
            np.clip(
                round((1.0 - float(strength)) * stale_count + float(strength) * posterior_count_mean),
                0,
                remaining_count,
            )
        )
        if invitation_rule == POSTERIOR_INVITATION_RULE_TOP_Y:
            requested_target_count = posterior_invitation_target_count(
                conditioned["marginals"],
                observed_count,
                cardinality_policy,
                cumulative_probability_target,
                lambda_fail,
                lambda_miss,
            )
        ranked_nodes = sorted(
            [int(index) for index in remaining_indices],
            key=lambda index: (
                float(posterior_marginals[index]),
                float(full_probabilities[index]),
                float(candidate["h_gain"][index]),
                -float(candidate["required_power"][index]),
                -int(index),
            ),
            reverse=True,
        )
        corrected_mask = np.zeros_like(stale_mask, dtype=bool)
        corrected_mask[ranked_nodes[:requested_target_count]] = True
    elif invitation_rule == POSTERIOR_INVITATION_RULE_THRESHOLD:
        posterior_mask = (
            posterior_marginals >= float(invitation_threshold)
        ) & remaining_mask
        if float(strength) >= 1.0:
            corrected_mask = posterior_mask
        else:
            corrected_mask = stale_remaining_mask.copy()
            changed = np.flatnonzero(posterior_mask != stale_remaining_mask)
            ranked_changes = sorted(
                [int(index) for index in changed],
                key=lambda index: abs(float(posterior_marginals[index]) - 0.5),
                reverse=True,
            )
            update_count = int(round(float(strength) * len(ranked_changes)))
            for index in ranked_changes[:update_count]:
                corrected_mask[index] = posterior_mask[index]
        requested_target_count = int(np.sum(corrected_mask))
    else:
        raise AssertionError(f"unhandled posterior invitation rule: {invitation_rule}")

    target_count = int(np.sum(corrected_mask))
    added_mask = corrected_mask & (~stale_remaining_mask) & remaining_mask
    pruned_mask = stale_remaining_mask & (~corrected_mask)
    scheduled_power = np.asarray(candidate["required_power"], dtype=float)[corrected_mask]
    adjusted["valid_mask"] = corrected_mask
    adjusted["corrected_invitation_mask"] = corrected_mask.copy()
    adjusted["tx_this_slot"] = target_count
    adjusted["power_avg"] = float(np.mean(scheduled_power)) if scheduled_power.size else 0.0
    adjusted["posterior_count_refinement_strength"] = float(strength)
    adjusted["posterior_count_noise_std_scale"] = float(count_noise_std_scale)
    adjusted["posterior_invitation_rule"] = invitation_rule
    adjusted["posterior_invitation_threshold"] = float(invitation_threshold)
    adjusted["posterior_invitation_cardinality_policy"] = validate_posterior_target_policy(
        cardinality_policy
    )
    adjusted["posterior_invitation_cumulative_probability_target"] = float(
        cumulative_probability_target
    )
    adjusted["posterior_invitation_lambda_fail"] = float(lambda_fail)
    adjusted["posterior_invitation_lambda_miss"] = float(lambda_miss)
    adjusted["posterior_invitation_feedback_count"] = float(observed_count)
    adjusted["posterior_invitation_selected_cardinality"] = target_count
    adjusted["posterior_invitation_prior_summary"] = _probability_summary(
        remaining_probabilities
    )
    adjusted["posterior_invitation_marginal_summary"] = _probability_summary(
        conditioned["marginals"]
    )
    stale_overlap = int(np.sum(corrected_mask & stale_remaining_mask))
    adjusted["posterior_invitation_stale_overlap_count"] = stale_overlap
    adjusted["posterior_invitation_stale_overlap_fraction"] = (
        float(stale_overlap) / float(max(target_count, 1))
    )
    adjusted["posterior_refinement_observed_count"] = float(observed_count)
    adjusted["posterior_refinement_prior_expected_count"] = float(np.sum(remaining_probabilities))
    adjusted["posterior_refinement_posterior_expected_count"] = posterior_count_mean
    adjusted["posterior_refinement_requested_target_count"] = int(requested_target_count)
    adjusted["posterior_refinement_target_count"] = target_count
    adjusted["posterior_refinement_added"] = int(np.sum(added_mask))
    adjusted["posterior_refinement_pruned"] = int(np.sum(pruned_mask))
    adjusted["posterior_refinement_applied"] = int(
        not np.array_equal(corrected_mask, stale_remaining_mask)
    )
    adjusted["posterior_refinement_count_evidence"] = float(conditioned["evidence"])
    adjusted["posterior_refinement_mean_marginal"] = float(
        np.mean(conditioned["marginals"])
    ) if conditioned["marginals"].size else 0.0
    return adjusted
