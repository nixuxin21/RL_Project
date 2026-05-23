# cost_frontier_main_v1 Evidence Summary

This evidence package summarizes the existing `cost_frontier_main_v1` analysis without rerunning experiments or changing code. The defensible paper claim is narrow:

> Posterior-greedy probing with count-conditioned invitation feedback (PGI) provides a low-protocol-cost cost-quality frontier under limited IRS probing.

The results do not support standalone dominance claims for posterior invitation alone or posterior-greedy probing alone. The reported NMSE is a `synthetic_unit_variance` proxy, not a true waveform/source-signal AirComp MSE.

## Source Artifacts

- Raw logs: `results/main/cost_frontier_main_v1/cost_frontier_main_v1_full_f6ca5c9dc1`
- Analysis directory: `results/main_analysis/cost_frontier_main_v1`
- Run rows: 35,520 from 8 scenario logs
- Git commit in logs: `9b9e826b29af8bf421266fac5220c60d8bf61210`
- Git dirty values in logs: `False`
- Seeds: 20
- Episodes per seed: 1

## Main Table Interpretation

| Method | Completion | Slots | Oracle Gap | NMSE Proxy | Total Cost | Current Probes | Stale Previews |
|---|---:|---:|---:|---:|---:|---:|---:|
| rotating_same_budget | 0.4465 | 4.6140 | 42.3423 | 0.2218 | 88.9540 | 42.1700 | 42.1700 |
| sparse_topk_same_budget | 0.6035 | 4.3781 | 36.7531 | 0.1715 | 124.1306 | 39.9175 | 79.8350 |
| coverage_aware | 0.6375 | 4.3213 | 35.3452 | 0.1571 | 122.3088 | 39.3292 | 78.6583 |
| count_only_mask_correction | 0.8417 | 3.6885 | 28.2875 | 0.0978 | 103.3610 | 33.2242 | 66.4483 |
| posterior_greedy_invitation_feedback | 0.8619 | 3.6521 | 27.5631 | 0.0979 | 69.5104 | 32.9292 | 32.9292 |
| posterior_invitation_feedback | 0.8435 | 3.7048 | 28.3575 | 0.0994 | 103.8898 | 33.3950 | 66.7900 |
| posterior_greedy_feedback | 0.6783 | 4.2840 | 36.7183 | 0.1668 | 82.6740 | 39.1950 | 39.1950 |
| full_stale_exhaustive | 0.7583 | 4.1365 | 33.0448 | 0.0812 | 399.4698 | 0.0000 | 395.3333 |
| full_current_oracle | 1.0000 | 2.5250 | 0.0000 | approximately 0 | 242.6583 | 0.0000 | 240.1333 |

PGI has the best deployable oracle gap among the main deployable methods and uses substantially lower protocol cost than count-only correction, coverage-aware probing, and sparse-topK.

## PGI vs Count-Only at Same Probe Budget

The paired same-B comparison against count-only mask correction is mixed but favorable for the main oracle-gap claim:

| Metric | PGI - Count-Only | Interpretation |
|---|---:|---|
| Completion | +0.0202 | Slightly higher completion |
| Oracle gap | -0.7244 | Lower gap; paired improvement |
| NMSE proxy | +0.0001 | Effectively tied; not an NMSE improvement claim |
| Total protocol cost | -33.8506 | Large cost reduction |
| Oracle-gap win rate | 0.5169 | Slight majority of paired rows |

The average total-cost reduction relative to count-only is 32.75 percent:

`(103.3610 - 69.5104) / 103.3610 = 32.75%`.

## Closest-Cost Comparison

Closest-cost comparisons are stronger and are the core evidence for the cost-quality frontier claim:

| Comparison | Mean Cost Distance | Delta Oracle Gap | Delta NMSE Proxy | Oracle-Gap Win Rate | NMSE Win Rate |
|---|---:|---:|---:|---:|---:|
| PGI - count_only_mask_correction | 9.0415 | -3.2471 | -0.0087 | 0.9042 | 0.7375 |
| PGI - coverage_aware | 10.8646 | -9.8179 | -0.0618 | 0.9917 | 1.0000 |
| PGI - rotating_same_budget | 6.8248 | -15.6233 | -0.1253 | 1.0000 | 1.0000 |
| PGI - sparse_topk_same_budget | 11.0506 | -11.3892 | -0.0778 | 1.0000 | 1.0000 |

This supports the claim that PGI occupies a favorable cost-quality frontier rather than merely winning by spending more protocol budget.

## Clustered Confidence Intervals

Cluster bootstrap intervals are grouped by `scenario_config_probe_budget_run_seed`.

| Comparison | Metric | Mean Delta | 95% CI | Interpretation |
|---|---|---:|---|---|
| PGI - count_only | oracle_gap | -0.7244 | [-0.9694, -0.4927] | Supports lower oracle gap at same B |
| PGI - count_only | NMSE proxy | +0.0001 | [-0.0013, +0.0015] | Effectively tied; do not claim improvement |
| PGI - coverage | oracle_gap | -7.7821 | [-8.0233, -7.5467] | Strongly lower oracle gap |
| PGI - coverage | NMSE proxy | -0.0591 | [-0.0608, -0.0574] | Strongly lower proxy NMSE |
| PGI - sparse_topk | oracle_gap | -9.1900 | [-9.4461, -8.9375] | Strongly lower oracle gap |
| PGI - rotating | oracle_gap | -14.7792 | [-15.0719, -14.4931] | Strongly lower oracle gap |

The confidence intervals support the frontier claim and the oracle-gap improvement claim. They do not support a same-B NMSE-proxy improvement over count-only.

## Cost Reduction Percentages

Using method-level total protocol cost:

| Reference | Reference Cost | PGI Cost | Cost Reduction |
|---|---:|---:|---:|
| count_only_mask_correction | 103.3610 | 69.5104 | 32.75% |
| coverage_aware | 122.3088 | 69.5104 | 43.17% |
| sparse_topk_same_budget | 124.1306 | 69.5104 | 44.00% |
| rotating_same_budget | 88.9540 | 69.5104 | 21.86% |

The central cost result is therefore not marginal: PGI achieves comparable or better deployable quality with roughly one-third lower cost than count-only and more than 40 percent lower cost than coverage-aware or sparse-topK.

## Frontier by Probe Budget

PGI is on the Pareto frontier for both oracle gap and NMSE proxy at every tested probe budget.

| B | Completion | Oracle Gap | NMSE Proxy | Total Cost | Oracle-Gap Frontier | NMSE Frontier |
|---:|---:|---:|---:|---:|---|---|
| 4 | 0.8167 | 32.2552 | 0.1151 | 34.9688 | yes | yes |
| 6 | 0.8510 | 29.3656 | 0.1030 | 48.9667 | yes | yes |
| 8 | 0.8698 | 27.5375 | 0.0972 | 61.8906 | yes | yes |
| 12 | 0.8688 | 25.1313 | 0.0903 | 88.1510 | yes | yes |
| 16 | 0.9031 | 23.5260 | 0.0840 | 113.5750 | yes | yes |

Count-only at B=16 reaches slightly lower NMSE proxy than PGI at B=16, but at much higher cost: 169.3563 vs 113.5750. This is why the cost-frontier framing is safer than a simple same-B dominance framing.

## Codebook and Device-Population Trends

Against count-only, the oracle-gap gain is stronger for larger codebooks and larger device populations:

| Subgroup | Delta Oracle Gap vs Count-Only | Delta NMSE Proxy vs Count-Only | Delta Cost vs Count-Only |
|---|---:|---:|---:|
| C=64 | -0.4483 | +0.0018 | -34.1963 |
| C=128 | -1.0004 | -0.0015 | -33.5050 |
| K=100 | -0.5488 | -0.0003 | -29.7654 |
| K=200 | -0.9000 | +0.0005 | -37.9358 |

Against coverage-aware, gains are consistently large:

| Subgroup | Delta Oracle Gap vs Coverage | Delta NMSE Proxy vs Coverage | Delta Cost vs Coverage |
|---|---:|---:|---:|
| C=64 | -7.4863 | -0.0582 | -52.8775 |
| C=128 | -8.0779 | -0.0600 | -52.7192 |
| K=100 | -6.3267 | -0.0610 | -51.8771 |
| K=200 | -9.2375 | -0.0572 | -53.7196 |

The larger-codebook and larger-population trends are consistent with the revised claim.

## CSI Staleness Trend

The result does not support the stronger claim that PGI is best under the stalest CSI. Against count-only:

| rho | Delta Oracle Gap | Delta NMSE Proxy | Delta Cost |
|---:|---:|---:|---:|
| 0.5 | -0.1800 | +0.0006 | -29.7469 |
| 0.7 | -0.1250 | +0.0017 | -33.3869 |
| 0.9 | -1.8681 | -0.0019 | -38.4181 |

The safe wording is that PGI remains cost-effective across tested CSI conditions, not that its gains monotonically increase with staleness.

## Saturation and Non-Saturated Evidence

Only 26 of 240 scenario-budget groups were flagged as saturated or weak-evidence groups, using the completion threshold rule in the analysis. That is 10.83 percent of scenario-budget groups.

In non-saturated groups, PGI remains favorable on oracle gap and cost:

| Comparison | Rows | Delta Oracle Gap | Delta NMSE Proxy | Delta Cost | Oracle-Gap Win Rate |
|---|---:|---:|---:|---:|---:|
| PGI - count_only | 4280 | -0.7685 | +0.0003 | -33.5145 | 0.5236 |
| PGI - coverage | 4280 | -7.9257 | -0.0593 | -50.8224 | 0.8051 |
| PGI - sparse_topk | 4280 | -9.3568 | -0.0735 | -52.1449 | 0.8313 |
| PGI - rotating | 4280 | -15.0372 | -0.1235 | -17.0633 | 0.9297 |

This reduces the risk that the observed advantage is merely caused by saturated easy settings.

## Component Ablation Interpretation

The component ablation is crucial for accurate paper positioning:

| Comparison | Delta Completion | Delta Oracle Gap | Delta NMSE Proxy | Delta Cost | Interpretation |
|---|---:|---:|---:|---:|---|
| posterior_invitation - count_only | +0.0019 | +0.0700 | +0.0016 | +0.5288 | Standalone posterior invitation does not improve count-only |
| posterior_greedy - coverage | +0.0408 | +1.3731 | +0.0097 | -39.6348 | Standalone posterior-greedy probing is cheaper but lower quality |
| PGI - posterior_invitation | +0.0183 | -0.7944 | -0.0015 | -34.3794 | Combining probing helps over invitation-only |
| PGI - posterior_greedy | +0.1835 | -9.1552 | -0.0688 | -13.1635 | Invitation feedback recovers much of the standalone probing weakness |

The paper should frame PGI as a coupled protocol: posterior-guided low-cost probing becomes competitive when paired with count-conditioned invitation feedback.

## Oracle Diagnostic Boundary

`full_current_oracle` is a non-deployable diagnostic using hidden current information and achieves zero oracle gap by construction. `full_stale_exhaustive` is also diagnostic, with very high stale-preview cost.

Deployable methods do not use hidden current device-level CSI in the logged diagnostics. No deployable method has negative oracle-gap rows or rows where achieved transmissions exceed the oracle objective.

This separation should be explicit in every table caption and method legend.

## NMSE Proxy Limitation

The measured NMSE uses the `synthetic_unit_variance` proxy. It is useful as a scheduling-quality proxy but should not be described as true waveform-level or source-signal AirComp MSE. A safe phrase is:

> We report a synthetic-unit-variance NMSE proxy induced by missing, failed, clipped, and noisy contributions; true waveform/source-signal AirComp MSE is left for future physical-layer validation.

## Bottom-Line Evidence Verdict

The evidence supports moving to paper writing with a narrow cost-frontier claim. It does not support broad dominance claims for either posterior component alone, and it does not support true AirComp MSE claims.
