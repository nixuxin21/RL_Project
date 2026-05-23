# Limitations for cost_frontier_main_v1

This document lists the limitations that should be disclosed in the paper, appendix, or reviewer-response materials.

## 1. Episodes Per Seed

The full focused validation uses 20 seeds but only 1 episode per seed. This is enough for a focused cost-frontier validation but not ideal for final archival claims. The paper should disclose:

> Full focused validation uses 20 seeds and one episode per seed; additional episode-level replication is left for extended validation.

Do not imply that the result has exhaustive episode-level averaging.

## 2. NMSE Is a Proxy

The reported NMSE is a `synthetic_unit_variance` proxy. It is not a true waveform-level or source-signal AirComp MSE.

Safe language:

> We report a synthetic-unit-variance NMSE proxy induced by scheduling, failed invitations, missed feasible devices, clipping, and receiver-noise terms in the simulator.

Unsafe language:

> The proposed method reduces true AirComp MSE.

## 3. Standalone Components Do Not Dominate

The ablations contradict standalone dominance:

- posterior_invitation_feedback - count_only_mask_correction: delta oracle gap = +0.0700
- posterior_greedy_feedback - coverage_aware: delta oracle gap = +1.3731

The contribution must be framed as the coupled PGI protocol, not as two independently dominant modules.

## 4. Same-B NMSE vs Count-Only Is Not Improved

Same-B PGI vs count-only:

- delta oracle gap = -0.7244
- delta NMSE proxy = +0.0001
- clustered 95% CI for NMSE proxy = [-0.0013, +0.0015]

This means same-B NMSE proxy is effectively tied, not improved. The NMSE-proxy improvement is supported in closest-cost comparisons and against coverage-aware, sparse-topK, and rotating.

## 5. CSI Staleness Claim Is Not Supported

Against count-only, oracle-gap deltas by rho are:

- rho=0.5: -0.1800
- rho=0.7: -0.1250
- rho=0.9: -1.8681

Lower rho corresponds to staler CSI, so the strongest gain is not in the stalest tested condition. The safe claim is robustness across tested rho values, not monotonic improvement with staleness.

## 6. Saturated Scenarios Exist

The analysis flags 26/240 scenario-budget groups as saturated or weak evidence. The non-saturated-only check still supports the oracle-gap and cost conclusions, but saturation should be disclosed.

Non-saturated PGI vs count-only:

- delta oracle gap = -0.7685
- delta cost = -33.5145
- delta NMSE proxy = +0.0003

## 7. Feedback Noise Was Not Swept Here

The focused run uses feedback noise = none. Claims about robustness to feedback noise should rely on separate feedback-noise sweep results, not this package.

## 8. IRS Element Count Is Fixed

This focused validation uses 64 IRS elements. It tests C=64 and C=128 codebooks but not multiple IRS-element counts. Claims about IRS hardware scaling require additional experiments.

## 9. Optional Random Baseline Not Included

`random_same_budget` was optional and was not included in the full focused run because runtime was already substantial. The paper can still use rotating, sparse-topK, coverage-aware, count-only, ablations, and diagnostics here, but a final appendix may include random if reviewer expectations require it.

## 10. Protocol Cost Model Is Simple

The cost model uses equal unit weights:

`total_protocol_cost = stale_preview_calls + current_probe_calls + execution_slots`

This is transparent and useful for frontier analysis but not a full wall-clock or hardware-energy model. A paper should state the cost model explicitly and avoid presenting it as universal latency.

## 11. Oracle Diagnostics Are Not Deployable

`full_current_oracle` uses hidden current information and must be labeled non-deployable. `full_stale_exhaustive` is diagnostic and has very high stale-preview cost.

Do not compare deployable methods to oracle diagnostics as if all methods share the same information budget.

## 12. Temporal-AR Channel Model Scope

The posterior viability model and experiments are tied to the simulator's temporal-AR stale/current mismatch setting. Claims should not automatically generalize to non-AR mobility, hardware impairments, nonstationary channels, or imperfect posterior model specification without additional tests.

## 13. Cluster Bootstrap Is Helpful but Not Final Proof

The clustered bootstrap uses `scenario_config_probe_budget_run_seed` as the cluster unit and gives useful uncertainty estimates. Still, because episodes per seed equal 1 and scenarios share simulator structure, statistical claims should be framed as empirical evidence rather than final physical-layer proof.

## 14. Paper Claim Boundary

The safest claim is:

> PGI provides a low-protocol-cost cost-quality frontier under limited IRS probing, especially in larger codebooks and larger device populations.

Avoid:

> PGI universally dominates all baselines.

Avoid:

> The proposed posterior modules independently improve all metrics.

Avoid:

> The proposed method reduces true waveform/source-signal AirComp MSE.
