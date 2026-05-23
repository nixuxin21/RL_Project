# Main Claim Matrix

This matrix classifies claims using only the existing `cost_frontier_main_v1` outputs.

## Supported Claims

| Claim | Evidence | Safe Wording |
|---|---|---|
| PGI improves the deployable cost-quality frontier. | PGI has 5 oracle-gap frontier points and 5 NMSE-proxy frontier points; closest-cost PGI vs count-only gives delta oracle gap = -3.2471 and oracle-gap win rate = 0.9042. | PGI forms a lower-cost cost-quality frontier under limited IRS probing. |
| PGI improves same-B oracle gap vs count-only. | Paired same-B delta oracle gap = -0.7244; clustered 95% CI [-0.9694, -0.4927]. | At the same nominal probe budget, PGI reduces oracle gap relative to count-only mask correction. |
| PGI improves same-B oracle gap and NMSE proxy vs coverage-aware. | Delta oracle gap = -7.7821; delta NMSE proxy = -0.0591; both clustered CIs are fully below zero. | PGI is substantially better than coverage-aware in this focused hard-grid evaluation. |
| PGI is stronger in larger codebooks. | Against count-only, C=64 delta gap = -0.4483 and C=128 delta gap = -1.0004. Against coverage, C=64 delta gap = -7.4863 and C=128 delta gap = -8.0779. | The gain is at least stable and appears stronger at C=128. |
| PGI is stronger in larger device populations. | Against count-only, K=100 delta gap = -0.5488 and K=200 delta gap = -0.9000. Against coverage, K=100 delta gap = -6.3267 and K=200 delta gap = -9.2375. | The gain is stable and larger for K=200 in this grid. |
| Aggregate feedback is most useful when combined with posterior-greedy probing. | PGI vs posterior_greedy_feedback gives delta oracle gap = -9.1552 and delta NMSE proxy = -0.0688. | The two components interact favorably; the combined method is the contribution. |
| Oracle diagnostics are clean. | No deployable method uses hidden current info; no deployable rows have negative oracle gap or achieved transmissions exceeding the oracle. | Deployable methods and diagnostic oracles are cleanly separated. |

## Weakly Supported Claims

| Claim | Evidence | Caveat |
|---|---|---|
| PGI improves completion vs count-only. | Method means: 0.8619 vs 0.8417; same-B delta completion = +0.0202. | Completion is partly saturated in some scenario-budget groups. |
| PGI improves NMSE proxy in cost-equivalent comparisons vs count-only. | Closest-cost delta NMSE proxy = -0.0087; NMSE win rate = 0.7375. | This is closest-cost, not same-B, and NMSE is synthetic proxy only. |
| PGI remains robust outside saturated settings. | Non-saturated PGI vs count-only delta oracle gap = -0.7685; cost delta = -33.5145. | Non-saturated paired NMSE proxy is effectively tied vs count-only. |

## Unsupported Claims

| Claim | Why Unsupported | Safer Replacement |
|---|---|---|
| PGI improves same-B NMSE proxy vs count-only. | Same-B delta NMSE proxy = +0.0001 with clustered 95% CI [-0.0013, +0.0015]. | Same-B NMSE proxy is tied vs count-only; cost-equivalent NMSE proxy is better. |
| PGI is strongest under the stalest CSI. | Against count-only, rho=0.5 delta gap = -0.1800, rho=0.7 = -0.1250, rho=0.9 = -1.8681. | PGI remains cost-effective across tested rho values. |
| The proposed method reduces true AirComp MSE. | The current metric is `synthetic_unit_variance` NMSE proxy, not waveform/source-signal MSE. | PGI improves oracle gap and cost-quality behavior; true AirComp MSE remains future work. |
| PGI dominates all baselines on every metric. | Count-only at B=16 has slightly lower NMSE proxy than PGI at B=16, although at much higher cost. | PGI is a frontier method, not a universal metric dominator. |

## Contradicted Claims

| Claim | Contradicting Evidence | Required Paper Stance |
|---|---|---|
| Posterior invitation alone improves count-only mask correction. | posterior_invitation_feedback - count_only gives delta oracle gap = +0.0700 and delta NMSE proxy = +0.0016. | Treat posterior invitation alone as an ablation, not as a winning method. |
| Posterior-greedy probing alone improves coverage-aware. | posterior_greedy_feedback - coverage_aware gives delta oracle gap = +1.3731 and delta NMSE proxy = +0.0097, although cost is lower. | Treat posterior-greedy alone as a cheaper but lower-quality ablation. |
| Full stale exhaustive is a deployable baseline. | It is diagnostic and has mean cost 399.4698 with exhaustive stale preview use. | Label it as diagnostic, not deployable. |
| Full current oracle is a deployable method. | It uses hidden current information and has oracle gap 0 by construction. | Label it as non-deployable oracle. |

## Recommended Abstract-Level Claim

Use:

> A coupled posterior-greedy probing and count-conditioned invitation policy achieves a favorable protocol-cost versus aggregation-quality frontier, reducing oracle gap at substantially lower protocol cost than count-only and coverage-aware baselines in hard limited-probing settings.

Avoid:

> Each posterior component independently dominates existing baselines.

Avoid:

> The method reduces true AirComp waveform MSE.
