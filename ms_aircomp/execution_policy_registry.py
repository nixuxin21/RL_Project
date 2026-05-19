"""维护执行信道错配策略别名、显示 label 和参数网格展开逻辑。"""

import ms_aircomp.limited_csi as limited
from ms_aircomp.probe_sets import validate_posterior_probe_objective

__all__ = [
    "MISMATCH_CHOICES",
    "MISMATCH_INDEPENDENT",
    "MISMATCH_TEMPORAL_AR1",
    "POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID",
    "POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID",
    "POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID",
    "POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID",
    "POLICY_ACTIVE_DIVERSE_FEEDBACK_GRID",
    "POLICY_AR1_PREDICT_ROTATING_GRID",
    "POLICY_CHOICES",
    "POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID",
    "POLICY_COUNT_CONDITIONED_INVITATION_FEEDBACK_GRID",
    "POLICY_COUNT_ONLY_MASK_CORRECTION_FEEDBACK_GRID",
    "POLICY_COVERAGE_ONLY_FILL_FEEDBACK_GRID",
    "POLICY_DEPLOYABLE_IRS_ORACLE_INVITATION",
    "POLICY_DIVERSITY_ONLY_FILL_FEEDBACK_GRID",
    "POLICY_EXECUTION_ORACLE",
    "POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID",
    "POLICY_FULL_CURRENT_ORACLE",
    "POLICY_FULL_STALE_EXHAUSTIVE",
    "POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID",
    "POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID",
    "POLICY_MS_AIRCOMP_WITHOUT_IRS",
    "POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID",
    "POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID",
    "POLICY_ORACLE_IRS_ORACLE_INVITATION",
    "POLICY_ORACLE_IRS_STALE_INVITATION",
    "POLICY_POSTERIOR_GREEDY_FEEDBACK_GRID",
    "POLICY_POSTERIOR_GREEDY_INVITATION_FEEDBACK_GRID",
    "POLICY_POSTERIOR_GUIDED_COUNT_REFINE_FEEDBACK_GRID",
    "POLICY_RANDOM_SAME_BUDGET_FEEDBACK_GRID",
    "POLICY_RANDOM_IRS",
    "POLICY_ROTATING_SAME_BUDGET_FEEDBACK_GRID",
    "POLICY_ROTATING_FEEDBACK_CONFIRM_GRID",
    "POLICY_SPARSE_TOPK_SAME_BUDGET_FEEDBACK_GRID",
    "POLICY_SPARSE_TOPK_FEEDBACK_GRID",
    "POLICY_STALE_TOPK_SAME_BUDGET_FEEDBACK_GRID",
    "POLICY_STALE_TOPK_FEEDBACK_GRID",
    "POLICY_TEMPORAL_DEVIATION_ORACLE_GRID",
    "POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID",
    "VALID_PROBING_POLICIES",
    "mismatch_scenarios",
    "policy_configs",
    "policy_label",
    "validate_posterior_probe_objective",
    "validate_probing_policy",
]


POLICY_EXECUTION_ORACLE = "Execution Oracle Full CSI"
POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID = "Execution-Risk Rotating Grid"
POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID = "Adaptive Execution-Risk Rotating Grid"
POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID = "Opportunity-Cost Execution-Risk Rotating Grid"
POLICY_AR1_PREDICT_ROTATING_GRID = "AR1-Predict Rotating Grid"
POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID = "Temporal-Reliability Rotating Grid"
POLICY_ROTATING_FEEDBACK_CONFIRM_GRID = "Rotating Feedback Confirm Grid"
POLICY_ROTATING_SAME_BUDGET_FEEDBACK_GRID = "Rotating Same-Budget Feedback Grid"
POLICY_STALE_TOPK_FEEDBACK_GRID = "Stale-TopK Feedback Grid"
POLICY_STALE_TOPK_SAME_BUDGET_FEEDBACK_GRID = "Stale-TopK Same-Budget Feedback Grid"
POLICY_ACTIVE_DIVERSE_FEEDBACK_GRID = "Active Diverse Feedback Grid"
POLICY_DIVERSITY_ONLY_FILL_FEEDBACK_GRID = "Diversity-Only Fill Feedback Grid"
POLICY_COVERAGE_ONLY_FILL_FEEDBACK_GRID = "Coverage-Only Fill Feedback Grid"
POLICY_RANDOM_SAME_BUDGET_FEEDBACK_GRID = "Random Same-Budget Feedback Grid"
POLICY_SPARSE_TOPK_FEEDBACK_GRID = "Sparse-TopK Feedback Grid"
POLICY_SPARSE_TOPK_SAME_BUDGET_FEEDBACK_GRID = "Sparse-TopK Same-Budget Feedback Grid"
POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID = "Coverage-Aware Sparse-TopK Feedback Grid"
POLICY_COUNT_ONLY_MASK_CORRECTION_FEEDBACK_GRID = "Count-Only Mask-Corrected Coverage-Aware Grid"
POLICY_COUNT_CONDITIONED_INVITATION_FEEDBACK_GRID = "Count-Conditioned Invitation Feedback Grid"
POLICY_POSTERIOR_GREEDY_FEEDBACK_GRID = "Posterior-Greedy Probing Feedback Grid"
POLICY_POSTERIOR_GREEDY_INVITATION_FEEDBACK_GRID = "Posterior-Greedy Probing + Count-Conditioned Invitation Grid"
POLICY_POSTERIOR_GUIDED_COUNT_REFINE_FEEDBACK_GRID = "Posterior-Guided Count-Refined Feedback Grid"
POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID = "Neighbor-Coverage Sparse-TopK Feedback Grid"
POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID = "Adaptive Sparse-TopK Feedback Grid"
POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID = "Adaptive Sparse-TopK V2 Feedback Grid"
POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID = "Adaptive Sparse-TopK V3 Feedback Grid"
POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID = "Learned Sparse Shortlist Feedback Grid"
POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID = "Learned Set Shortlist Feedback Grid"
POLICY_TEMPORAL_DEVIATION_ORACLE_GRID = "Temporal Deviation Oracle"
POLICY_MS_AIRCOMP_WITHOUT_IRS = "MS-AirComp without IRS"
POLICY_RANDOM_IRS = "Random IRS"
POLICY_FULL_STALE_EXHAUSTIVE = "Full Stale Exhaustive"
POLICY_FULL_CURRENT_ORACLE = "Full Current Oracle"
POLICY_ORACLE_IRS_STALE_INVITATION = "Oracle IRS with Stale Invitation"
POLICY_DEPLOYABLE_IRS_ORACLE_INVITATION = "Deployable IRS with Oracle Invitation"
POLICY_ORACLE_IRS_ORACLE_INVITATION = "Oracle IRS with Oracle Invitation"

MISMATCH_INDEPENDENT = "independent"
MISMATCH_TEMPORAL_AR1 = "temporal_ar1"
MISMATCH_CHOICES = {MISMATCH_INDEPENDENT, MISMATCH_TEMPORAL_AR1}
VALID_PROBING_POLICIES = ("rotating", "sparse_topk", "coverage_aware", "posterior_greedy")


def validate_probing_policy(probing_policy):
    """Normalize and validate the candidate-probing composition switch."""
    probing_policy = str(probing_policy).strip()
    if probing_policy not in VALID_PROBING_POLICIES:
        valid = ", ".join(VALID_PROBING_POLICIES)
        raise ValueError(f"unknown probing policy {probing_policy!r}; expected one of: {valid}")
    return probing_policy

POLICY_CHOICES = {
    "no_irs": limited.POLICY_NO_IRS,
    "ms_aircomp_without_irs": POLICY_MS_AIRCOMP_WITHOUT_IRS,
    "fixed": limited.POLICY_FIXED_IRS,
    "fixed_irs": limited.POLICY_FIXED_IRS,
    "random_irs": POLICY_RANDOM_IRS,
    "exact_greedy": limited.POLICY_EXACT_GREEDY,
    "estimated_greedy": limited.POLICY_EST_GREEDY,
    "random_probe": limited.POLICY_RANDOM_PROBE,
    "rotating": limited.POLICY_ROTATING_GRID,
    "robust_rotating": limited.POLICY_ROBUST_ROTATING_GRID,
    "risk_rotating": limited.POLICY_RISK_AWARE_ROTATING_GRID,
    "adaptive_risk_rotating": limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
    "execution_risk_rotating": POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
    "adaptive_execution_risk_rotating": POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
    "opportunity_execution_risk_rotating": POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID,
    "ar1_predict_rotating": POLICY_AR1_PREDICT_ROTATING_GRID,
    "temporal_reliability_rotating": POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID,
    "rotating_feedback_confirm": POLICY_ROTATING_FEEDBACK_CONFIRM_GRID,
    "feedback_confirm_rotating": POLICY_ROTATING_FEEDBACK_CONFIRM_GRID,
    "rotating_same_budget": POLICY_ROTATING_SAME_BUDGET_FEEDBACK_GRID,
    "rotating_same_budget_feedback": POLICY_ROTATING_SAME_BUDGET_FEEDBACK_GRID,
    "stale_topk_feedback": POLICY_STALE_TOPK_FEEDBACK_GRID,
    "stale_topk_rotating": POLICY_STALE_TOPK_FEEDBACK_GRID,
    "stale_topk_same_budget": POLICY_STALE_TOPK_SAME_BUDGET_FEEDBACK_GRID,
    "stale_topk_same_budget_feedback": POLICY_STALE_TOPK_SAME_BUDGET_FEEDBACK_GRID,
    "active_diverse_feedback": POLICY_ACTIVE_DIVERSE_FEEDBACK_GRID,
    "active_probe_set": POLICY_ACTIVE_DIVERSE_FEEDBACK_GRID,
    "diversity_only_fill": POLICY_DIVERSITY_ONLY_FILL_FEEDBACK_GRID,
    "diversity_only_fill_feedback": POLICY_DIVERSITY_ONLY_FILL_FEEDBACK_GRID,
    "coverage_only_fill": POLICY_COVERAGE_ONLY_FILL_FEEDBACK_GRID,
    "coverage_only_fill_feedback": POLICY_COVERAGE_ONLY_FILL_FEEDBACK_GRID,
    "random_same_budget": POLICY_RANDOM_SAME_BUDGET_FEEDBACK_GRID,
    "random_same_budget_feedback": POLICY_RANDOM_SAME_BUDGET_FEEDBACK_GRID,
    "random_sparse_feedback": POLICY_RANDOM_SAME_BUDGET_FEEDBACK_GRID,
    "sparse_topk_same_budget": POLICY_SPARSE_TOPK_SAME_BUDGET_FEEDBACK_GRID,
    "sparse_topk_same_budget_feedback": POLICY_SPARSE_TOPK_SAME_BUDGET_FEEDBACK_GRID,
    "sparse_topk_feedback": POLICY_SPARSE_TOPK_FEEDBACK_GRID,
    "active_sparse_feedback": POLICY_SPARSE_TOPK_FEEDBACK_GRID,
    "coverage_sparse_topk_feedback": POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
    "coverage_sparse_feedback": POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
    "coverage_topk_feedback": POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
    "count_only_mask_correction": POLICY_COUNT_ONLY_MASK_CORRECTION_FEEDBACK_GRID,
    "count_only_mask_correction_feedback": POLICY_COUNT_ONLY_MASK_CORRECTION_FEEDBACK_GRID,
    "mask_corrected_coverage_aware": POLICY_COUNT_ONLY_MASK_CORRECTION_FEEDBACK_GRID,
    "count_conditioned_invitation_feedback": POLICY_COUNT_CONDITIONED_INVITATION_FEEDBACK_GRID,
    "posterior_invitation_feedback": POLICY_COUNT_CONDITIONED_INVITATION_FEEDBACK_GRID,
    "posterior_greedy_feedback": POLICY_POSTERIOR_GREEDY_FEEDBACK_GRID,
    "posterior_greedy_probe_feedback": POLICY_POSTERIOR_GREEDY_FEEDBACK_GRID,
    "posterior_greedy_invitation_feedback": POLICY_POSTERIOR_GREEDY_INVITATION_FEEDBACK_GRID,
    "posterior_greedy_posterior_invitation_feedback": POLICY_POSTERIOR_GREEDY_INVITATION_FEEDBACK_GRID,
    "posterior_guided_feedback": POLICY_POSTERIOR_GUIDED_COUNT_REFINE_FEEDBACK_GRID,
    "posterior_guided_count_refine_feedback": POLICY_POSTERIOR_GUIDED_COUNT_REFINE_FEEDBACK_GRID,
    "posterior_count_refined_feedback": POLICY_POSTERIOR_GUIDED_COUNT_REFINE_FEEDBACK_GRID,
    "neighbor_coverage_sparse_topk_feedback": POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
    "coverage_neighbor_sparse_topk_feedback": POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
    "neighbor_coverage_sparse_feedback": POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
    "adaptive_sparse_topk_feedback": POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
    "adaptive_sparse_feedback": POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
    "adaptive_sparse_topk_v2_feedback": POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
    "adaptive_sparse_v2_feedback": POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
    "adaptive_sparse_topk_v3_feedback": POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
    "adaptive_sparse_v3_feedback": POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
    "learned_sparse_shortlist_feedback": POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
    "learned_shortlist_feedback": POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
    "learned_set_shortlist_feedback": POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
    "learned_subset_shortlist_feedback": POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
    "full_stale_exhaustive": POLICY_FULL_STALE_EXHAUSTIVE,
    "full_current_oracle": POLICY_FULL_CURRENT_ORACLE,
    "oracle_irs_with_stale_invitation": POLICY_ORACLE_IRS_STALE_INVITATION,
    "deployable_irs_with_oracle_invitation": POLICY_DEPLOYABLE_IRS_ORACLE_INVITATION,
    "oracle_irs_with_oracle_invitation": POLICY_ORACLE_IRS_ORACLE_INVITATION,
    "temporal_deviation_oracle": POLICY_TEMPORAL_DEVIATION_ORACLE_GRID,
    "execution_oracle": POLICY_EXECUTION_ORACLE,
}


def policy_label(
    policy_name,
    budget=0,
    gain_margin=1.0,
    power_margin=1.0,
    risk_weight=0.0,
    risk_invite_threshold=0.0,
    opportunity_failure_cost=0.0,
    opportunity_missed_cost=0.0,
    opportunity_deadline_gain=0.0,
    opportunity_backlog_gain=0.0,
    temporal_reliability_z=0.0,
    sparse_topk_seed_multiplier=2.0,
    sparse_topk_fraction=0.75,
    sparse_topk_show_params=False,
    coverage_sparse_weight=0.5,
    coverage_sparse_power_weight=0.0,
    adaptive_sparse_base_multiplier=2.0,
    adaptive_sparse_expanded_multiplier=3.0,
    adaptive_sparse_margin_threshold=0.05,
    adaptive_sparse_v2_preview_cost=0.002,
    adaptive_sparse_v3_neighbor_radius=1,
    adaptive_sparse_v3_neighbor_count=2,
    adaptive_sparse_v3_history_count=1,
    learned_shortlist_extra_count=1,
    posterior_sample_count=64,
    posterior_uncertainty_scale=1.0,
    posterior_probe_uncertainty_weight=0.0,
    posterior_count_refinement_strength=1.0,
    posterior_count_noise_std_scale=1.0,
    posterior_mean_mode="ar1_predict",
    posterior_invitation_rule="posterior_mean_topk",
    posterior_invitation_threshold=0.5,
    probing_policy="coverage_aware",
    adaptive_sparse_show_params=False,
):
    """处理策略、标签相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if policy_name in {
        limited.POLICY_RANDOM_PROBE,
        limited.POLICY_ROTATING_GRID,
        limited.POLICY_ROBUST_ROTATING_GRID,
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
        POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
        POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
        POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID,
        POLICY_AR1_PREDICT_ROTATING_GRID,
        POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID,
        POLICY_ROTATING_FEEDBACK_CONFIRM_GRID,
        POLICY_ROTATING_SAME_BUDGET_FEEDBACK_GRID,
        POLICY_STALE_TOPK_FEEDBACK_GRID,
        POLICY_STALE_TOPK_SAME_BUDGET_FEEDBACK_GRID,
        POLICY_ACTIVE_DIVERSE_FEEDBACK_GRID,
        POLICY_DIVERSITY_ONLY_FILL_FEEDBACK_GRID,
        POLICY_COVERAGE_ONLY_FILL_FEEDBACK_GRID,
        POLICY_RANDOM_SAME_BUDGET_FEEDBACK_GRID,
        POLICY_SPARSE_TOPK_FEEDBACK_GRID,
        POLICY_SPARSE_TOPK_SAME_BUDGET_FEEDBACK_GRID,
        POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
        POLICY_COUNT_ONLY_MASK_CORRECTION_FEEDBACK_GRID,
        POLICY_COUNT_CONDITIONED_INVITATION_FEEDBACK_GRID,
        POLICY_POSTERIOR_GREEDY_FEEDBACK_GRID,
        POLICY_POSTERIOR_GREEDY_INVITATION_FEEDBACK_GRID,
        POLICY_POSTERIOR_GUIDED_COUNT_REFINE_FEEDBACK_GRID,
        POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID,
        POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID,
        POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID,
        POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID,
        POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID,
        POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID,
        POLICY_TEMPORAL_DEVIATION_ORACLE_GRID,
        POLICY_MS_AIRCOMP_WITHOUT_IRS,
        POLICY_RANDOM_IRS,
        POLICY_FULL_STALE_EXHAUSTIVE,
        POLICY_FULL_CURRENT_ORACLE,
        POLICY_ORACLE_IRS_STALE_INVITATION,
        POLICY_DEPLOYABLE_IRS_ORACLE_INVITATION,
        POLICY_ORACLE_IRS_ORACLE_INVITATION,
    }:
        label = f"{policy_name} B={int(budget)}"
    else:
        label = policy_name
    if policy_name == limited.POLICY_ROBUST_ROTATING_GRID:
        label += f" gm={gain_margin:g} pm={power_margin:g}"
    if policy_name in {
        limited.POLICY_RISK_AWARE_ROTATING_GRID,
        limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
        POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
        POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
    }:
        label += f" rw={risk_weight:g} rt={risk_invite_threshold:g}"
    if policy_name == POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID:
        label += (
            f" fc={opportunity_failure_cost:g} mc={opportunity_missed_cost:g}"
            f" dg={opportunity_deadline_gain:g} bg={opportunity_backlog_gain:g}"
        )
    if policy_name == POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID:
        label += f" rw={risk_weight:g} qz={temporal_reliability_z:g}"
    if policy_name == POLICY_SPARSE_TOPK_FEEDBACK_GRID and (
        sparse_topk_show_params
        or float(sparse_topk_seed_multiplier) != 2.0
        or float(sparse_topk_fraction) != 0.75
    ):
        label += f" sm={sparse_topk_seed_multiplier:g} tf={sparse_topk_fraction:g}"
    if policy_name == POLICY_SPARSE_TOPK_SAME_BUDGET_FEEDBACK_GRID and (
        sparse_topk_show_params
        or float(sparse_topk_seed_multiplier) != 2.0
        or float(sparse_topk_fraction) != 0.75
    ):
        label += f" sm={sparse_topk_seed_multiplier:g} tf={sparse_topk_fraction:g}"
    if policy_name == POLICY_COVERAGE_ONLY_FILL_FEEDBACK_GRID:
        label += f" sm={sparse_topk_seed_multiplier:g}"
    if policy_name == POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID:
        label += (
            f" sm={sparse_topk_seed_multiplier:g}"
            f" tf={sparse_topk_fraction:g}"
            f" cw={coverage_sparse_weight:g}"
        )
        label += f" cpw={coverage_sparse_power_weight:g}"
    if policy_name == POLICY_COUNT_ONLY_MASK_CORRECTION_FEEDBACK_GRID:
        label += (
            f" sm={sparse_topk_seed_multiplier:g}"
            f" tf={sparse_topk_fraction:g}"
            f" cw={coverage_sparse_weight:g}"
            f" cpw={coverage_sparse_power_weight:g}"
            f" mc=count_only"
        )
    if policy_name == POLICY_COUNT_CONDITIONED_INVITATION_FEEDBACK_GRID:
        label += (
            f" sm={sparse_topk_seed_multiplier:g}"
            f" tf={sparse_topk_fraction:g}"
            f" cw={coverage_sparse_weight:g}"
            f" cpw={coverage_sparse_power_weight:g}"
            f" ir=posterior_top_y"
        )
        if str(probing_policy) != "coverage_aware":
            label += f" pp={probing_policy}"
    if policy_name == POLICY_DEPLOYABLE_IRS_ORACLE_INVITATION:
        label += (
            f" sm={sparse_topk_seed_multiplier:g}"
            f" tf={sparse_topk_fraction:g}"
            f" cw={coverage_sparse_weight:g}"
            f" cpw={coverage_sparse_power_weight:g}"
        )
    if policy_name == POLICY_RANDOM_SAME_BUDGET_FEEDBACK_GRID:
        label += f" sm={sparse_topk_seed_multiplier:g}"
    if policy_name == POLICY_POSTERIOR_GUIDED_COUNT_REFINE_FEEDBACK_GRID:
        label += (
            f" sm={sparse_topk_seed_multiplier:g}"
            f" tf={sparse_topk_fraction:g}"
            f" cw={coverage_sparse_weight:g}"
            f" cpw={coverage_sparse_power_weight:g}"
            f" ps={int(posterior_sample_count)}"
            f" us={posterior_uncertainty_scale:g}"
            f" puw={posterior_probe_uncertainty_weight:g}"
            f" cr={posterior_count_refinement_strength:g}"
            f" cns={posterior_count_noise_std_scale:g}"
            f" pm={posterior_mean_mode}"
            f" ir={posterior_invitation_rule}"
        )
        if float(posterior_invitation_threshold) != 0.5:
            label += f" it={posterior_invitation_threshold:g}"
    if policy_name == POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID:
        label += (
            f" sm={sparse_topk_seed_multiplier:g}"
            f" tf={sparse_topk_fraction:g}"
            f" cw={coverage_sparse_weight:g}"
            f" cpw={coverage_sparse_power_weight:g}"
            f" nr={int(adaptive_sparse_v3_neighbor_radius)}"
            f" nc={int(adaptive_sparse_v3_neighbor_count)}"
        )
    if policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID and (
        adaptive_sparse_show_params
        or float(adaptive_sparse_base_multiplier) != 2.0
        or float(adaptive_sparse_expanded_multiplier) != 3.0
        or float(adaptive_sparse_margin_threshold) != 0.05
        or float(sparse_topk_fraction) != 0.75
    ):
        label += (
            f" bm={adaptive_sparse_base_multiplier:g}"
            f" em={adaptive_sparse_expanded_multiplier:g}"
            f" mt={adaptive_sparse_margin_threshold:g}"
            f" tf={sparse_topk_fraction:g}"
        )
    if policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID:
        label += (
            f" bm={adaptive_sparse_base_multiplier:g}"
            f" em={adaptive_sparse_expanded_multiplier:g}"
            f" mt={adaptive_sparse_margin_threshold:g}"
            f" pc={adaptive_sparse_v2_preview_cost:g}"
            f" tf={sparse_topk_fraction:g}"
        )
    if policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID:
        label += (
            f" bm={adaptive_sparse_base_multiplier:g}"
            f" nr={int(adaptive_sparse_v3_neighbor_radius)}"
            f" nc={int(adaptive_sparse_v3_neighbor_count)}"
            f" hc={int(adaptive_sparse_v3_history_count)}"
            f" tf={sparse_topk_fraction:g}"
        )
    if policy_name == POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID:
        label += (
            f" bm={adaptive_sparse_base_multiplier:g}"
            f" ex={int(learned_shortlist_extra_count)}"
            f" tf={sparse_topk_fraction:g}"
        )
    if policy_name == POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID:
        label += (
            f" bm={adaptive_sparse_base_multiplier:g}"
            f" maxex={int(learned_shortlist_extra_count)}"
            f" tf={sparse_topk_fraction:g}"
        )
    return label


def policy_configs(args):
    """处理策略、configs相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    configs = []
    posterior_probe_budgets = (
        [int(args.posterior_probe_budget)]
        if int(getattr(args, "posterior_probe_budget", 0)) > 0
        else args.probe_budgets
    )
    for alias in args.policies:
        policy_name = POLICY_CHOICES[alias]
        if policy_name == POLICY_EXECUTION_ORACLE:
            configs.append({"policy_name": policy_name, "budget": args.num_codebook_states})
        elif policy_name in {
            POLICY_FULL_CURRENT_ORACLE,
            POLICY_ORACLE_IRS_ORACLE_INVITATION,
            POLICY_ORACLE_IRS_STALE_INVITATION,
            POLICY_FULL_STALE_EXHAUSTIVE,
        }:
            configs.append({"policy_name": policy_name, "budget": args.num_codebook_states})
        elif policy_name == POLICY_RANDOM_IRS:
            configs.append({"policy_name": policy_name, "budget": 1})
        elif policy_name == POLICY_MS_AIRCOMP_WITHOUT_IRS:
            configs.append({"policy_name": policy_name, "budget": 0})
        elif policy_name in {
            limited.POLICY_NO_IRS,
            limited.POLICY_FIXED_IRS,
            limited.POLICY_EXACT_GREEDY,
            limited.POLICY_EST_GREEDY,
        }:
            budget = args.num_codebook_states if policy_name in {
                limited.POLICY_EXACT_GREEDY,
                limited.POLICY_EST_GREEDY,
            } else 0
            configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == limited.POLICY_ROBUST_ROTATING_GRID:
            for budget in args.probe_budgets:
                for gain_margin in args.robust_gain_margins:
                    for power_margin in args.robust_power_margins:
                        configs.append(
                            {
                                "policy_name": policy_name,
                                "budget": budget,
                                "gain_margin": gain_margin,
                                "power_margin": power_margin,
                            }
                        )
        elif policy_name in {
            limited.POLICY_RISK_AWARE_ROTATING_GRID,
            POLICY_EXECUTION_RISK_AWARE_ROTATING_GRID,
        }:
            for budget in args.probe_budgets:
                for risk_weight in args.risk_weights:
                    for risk_power_weight in args.risk_power_weights:
                        for risk_invite_threshold in args.risk_invite_thresholds:
                            configs.append(
                                {
                                    "policy_name": policy_name,
                                    "budget": budget,
                                    "risk_weight": risk_weight,
                                    "risk_power_weight": risk_power_weight,
                                    "risk_invite_threshold": risk_invite_threshold,
                                }
                            )
        elif policy_name in {
            limited.POLICY_ADAPTIVE_RISK_AWARE_ROTATING_GRID,
            POLICY_ADAPTIVE_EXECUTION_RISK_AWARE_ROTATING_GRID,
        }:
            for budget in args.probe_budgets:
                for adaptive_risk_base_weight in args.adaptive_risk_base_weights:
                    for risk_power_weight in args.risk_power_weights:
                        for risk_invite_threshold in args.risk_invite_thresholds:
                            configs.append(
                                {
                                    "policy_name": policy_name,
                                    "budget": budget,
                                    "risk_weight": adaptive_risk_base_weight,
                                    "adaptive_risk_base_weight": adaptive_risk_base_weight,
                                    "risk_power_weight": risk_power_weight,
                                    "risk_invite_threshold": risk_invite_threshold,
                                }
                            )
        elif policy_name == POLICY_OPPORTUNITY_EXECUTION_RISK_ROTATING_GRID:
            for budget in args.probe_budgets:
                for opportunity_failure_cost in args.opportunity_failure_costs:
                    for opportunity_missed_cost in args.opportunity_missed_costs:
                        for opportunity_deadline_gain in args.opportunity_deadline_gains:
                            for opportunity_backlog_gain in args.opportunity_backlog_gains:
                                for risk_power_weight in args.risk_power_weights:
                                    configs.append(
                                        {
                                            "policy_name": policy_name,
                                            "budget": budget,
                                            "risk_power_weight": risk_power_weight,
                                            "opportunity_failure_cost": opportunity_failure_cost,
                                            "opportunity_missed_cost": opportunity_missed_cost,
                                            "opportunity_deadline_gain": opportunity_deadline_gain,
                                            "opportunity_backlog_gain": opportunity_backlog_gain,
                                        }
                                    )
        elif policy_name == POLICY_AR1_PREDICT_ROTATING_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_ROTATING_FEEDBACK_CONFIRM_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_ROTATING_SAME_BUDGET_FEEDBACK_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_STALE_TOPK_FEEDBACK_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_STALE_TOPK_SAME_BUDGET_FEEDBACK_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_ACTIVE_DIVERSE_FEEDBACK_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_DIVERSITY_ONLY_FILL_FEEDBACK_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_COVERAGE_ONLY_FILL_FEEDBACK_GRID:
            for budget in args.probe_budgets:
                for seed_multiplier in args.sparse_topk_seed_multipliers:
                    configs.append(
                        {
                            "policy_name": policy_name,
                            "budget": budget,
                            "sparse_topk_seed_multiplier": seed_multiplier,
                        }
                    )
        elif policy_name == POLICY_RANDOM_SAME_BUDGET_FEEDBACK_GRID:
            for budget in args.probe_budgets:
                for seed_multiplier in args.sparse_topk_seed_multipliers:
                    configs.append(
                        {
                            "policy_name": policy_name,
                            "budget": budget,
                            "sparse_topk_seed_multiplier": seed_multiplier,
                        }
                    )
        elif policy_name == POLICY_SPARSE_TOPK_FEEDBACK_GRID:
            show_sparse_params = (
                len(args.sparse_topk_seed_multipliers) > 1
                or len(args.sparse_topk_fractions) > 1
                or args.sparse_topk_seed_multipliers[0] != 2.0
                or args.sparse_topk_fractions[0] != 0.75
            )
            for budget in args.probe_budgets:
                for seed_multiplier in args.sparse_topk_seed_multipliers:
                    for topk_fraction in args.sparse_topk_fractions:
                        configs.append(
                            {
                                "policy_name": policy_name,
                                "budget": budget,
                                "sparse_topk_seed_multiplier": seed_multiplier,
                                "sparse_topk_fraction": topk_fraction,
                                "sparse_topk_show_params": show_sparse_params,
                            }
                        )
        elif policy_name == POLICY_SPARSE_TOPK_SAME_BUDGET_FEEDBACK_GRID:
            show_sparse_params = (
                len(args.sparse_topk_seed_multipliers) > 1
                or len(args.sparse_topk_fractions) > 1
                or args.sparse_topk_seed_multipliers[0] != 2.0
                or args.sparse_topk_fractions[0] != 0.75
            )
            for budget in args.probe_budgets:
                for seed_multiplier in args.sparse_topk_seed_multipliers:
                    for topk_fraction in args.sparse_topk_fractions:
                        configs.append(
                            {
                                "policy_name": policy_name,
                                "budget": budget,
                                "sparse_topk_seed_multiplier": seed_multiplier,
                                "sparse_topk_fraction": topk_fraction,
                                "sparse_topk_show_params": show_sparse_params,
                            }
                        )
        elif policy_name == POLICY_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID:
            show_sparse_params = (
                len(args.sparse_topk_seed_multipliers) > 1
                or len(args.sparse_topk_fractions) > 1
                or len(args.coverage_sparse_weights) > 1
                or len(args.coverage_sparse_power_weights) > 1
                or args.sparse_topk_seed_multipliers[0] != 3.0
                or args.sparse_topk_fractions[0] != 0.75
                or args.coverage_sparse_weights[0] != 0.5
                or args.coverage_sparse_power_weights[0] != 0.0
            )
            for budget in args.probe_budgets:
                for seed_multiplier in args.sparse_topk_seed_multipliers:
                    for topk_fraction in args.sparse_topk_fractions:
                        for coverage_weight in args.coverage_sparse_weights:
                            for coverage_power_weight in args.coverage_sparse_power_weights:
                                configs.append(
                                    {
                                        "policy_name": policy_name,
                                        "budget": budget,
                                        "sparse_topk_seed_multiplier": seed_multiplier,
                                        "sparse_topk_fraction": topk_fraction,
                                        "coverage_sparse_weight": coverage_weight,
                                        "coverage_sparse_power_weight": coverage_power_weight,
                                                "sparse_topk_show_params": show_sparse_params,
                                            }
                                        )
        elif policy_name == POLICY_COUNT_ONLY_MASK_CORRECTION_FEEDBACK_GRID:
            show_sparse_params = (
                len(args.sparse_topk_seed_multipliers) > 1
                or len(args.sparse_topk_fractions) > 1
                or len(args.coverage_sparse_weights) > 1
                or len(args.coverage_sparse_power_weights) > 1
                or args.sparse_topk_seed_multipliers[0] != 3.0
                or args.sparse_topk_fractions[0] != 0.75
                or args.coverage_sparse_weights[0] != 0.5
                or args.coverage_sparse_power_weights[0] != 0.0
            )
            for budget in args.probe_budgets:
                for seed_multiplier in args.sparse_topk_seed_multipliers:
                    for topk_fraction in args.sparse_topk_fractions:
                        for coverage_weight in args.coverage_sparse_weights:
                            for coverage_power_weight in args.coverage_sparse_power_weights:
                                configs.append(
                                    {
                                        "policy_name": policy_name,
                                        "budget": budget,
                                        "sparse_topk_seed_multiplier": seed_multiplier,
                                        "sparse_topk_fraction": topk_fraction,
                                        "coverage_sparse_weight": coverage_weight,
                                        "coverage_sparse_power_weight": coverage_power_weight,
                                        "sparse_topk_show_params": show_sparse_params,
                                    }
                                )
        elif policy_name == POLICY_DEPLOYABLE_IRS_ORACLE_INVITATION:
            for budget in args.probe_budgets:
                for seed_multiplier in args.sparse_topk_seed_multipliers:
                    for topk_fraction in args.sparse_topk_fractions:
                        for coverage_weight in args.coverage_sparse_weights:
                            for coverage_power_weight in args.coverage_sparse_power_weights:
                                configs.append(
                                    {
                                        "policy_name": policy_name,
                                        "budget": budget,
                                        "sparse_topk_seed_multiplier": seed_multiplier,
                                        "sparse_topk_fraction": topk_fraction,
                                        "coverage_sparse_weight": coverage_weight,
                                        "coverage_sparse_power_weight": coverage_power_weight,
                                    }
                                )
        elif policy_name == POLICY_COUNT_CONDITIONED_INVITATION_FEEDBACK_GRID:
            budgets = (
                posterior_probe_budgets
                if getattr(args, "probing_policy", "coverage_aware") == "posterior_greedy"
                else args.probe_budgets
            )
            for budget in budgets:
                for seed_multiplier in args.sparse_topk_seed_multipliers:
                    for topk_fraction in args.sparse_topk_fractions:
                        for coverage_weight in args.coverage_sparse_weights:
                            for coverage_power_weight in args.coverage_sparse_power_weights:
                                configs.append(
                                    {
                                        "policy_name": policy_name,
                                        "budget": budget,
                                        "sparse_topk_seed_multiplier": seed_multiplier,
                                        "sparse_topk_fraction": topk_fraction,
                                        "coverage_sparse_weight": coverage_weight,
                                        "coverage_sparse_power_weight": coverage_power_weight,
                                        "probing_policy": getattr(args, "probing_policy", "coverage_aware"),
                                    }
                                )
        elif policy_name in {
            POLICY_POSTERIOR_GREEDY_FEEDBACK_GRID,
            POLICY_POSTERIOR_GREEDY_INVITATION_FEEDBACK_GRID,
        }:
            for budget in posterior_probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_POSTERIOR_GUIDED_COUNT_REFINE_FEEDBACK_GRID:
            for budget in args.probe_budgets:
                for seed_multiplier in args.sparse_topk_seed_multipliers:
                    for topk_fraction in args.sparse_topk_fractions:
                        for coverage_weight in args.coverage_sparse_weights:
                            for coverage_power_weight in args.coverage_sparse_power_weights:
                                for posterior_sample_count in args.posterior_sample_counts:
                                    for posterior_uncertainty_scale in args.posterior_uncertainty_scales:
                                        for posterior_probe_uncertainty_weight in args.posterior_probe_uncertainty_weights:
                                            for posterior_count_refinement_strength in args.posterior_count_refinement_strengths:
                                                configs.append(
                                                    {
                                                        "policy_name": policy_name,
                                                        "budget": budget,
                                                        "sparse_topk_seed_multiplier": seed_multiplier,
                                                        "sparse_topk_fraction": topk_fraction,
                                                        "coverage_sparse_weight": coverage_weight,
                                                        "coverage_sparse_power_weight": coverage_power_weight,
                                                        "posterior_sample_count": posterior_sample_count,
                                                        "posterior_uncertainty_scale": posterior_uncertainty_scale,
                                                        "posterior_probe_uncertainty_weight": posterior_probe_uncertainty_weight,
                                                        "posterior_count_refinement_strength": posterior_count_refinement_strength,
                                                        "posterior_count_noise_std_scale": args.posterior_count_noise_std_scale,
                                                        "posterior_mean_mode": args.posterior_mean_mode,
                                                        "posterior_invitation_rule": args.posterior_invitation_rule,
                                                        "posterior_invitation_threshold": args.posterior_invitation_threshold,
                                                    }
                                                )
        elif policy_name == POLICY_NEIGHBOR_COVERAGE_SPARSE_TOPK_FEEDBACK_GRID:
            show_sparse_params = (
                len(args.sparse_topk_seed_multipliers) > 1
                or len(args.sparse_topk_fractions) > 1
                or len(args.coverage_sparse_weights) > 1
                or len(args.coverage_sparse_power_weights) > 1
                or args.sparse_topk_seed_multipliers[0] != 4.1
                or args.sparse_topk_fractions[0] != 0.75
                or args.coverage_sparse_weights[0] != 0.5
                or args.coverage_sparse_power_weights[0] != 0.0
            )
            for budget in args.probe_budgets:
                for seed_multiplier in args.sparse_topk_seed_multipliers:
                    for topk_fraction in args.sparse_topk_fractions:
                        for coverage_weight in args.coverage_sparse_weights:
                            for coverage_power_weight in args.coverage_sparse_power_weights:
                                configs.append(
                                    {
                                        "policy_name": policy_name,
                                        "budget": budget,
                                        "sparse_topk_seed_multiplier": seed_multiplier,
                                        "sparse_topk_fraction": topk_fraction,
                                        "coverage_sparse_weight": coverage_weight,
                                        "coverage_sparse_power_weight": coverage_power_weight,
                                        "adaptive_sparse_v3_neighbor_radius": args.adaptive_sparse_v3_neighbor_radius,
                                        "adaptive_sparse_v3_neighbor_count": args.adaptive_sparse_v3_neighbor_count,
                                        "sparse_topk_show_params": show_sparse_params,
                                    }
                                )
        elif policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_FEEDBACK_GRID:
            show_adaptive_params = (
                len(args.adaptive_sparse_margin_thresholds) > 1
                or len(args.sparse_topk_fractions) > 1
                or args.adaptive_sparse_base_multiplier != 2.0
                or args.adaptive_sparse_expanded_multiplier != 3.0
                or args.adaptive_sparse_margin_thresholds[0] != 0.05
                or args.sparse_topk_fractions[0] != 0.75
            )
            for budget in args.probe_budgets:
                for margin_threshold in args.adaptive_sparse_margin_thresholds:
                    for topk_fraction in args.sparse_topk_fractions:
                        configs.append(
                            {
                                "policy_name": policy_name,
                                "budget": budget,
                                "sparse_topk_fraction": topk_fraction,
                                "adaptive_sparse_base_multiplier": args.adaptive_sparse_base_multiplier,
                                "adaptive_sparse_expanded_multiplier": args.adaptive_sparse_expanded_multiplier,
                                "adaptive_sparse_margin_threshold": margin_threshold,
                                "adaptive_sparse_show_params": show_adaptive_params,
                            }
                        )
        elif policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V2_FEEDBACK_GRID:
            show_adaptive_params = (
                len(args.adaptive_sparse_margin_thresholds) > 1
                or len(args.sparse_topk_fractions) > 1
                or len(args.adaptive_sparse_v2_preview_costs) > 1
                or args.adaptive_sparse_base_multiplier != 2.0
                or args.adaptive_sparse_expanded_multiplier != 3.0
                or args.adaptive_sparse_margin_thresholds[0] != 0.05
                or args.sparse_topk_fractions[0] != 0.75
            )
            for budget in args.probe_budgets:
                for margin_threshold in args.adaptive_sparse_margin_thresholds:
                    for preview_cost in args.adaptive_sparse_v2_preview_costs:
                        for topk_fraction in args.sparse_topk_fractions:
                            configs.append(
                                {
                                    "policy_name": policy_name,
                                    "budget": budget,
                                    "sparse_topk_fraction": topk_fraction,
                                    "adaptive_sparse_base_multiplier": args.adaptive_sparse_base_multiplier,
                                    "adaptive_sparse_expanded_multiplier": args.adaptive_sparse_expanded_multiplier,
                                    "adaptive_sparse_margin_threshold": margin_threshold,
                                    "adaptive_sparse_v2_preview_cost": preview_cost,
                                    "adaptive_sparse_show_params": show_adaptive_params,
                                }
                            )
        elif policy_name == POLICY_ADAPTIVE_SPARSE_TOPK_V3_FEEDBACK_GRID:
            show_adaptive_params = (
                len(args.sparse_topk_fractions) > 1
                or args.adaptive_sparse_base_multiplier != 2.0
                or args.sparse_topk_fractions[0] != 0.75
                or args.adaptive_sparse_v3_neighbor_radius != 1
                or args.adaptive_sparse_v3_neighbor_count != 2
                or args.adaptive_sparse_v3_history_count != 1
            )
            for budget in args.probe_budgets:
                for topk_fraction in args.sparse_topk_fractions:
                    configs.append(
                        {
                            "policy_name": policy_name,
                            "budget": budget,
                            "sparse_topk_fraction": topk_fraction,
                            "adaptive_sparse_base_multiplier": args.adaptive_sparse_base_multiplier,
                            "adaptive_sparse_expanded_multiplier": args.adaptive_sparse_base_multiplier,
                            "adaptive_sparse_v3_neighbor_radius": args.adaptive_sparse_v3_neighbor_radius,
                            "adaptive_sparse_v3_neighbor_count": args.adaptive_sparse_v3_neighbor_count,
                            "adaptive_sparse_v3_history_count": args.adaptive_sparse_v3_history_count,
                            "adaptive_sparse_show_params": show_adaptive_params,
                        }
                    )
        elif policy_name == POLICY_LEARNED_SPARSE_SHORTLIST_FEEDBACK_GRID:
            show_adaptive_params = (
                len(args.sparse_topk_fractions) > 1
                or len(args.learned_shortlist_extra_counts) > 1
                or args.adaptive_sparse_base_multiplier != 2.0
                or args.sparse_topk_fractions[0] != 0.75
                or args.learned_shortlist_extra_counts[0] != 1
            )
            for budget in args.probe_budgets:
                for extra_count in args.learned_shortlist_extra_counts:
                    for topk_fraction in args.sparse_topk_fractions:
                        configs.append(
                            {
                                "policy_name": policy_name,
                                "budget": budget,
                                "sparse_topk_fraction": topk_fraction,
                                "adaptive_sparse_base_multiplier": args.adaptive_sparse_base_multiplier,
                                "adaptive_sparse_expanded_multiplier": args.adaptive_sparse_base_multiplier,
                                "learned_shortlist_extra_count": extra_count,
                                "adaptive_sparse_show_params": show_adaptive_params,
                            }
                        )
        elif policy_name == POLICY_LEARNED_SET_SHORTLIST_FEEDBACK_GRID:
            show_adaptive_params = (
                len(args.sparse_topk_fractions) > 1
                or len(args.learned_set_extra_counts) > 1
                or args.adaptive_sparse_base_multiplier != 2.0
                or args.sparse_topk_fractions[0] != 0.75
                or args.learned_set_extra_counts[0] != 1
            )
            for budget in args.probe_budgets:
                for extra_count in args.learned_set_extra_counts:
                    for topk_fraction in args.sparse_topk_fractions:
                        configs.append(
                            {
                                "policy_name": policy_name,
                                "budget": budget,
                                "sparse_topk_fraction": topk_fraction,
                                "adaptive_sparse_base_multiplier": args.adaptive_sparse_base_multiplier,
                                "adaptive_sparse_expanded_multiplier": args.adaptive_sparse_base_multiplier,
                                "learned_shortlist_extra_count": extra_count,
                                "adaptive_sparse_show_params": show_adaptive_params,
                            }
                        )
        elif policy_name == POLICY_TEMPORAL_DEVIATION_ORACLE_GRID:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
        elif policy_name == POLICY_TEMPORAL_RELIABILITY_ROTATING_GRID:
            for budget in args.probe_budgets:
                for risk_weight in args.risk_weights:
                    for risk_power_weight in args.risk_power_weights:
                        for temporal_reliability_z in args.temporal_reliability_z_values:
                            configs.append(
                                {
                                    "policy_name": policy_name,
                                    "budget": budget,
                                    "risk_weight": risk_weight,
                                    "risk_power_weight": risk_power_weight,
                                    "temporal_reliability_z": temporal_reliability_z,
                                }
                            )
        else:
            for budget in args.probe_budgets:
                configs.append({"policy_name": policy_name, "budget": budget})
    return configs


def mismatch_scenarios(args):
    """处理mismatch、scenarios相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    for mismatch_model in args.mismatch_models:
        if mismatch_model == MISMATCH_INDEPENDENT:
            yield mismatch_model, 0.0, 0
            continue
        for channel_rho in args.channel_rho_values:
            for csi_delay_slots in args.csi_delay_slots:
                yield mismatch_model, float(channel_rho), int(csi_delay_slots)
