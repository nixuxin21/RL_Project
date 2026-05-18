PYTHON ?= ./.venv/bin/python

.PHONY: smoke test pytest-test lint boundary-test regression-test mainline-audit check quick-audit quick-audit-dry-run py-compile help docs policy-comparison policy-comparison-learning policy-comparison-static runtime parameter-sweep noisy-feature-sweep partial-probing-sweep learned-probing learned-feedback-probing learned-temporal-deviation learned-sparse-shortlist-pilot learned-sparse-shortlist-marginal-pilot learned-set-shortlist-pilot learned-execution-value-shortlist-pilot learned-pairwise-shortlist-pilot execution-baseline-summary main-results-analysis coverage-aware-analysis final-invitation-mask-analysis paper-table1 paper-tables paper-figures coverage-b3-failure-diagnosis invitation-mask-correction-pilot invitation-mask-correction-formal invitation-mask-correction-noise-sweep invitation-mask-correction-noise-aware-pilot invitation-mask-correction-noise-aware-formal invitation-mask-rerank-ablation adaptive-feedback-probing probing-cost-tradeoff channel-estimation-sweep limited-csi-sweep execution-mismatch-sweep active-probe-set-pilot sparse-topk-cost-pilot sparse-topk-frontier coverage-sparse-topk-pilot coverage-sparse-topk-frontier coverage-sparse-topk-ablation coverage-sparse-power-pilot coverage-sparse-power-ablation coverage-budget-split-pilot coverage-budget-split-selected coverage-budget-split-formal adaptive-sparse-topk-pilot adaptive-sparse-topk-v2-pilot adaptive-sparse-topk-v3-pilot bandit-feedback-sweep bandit-feedback-stress action-diagnostics

py-compile:
	$(PYTHON) -m py_compile \
		ms_aircomp/__init__.py \
		ms_aircomp/adaptive_sparse_policies.py \
		ms_aircomp/channel_models.py \
		ms_aircomp/confirmation.py \
		ms_aircomp/experiment_utils.py \
		ms_aircomp/execution_candidates.py \
		ms_aircomp/execution_decision_dispatch.py \
		ms_aircomp/execution_output.py \
		ms_aircomp/execution_policies.py \
		ms_aircomp/execution_policy_registry.py \
		ms_aircomp/execution_result_summary.py \
		ms_aircomp/execution_risk_policies.py \
		ms_aircomp/feedback.py \
		ms_aircomp/invitation_mask_correction.py \
		ms_aircomp/learned_shortlist.py \
		ms_aircomp/limited_csi.py \
		ms_aircomp/probe_sets.py \
		ms_aircomp/temporal_policies.py \
		test_env.py \
		train_agent.py \
		train_codebook_aware_agent.py \
		train_greedy_imitation_selector.py \
		train_bandit_feedback_selector.py \
		train_temporal_deviation_selector.py \
		train_learned_sparse_shortlist.py \
		experiments/archive/train_learned_probing_selector.py \
		evaluate_agent.py \
		evaluate_batch.py \
		evaluate_random_irs_baseline.py \
		evaluate_policy_comparison.py \
		evaluate_parameter_sweep.py \
		experiments/archive/evaluate_noisy_feature_sweep.py \
		evaluate_partial_probing_sweep.py \
		experiments/archive/evaluate_probing_cost_tradeoff.py \
		evaluate_channel_estimation_error_sweep.py \
		evaluate_limited_csi_ms_aircomp.py \
		evaluate_execution_channel_mismatch.py \
		evaluate_bandit_feedback_ms_aircomp.py \
		evaluate_bandit_feedback_stress_sweep.py \
		evaluate_adaptive_feedback_probing.py \
		diagnose_policy_actions.py \
		benchmark_policy_runtime.py \
		summarize_execution_baselines.py \
		analyze_main_frontier.py \
		analyze_coverage_aware.py \
		analyze_invitation_mask_final.py \
		generate_paper_tables.py \
		generate_paper_figures.py \
		diagnose_coverage_b3_failures.py \
		evaluate_invitation_mask_correction.py \
		tests/dependency_boundary_checks.py \
		tests/__init__.py \
		tests/test_project_checks.py \
		tests/mainline_artifact_checks.py \
		tests/validation_checks.py \
		tests/smoke_checks.py \
		tests/mainline_regression_checks.py

test: py-compile
	$(PYTHON) tests/smoke_checks.py

pytest-test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

boundary-test:
	$(PYTHON) tests/dependency_boundary_checks.py

regression-test: py-compile
	$(PYTHON) tests/mainline_regression_checks.py

mainline-audit: py-compile
	$(PYTHON) tests/mainline_artifact_checks.py

check: py-compile lint pytest-test

quick-audit: check mainline-audit quick-audit-dry-run

quick-audit-dry-run:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 1 \
		--num-seeds 1 \
		--probe-budgets 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.9 \
		--csi-delay-slots 1 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--policies rotating_feedback_confirm,sparse_topk_feedback \
		--num-nodes 6 \
		--num-slots 3 \
		--num-codebook-states 4 \
		--num-irs-elements 8 \
		--fixed-irs-index 1 \
		--no-plots \
		--output-prefix /tmp/rl_project_quick_execution_mismatch

smoke: test
	$(PYTHON) evaluate_policy_comparison.py \
		--episodes 2 \
		--num-seeds 1 \
		--skip-sac \
		--output /tmp/policy_comparison_smoke.png \
		--csv-output /tmp/policy_comparison_smoke.csv
	$(PYTHON) benchmark_policy_runtime.py \
		--episodes 2 \
		--skip-sac \
		--skip-codebook-aware-sac \
		--output /tmp/runtime_benchmark_smoke.csv
	$(PYTHON) evaluate_limited_csi_ms_aircomp.py \
		--episodes 2 \
		--num-seeds 1 \
		--error-std-values 0.1 \
		--probe-budgets 2 \
		--robust-gain-margins 1.25 \
		--robust-power-margins 0.9 \
		--risk-weights 0.5 \
		--risk-power-weights 0.1 \
		--risk-invite-thresholds 0.5 \
		--adaptive-risk-base-weights 0.5 \
		--output-prefix /tmp/limited_csi_smoke \
		--no-plots
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 1 \
		--num-seeds 1 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0.1 \
		--probe-budgets 1 \
		--policies no_irs,execution_oracle,exact_greedy,rotating,execution_risk_rotating,opportunity_execution_risk_rotating \
		--output-prefix /tmp/execution_mismatch_smoke \
		--no-plots
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 1 \
		--num-seeds 1 \
		--num-slots 3 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.9 \
		--csi-delay-slots 1 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 1 \
		--policies execution_oracle,rotating,ar1_predict_rotating,temporal_reliability_rotating,rotating_feedback_confirm,active_diverse_feedback,sparse_topk_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--risk-weights 0.5 \
		--risk-power-weights 0.1 \
		--temporal-reliability-z-values 1 \
		--output-prefix /tmp/temporal_ar1_smoke \
		--no-plots
	$(PYTHON) evaluate_bandit_feedback_ms_aircomp.py \
		--episodes 2 \
		--num-seeds 1 \
		--feedback-noise-std-values 0.1 \
		--probe-budgets 2 \
		--output-prefix /tmp/bandit_feedback_smoke \
		--no-plots
	$(PYTHON) evaluate_bandit_feedback_stress_sweep.py \
		--episodes 1 \
		--num-seeds 1 \
		--scenarios default \
		--feedback-noise-std-values 0.1 \
		--probe-budgets 1 \
		--baseline-policies no_irs,oracle \
		--probe-policies rotating \
		--output-prefix /tmp/bandit_feedback_stress_smoke \
		--no-plots
	$(PYTHON) train_bandit_feedback_selector.py \
		--scenario short_slots \
		--train-episodes 2 \
		--val-episodes 1 \
		--eval-episodes 1 \
		--num-eval-seeds 1 \
		--epochs 1 \
		--batch-size 4 \
		--feedback-noise-std-values 0.1 \
		--probe-budgets 1 \
		--baseline-policies no_irs,oracle \
		--probe-policies rotating \
		--output-prefix /tmp/learned_feedback_probe_smoke \
		--no-plots
	$(PYTHON) train_temporal_deviation_selector.py \
		--train-episodes 2 \
		--val-episodes 1 \
		--eval-episodes 1 \
		--num-eval-seeds 1 \
		--epochs 1 \
		--batch-size 4 \
		--feature-mode window \
		--gate-margin-thresholds 0,0.01 \
		--dagger-iterations 1 \
		--dagger-episodes 1 \
		--dagger-beta-start 0.5 \
		--dagger-beta-end 0.5 \
		--channel-rho-values 0.9 \
		--csi-delay-slots 1 \
		--probe-budgets 4 \
		--offsets=-1,0,1 \
		--output-prefix /tmp/learned_temporal_deviation_smoke \
		--no-plots
	$(PYTHON) evaluate_adaptive_feedback_probing.py \
		--episodes 1 \
		--num-seeds 1 \
		--scenarios short_slots \
		--feedback-noise-std-values 0.1 \
		--gate-ratios 1.0 \
		--backup-strategies next \
		--probe-budgets 1 \
		--baseline-policies no_irs,oracle \
		--probe-policies rotating \
		--output-prefix /tmp/adaptive_feedback_probe_smoke \
		--no-plots

policy-comparison:
	$(PYTHON) evaluate_policy_comparison.py \
		--episodes 1000 \
		--num-seeds 5 \
		--skip-sac

policy-comparison-learning:
	$(PYTHON) evaluate_policy_comparison.py \
		--episodes 1000 \
		--num-seeds 5 \
		--include-codebook-aware-sac

policy-comparison-static:
	$(PYTHON) evaluate_policy_comparison.py \
		--episodes 1000 \
		--num-seeds 5 \
		--skip-sac \
		--include-fixed-irs-baselines

runtime:
	$(PYTHON) benchmark_policy_runtime.py \
		--episodes 200

parameter-sweep:
	$(PYTHON) evaluate_parameter_sweep.py

noisy-feature-sweep:
	$(PYTHON) experiments/archive/evaluate_noisy_feature_sweep.py

partial-probing-sweep:
	$(PYTHON) evaluate_partial_probing_sweep.py

learned-probing:
	$(PYTHON) experiments/archive/train_learned_probing_selector.py

learned-feedback-probing:
	$(PYTHON) train_bandit_feedback_selector.py

learned-temporal-deviation:
	$(PYTHON) train_temporal_deviation_selector.py

adaptive-feedback-probing:
	$(PYTHON) evaluate_adaptive_feedback_probing.py

probing-cost-tradeoff:
	$(PYTHON) experiments/archive/evaluate_probing_cost_tradeoff.py

channel-estimation-sweep:
	$(PYTHON) evaluate_channel_estimation_error_sweep.py

limited-csi-sweep:
	$(PYTHON) evaluate_limited_csi_ms_aircomp.py

execution-mismatch-sweep:
	$(PYTHON) evaluate_execution_channel_mismatch.py

active-probe-set-pilot:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 300 \
		--num-seeds 3 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-fractions 0.75 \
		--policies execution_oracle,rotating,rotating_feedback_confirm,active_diverse_feedback,sparse_topk_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--output-prefix results/execution_mismatch/active_probe_set_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_tf0p75

sparse-topk-cost-pilot:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 2,4,8 \
		--sparse-topk-seed-multipliers 1,2,3 \
		--sparse-topk-fractions 0.25,0.5,0.75 \
		--policies rotating,sparse_topk_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--output-prefix results/execution_mismatch/sparse_topk_cost_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-3_b2-4-8_sm1-2-3_tf0p25-0p5-0p75 \
		--no-plots

sparse-topk-frontier:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 300 \
		--num-seeds 3 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4,8 \
		--sparse-topk-seed-multipliers 2,3 \
		--sparse-topk-fractions 0.75 \
		--policies rotating,sparse_topk_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--output-prefix results/execution_mismatch/sparse_topk_frontier_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4-8_sm2-3_tf0p75 \
		--no-plots

coverage-sparse-topk-pilot:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 3 \
		--sparse-topk-fractions 0.75 \
		--coverage-sparse-weights 0,0.25,0.5,1,2 \
		--coverage-sparse-power-weights 0 \
		--adaptive-sparse-margin-thresholds 0.05 \
		--adaptive-sparse-v2-preview-costs 0.002,0.005 \
		--policies rotating,sparse_topk_feedback,coverage_sparse_topk_feedback,adaptive_sparse_topk_v2_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--output-prefix results/execution_mismatch/coverage_sparse_topk_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0-0p25-0p5-1-2_cpw0 \
		--no-plots

coverage-sparse-topk-frontier:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 300 \
		--num-seeds 3 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 3 \
		--sparse-topk-fractions 0.75 \
		--coverage-sparse-weights 0.5 \
		--coverage-sparse-power-weights 0 \
		--policies coverage_sparse_topk_feedback \
		--output-prefix results/execution_mismatch/coverage_sparse_topk_frontier_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0p5_cpw0 \
		--no-plots

coverage-sparse-topk-ablation:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 300 \
		--num-seeds 3 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 3 \
		--sparse-topk-fractions 0.75 \
		--coverage-sparse-weights 0,0.25,0.5,1,2 \
		--coverage-sparse-power-weights 0 \
		--policies coverage_sparse_topk_feedback \
		--output-prefix results/execution_mismatch/coverage_sparse_topk_ablation_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0-0p25-0p5-1-2_cpw0 \
		--no-plots

coverage-sparse-power-pilot:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 3 \
		--sparse-topk-fractions 0.75 \
		--coverage-sparse-weights 0.5 \
		--coverage-sparse-power-weights 0,0.02,0.05,0.1,0.2 \
		--policies coverage_sparse_topk_feedback \
		--output-prefix results/execution_mismatch/coverage_sparse_power_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0p5_cpw0-0p02-0p05-0p1-0p2 \
		--no-plots

coverage-sparse-power-ablation:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 300 \
		--num-seeds 3 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 3 \
		--sparse-topk-fractions 0.75 \
		--coverage-sparse-weights 0.5 \
		--coverage-sparse-power-weights 0,0.02,0.05,0.1,0.2 \
		--policies coverage_sparse_topk_feedback \
		--output-prefix results/execution_mismatch/coverage_sparse_power_ablation_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0p5_cpw0-0p02-0p05-0p1-0p2 \
		--no-plots

coverage-budget-split-pilot:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 3,4,5,6,8 \
		--sparse-topk-seed-multipliers 1,1.6,2.2,3,4.1 \
		--sparse-topk-fractions 0.75 \
		--coverage-sparse-weights 0.5 \
		--coverage-sparse-power-weights 0 \
		--policies coverage_sparse_topk_feedback \
		--output-prefix results/execution_mismatch/coverage_budget_split_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3-4-5-6-8_sm1-1p6-2p2-3-4p1_tf0p75_cw0p5_cpw0 \
		--no-plots

coverage-budget-split-selected:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 300 \
		--num-seeds 3 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 3 \
		--sparse-topk-seed-multipliers 4.1 \
		--sparse-topk-fractions 0.75 \
		--coverage-sparse-weights 0.5 \
		--coverage-sparse-power-weights 0 \
		--policies coverage_sparse_topk_feedback \
		--output-prefix results/execution_mismatch/coverage_budget_split_selected_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0 \
		--no-plots

coverage-budget-split-formal: coverage-sparse-topk-frontier coverage-budget-split-selected
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 300 \
		--num-seeds 3 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 5 \
		--sparse-topk-seed-multipliers 2.2 \
		--sparse-topk-fractions 0.75 \
		--coverage-sparse-weights 0.5 \
		--coverage-sparse-power-weights 0 \
		--policies coverage_sparse_topk_feedback \
		--output-prefix results/execution_mismatch/coverage_budget_split_selected_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b5_sm2p2_tf0p75_cw0p5_cpw0 \
		--no-plots
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 300 \
		--num-seeds 3 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 6 \
		--sparse-topk-seed-multipliers 1.6 \
		--sparse-topk-fractions 0.75 \
		--coverage-sparse-weights 0.5 \
		--coverage-sparse-power-weights 0 \
		--policies coverage_sparse_topk_feedback \
		--output-prefix results/execution_mismatch/coverage_budget_split_selected_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b6_sm1p6_tf0p75_cw0p5_cpw0 \
		--no-plots
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 300 \
		--num-seeds 3 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 8 \
		--sparse-topk-seed-multipliers 1 \
		--sparse-topk-fractions 0.75 \
		--coverage-sparse-weights 0.5 \
		--coverage-sparse-power-weights 0 \
		--policies coverage_sparse_topk_feedback \
		--output-prefix results/execution_mismatch/coverage_budget_split_selected_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b8_sm1_tf0p75_cw0p5_cpw0 \
		--no-plots

coverage-b3-failure-diagnosis:
	$(PYTHON) diagnose_coverage_b3_failures.py \
		--episodes 100 \
		--num-seeds 2 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--probe-budget 3 \
		--sparse-topk-seed-multiplier 4.1 \
		--sparse-topk-fraction 0.75 \
		--coverage-sparse-weight 0.5 \
		--coverage-sparse-power-weight 0 \
		--output-prefix results/execution_mismatch/coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0

invitation-mask-correction-pilot:
	$(PYTHON) evaluate_invitation_mask_correction.py \
		--episodes 100 \
		--num-seeds 2 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--probe-budget 3 \
		--sparse-topk-seed-multiplier 4.1 \
		--sparse-topk-fraction 0.75 \
		--coverage-sparse-weight 0.5 \
		--coverage-sparse-power-weight 0 \
		--mask-correction-strengths 0,0.25,0.5,0.75,1 \
		--output-prefix results/execution_mismatch/invitation_mask_correction_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p25-0p5-0p75-1

invitation-mask-correction-formal:
	$(PYTHON) evaluate_invitation_mask_correction.py \
		--episodes 300 \
		--num-seeds 3 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--probe-budget 3 \
		--sparse-topk-seed-multiplier 4.1 \
		--sparse-topk-fraction 0.75 \
		--coverage-sparse-weight 0.5 \
		--coverage-sparse-power-weight 0 \
		--mask-correction-strengths 0,0.75,1 \
		--output-prefix results/execution_mismatch/invitation_mask_correction_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1

invitation-mask-correction-noise-sweep:
	$(PYTHON) evaluate_invitation_mask_correction.py \
		--episodes 300 \
		--num-seeds 3 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--probe-budget 3 \
		--sparse-topk-seed-multiplier 4.1 \
		--sparse-topk-fraction 0.75 \
		--coverage-sparse-weight 0.5 \
		--coverage-sparse-power-weight 0 \
		--mask-correction-strengths 0,0.75,1 \
		--confirmation-feedback-noise-std-values 0,0.02,0.05,0.1 \
		--output-prefix results/execution_mismatch/invitation_mask_correction_noise_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_fbn0-0p02-0p05-0p1 \
		--doc-output docs/INVITATION_MASK_CORRECTION_NOISE.md

invitation-mask-correction-noise-aware-pilot:
	$(PYTHON) evaluate_invitation_mask_correction.py \
		--episodes 100 \
		--num-seeds 2 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--probe-budget 3 \
		--sparse-topk-seed-multiplier 4.1 \
		--sparse-topk-fraction 0.75 \
		--coverage-sparse-weight 0.5 \
		--coverage-sparse-power-weight 0 \
		--mask-correction-strengths 0,0.75,1 \
		--mask-correction-noise-deadband-z-values 0,0.25,0.5,1 \
		--mask-correction-max-delta-values=-1,2,3 \
		--confirmation-feedback-noise-std-values 0,0.02,0.05,0.1 \
		--output-prefix results/execution_mismatch/invitation_mask_correction_noise_aware_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_dbz0-0p25-0p5-1_clipinf-2-3_fbn0-0p02-0p05-0p1 \
		--doc-output docs/INVITATION_MASK_CORRECTION_NOISE_AWARE.md

invitation-mask-correction-noise-aware-formal:
	$(PYTHON) evaluate_invitation_mask_correction.py \
		--episodes 300 \
		--num-seeds 3 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--probe-budget 3 \
		--sparse-topk-seed-multiplier 4.1 \
		--sparse-topk-fraction 0.75 \
		--coverage-sparse-weight 0.5 \
		--coverage-sparse-power-weight 0 \
		--mask-correction-strengths 0,0.75,1 \
		--mask-correction-noise-deadband-z-values 0 \
		--mask-correction-max-delta-values=-1,2 \
		--confirmation-feedback-noise-std-values 0,0.02,0.05,0.1 \
		--output-prefix results/execution_mismatch/invitation_mask_correction_noise_aware_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_clipinf-2_fbn0-0p02-0p05-0p1 \
		--doc-output docs/INVITATION_MASK_CORRECTION_NOISE_AWARE.md

invitation-mask-rerank-ablation:
	$(PYTHON) evaluate_invitation_mask_correction.py \
		--episodes 100 \
		--num-seeds 2 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--probe-budget 3 \
		--sparse-topk-seed-multiplier 4.1 \
		--sparse-topk-fraction 0.75 \
		--coverage-sparse-weight 0.5 \
		--coverage-sparse-power-weight 0 \
		--mask-correction-strengths 0,1 \
		--mask-correction-rerank-modes global_stale_gain,prune_only \
		--output-prefix results/execution_mismatch/invitation_mask_rerank_ablation_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-1_modesglobal-prune \
		--doc-output docs/INVITATION_MASK_RERANK_ABLATION.md

adaptive-sparse-topk-pilot:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 2,3 \
		--sparse-topk-fractions 0.75 \
		--adaptive-sparse-margin-thresholds 0,0.02,0.05,0.1,0.2 \
		--policies rotating,sparse_topk_feedback,adaptive_sparse_topk_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--output-prefix results/execution_mismatch/adaptive_sparse_topk_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_mt0-0p02-0p05-0p1-0p2 \
		--no-plots

adaptive-sparse-topk-v2-pilot:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 2,3 \
		--sparse-topk-fractions 0.75 \
		--adaptive-sparse-margin-thresholds 0.02,0.05 \
		--adaptive-sparse-v2-preview-costs 0,0.002,0.005,0.01 \
		--policies rotating,sparse_topk_feedback,adaptive_sparse_topk_feedback,adaptive_sparse_topk_v2_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--output-prefix results/execution_mismatch/adaptive_sparse_topk_v2_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_mt0p02-0p05_pc0-0p002-0p005-0p01 \
		--no-plots

adaptive-sparse-topk-v3-pilot:
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 2,3 \
		--sparse-topk-fractions 0.75 \
		--adaptive-sparse-v3-neighbor-radius 1 \
		--adaptive-sparse-v3-neighbor-count 2 \
		--adaptive-sparse-v3-history-count 1 \
		--policies rotating,sparse_topk_feedback,adaptive_sparse_topk_v2_feedback,adaptive_sparse_topk_v3_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--output-prefix results/execution_mismatch/adaptive_sparse_topk_v3_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_nr1_nc2_hc1 \
		--no-plots

learned-sparse-shortlist-pilot:
	$(PYTHON) train_learned_sparse_shortlist.py \
		--train-episodes 500 \
		--val-episodes 100 \
		--probe-budget 4 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--base-multiplier 2 \
		--topk-fraction 0.75 \
		--output-prefix results/execution_mismatch/learned_sparse_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 2,3 \
		--sparse-topk-fractions 0.75 \
		--learned-shortlist-model results/execution_mismatch/learned_sparse_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_model.npz \
		--learned-shortlist-extra-counts 1,2 \
		--policies rotating,sparse_topk_feedback,adaptive_sparse_topk_v2_feedback,adaptive_sparse_topk_v3_feedback,learned_sparse_shortlist_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--output-prefix results/execution_mismatch/learned_sparse_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_ex1-2 \
		--no-plots

learned-sparse-shortlist-marginal-pilot:
	$(PYTHON) train_learned_sparse_shortlist.py \
		--train-episodes 500 \
		--val-episodes 100 \
		--probe-budget 4 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--base-multiplier 2 \
		--topk-fraction 0.75 \
		--target-mode marginal \
		--target-extra-count 2 \
		--output-prefix results/execution_mismatch/learned_sparse_shortlist_marginal_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_tex2
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 2,3 \
		--sparse-topk-fractions 0.75 \
		--learned-shortlist-model results/execution_mismatch/learned_sparse_shortlist_marginal_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_tex2_model.npz \
		--learned-shortlist-extra-counts 1,2 \
		--policies rotating,sparse_topk_feedback,adaptive_sparse_topk_v2_feedback,adaptive_sparse_topk_v3_feedback,learned_sparse_shortlist_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--output-prefix results/execution_mismatch/learned_sparse_shortlist_marginal_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_ex1-2 \
		--no-plots

learned-set-shortlist-pilot:
	$(PYTHON) train_learned_sparse_shortlist.py \
		--train-episodes 500 \
		--val-episodes 100 \
		--probe-budget 4 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--base-multiplier 2 \
		--topk-fraction 0.75 \
		--target-mode set_value \
		--target-extra-count 2 \
		--output-prefix results/execution_mismatch/learned_set_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_maxex2
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 2,3 \
		--sparse-topk-fractions 0.75 \
		--learned-set-shortlist-model results/execution_mismatch/learned_set_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_maxex2_model.npz \
		--learned-set-extra-counts 1,2 \
		--policies rotating,sparse_topk_feedback,adaptive_sparse_topk_v2_feedback,learned_sparse_shortlist_feedback,learned_set_shortlist_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--learned-shortlist-model results/execution_mismatch/learned_sparse_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_model.npz \
		--learned-shortlist-extra-counts 2 \
		--output-prefix results/execution_mismatch/learned_set_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2 \
		--no-plots

learned-execution-value-shortlist-pilot:
	$(PYTHON) train_learned_sparse_shortlist.py \
		--train-episodes 500 \
		--val-episodes 100 \
		--probe-budget 4 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--base-multiplier 2 \
		--topk-fraction 0.75 \
		--target-mode execution_value \
		--target-extra-count 2 \
		--label-failure-weight 0.5 \
		--label-missed-weight 0.5 \
		--output-prefix results/execution_mismatch/learned_execution_value_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_maxex2_fw0p5_mw0p5
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 2,3 \
		--sparse-topk-fractions 0.75 \
		--learned-set-shortlist-model results/execution_mismatch/learned_execution_value_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_maxex2_fw0p5_mw0p5_model.npz \
		--learned-set-extra-counts 1,2 \
		--policies rotating,sparse_topk_feedback,adaptive_sparse_topk_v2_feedback,learned_sparse_shortlist_feedback,learned_set_shortlist_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--learned-shortlist-model results/execution_mismatch/learned_sparse_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_model.npz \
		--learned-shortlist-extra-counts 2 \
		--output-prefix results/execution_mismatch/learned_execution_value_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5 \
		--no-plots

learned-pairwise-shortlist-pilot:
	$(PYTHON) train_learned_sparse_shortlist.py \
		--train-episodes 500 \
		--val-episodes 100 \
		--probe-budget 4 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--base-multiplier 2 \
		--topk-fraction 0.75 \
		--target-mode pairwise_execution \
		--target-extra-count 2 \
		--label-failure-weight 0.5 \
		--label-missed-weight 0.5 \
		--label-preview-cost 0.005 \
		--output-prefix results/execution_mismatch/learned_pairwise_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_maxex2_fw0p5_mw0p5_pc0p005
	$(PYTHON) evaluate_execution_channel_mismatch.py \
		--episodes 100 \
		--num-seeds 2 \
		--mismatch-models temporal_ar1 \
		--channel-rho-values 0.7,0.9,0.98 \
		--csi-delay-slots 1,2,3 \
		--decision-error-std-values 0 \
		--execution-error-std-values 0 \
		--probe-budgets 4 \
		--sparse-topk-seed-multipliers 2,3 \
		--sparse-topk-fractions 0.75 \
		--learned-set-shortlist-model results/execution_mismatch/learned_pairwise_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_maxex2_fw0p5_mw0p5_pc0p005_model.npz \
		--learned-set-extra-counts 1,2 \
		--policies rotating,sparse_topk_feedback,adaptive_sparse_topk_v2_feedback,learned_sparse_shortlist_feedback,learned_set_shortlist_feedback,stale_topk_feedback,temporal_deviation_oracle \
		--learned-shortlist-model results/execution_mismatch/learned_sparse_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_model.npz \
		--learned-shortlist-extra-counts 2 \
		--output-prefix results/execution_mismatch/learned_pairwise_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5_pc0p005 \
		--no-plots

execution-baseline-summary:
	$(PYTHON) summarize_execution_baselines.py

main-results-analysis:
	$(PYTHON) analyze_main_frontier.py

coverage-aware-analysis:
	$(PYTHON) analyze_coverage_aware.py

final-invitation-mask-analysis:
	$(PYTHON) analyze_invitation_mask_final.py

paper-table1:
	$(PYTHON) generate_paper_tables.py

paper-tables: paper-table1

paper-figures:
	$(PYTHON) generate_paper_figures.py

bandit-feedback-sweep:
	$(PYTHON) evaluate_bandit_feedback_ms_aircomp.py

bandit-feedback-stress:
	$(PYTHON) evaluate_bandit_feedback_stress_sweep.py

action-diagnostics:
	$(PYTHON) diagnose_policy_actions.py --episodes 1000

docs:
	@printf '%s\n' \
		'Project status: docs/PROJECT_STATUS.md' \
		'Main story: docs/MAIN_STORY.md' \
		'Paper result package: docs/PAPER_RESULT_PACKAGE.md' \
		'Paper structure map: docs/PAPER_STRUCTURE_MAP.md' \
		'Paper figure/table specs: docs/PAPER_FIGURE_TABLE_SPECS.md' \
		'Paper appendix boundary: docs/PAPER_APPENDIX_BOUNDARY.md' \
		'Paper text outline: docs/PAPER_TEXT_OUTLINE.md' \
		'Paper asset gap checklist: docs/PAPER_ASSET_GAP_CHECKLIST.md' \
		'Paper freeze manifest: docs/PAPER_FREEZE_MANIFEST.md' \
		'Environment: docs/ENVIRONMENT.md' \
		'Paper Figure 1 source: docs/figures/figure1_system_flow.mmd' \
		'Paper Table 1 main results: docs/PAPER_TABLE1_MAIN_RESULTS.md' \
		'Paper Table 1 CSV: results/paper/table1_main_results.csv' \
		'Paper Table 1 uncertainty: docs/PAPER_TABLE1_UNCERTAINTY.md' \
		'Paper Table 1 scenario uncertainty CSV: results/paper/table1_scenario_uncertainty.csv' \
		'Paper Table 1 paired deltas CSV: results/paper/table1_paired_scenario_deltas.csv' \
		'Paper Table 2 coverage-aware ablation: docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md' \
		'Paper Table 2 CSV: results/paper/table2_coverage_aware_ablation.csv' \
		'Paper Table 3 failure diagnosis: docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md' \
		'Paper Table 3 CSV: results/paper/table3_failure_diagnosis.csv' \
		'Paper Figure 2/3 points: results/paper/figure2_figure3_points.csv' \
		'Paper Figure 2 preview-gap: results/paper/figure2_preview_gap_frontier.png' \
		'Paper Figure 3 failed-missed: results/paper/figure3_failed_missed_tradeoff.png' \
		'Paper Figure 4 points: results/paper/figure4_invitation_mask_noise_points.csv' \
		'Paper Figure 4 gap-noise: results/paper/figure4_invitation_mask_gap_noise.png' \
		'Paper Figure 4 failed-missed-noise: results/paper/figure4_invitation_mask_failed_missed_noise.png' \
		'Project map: docs/PROJECT_MAP.md' \
		'Baseline strategy: docs/BASELINE_STRATEGY.md' \
		'Results index: docs/RESULTS_INDEX.md' \
		'Execution baseline summary: docs/EXECUTION_BASELINE_SUMMARY.md' \
		'Main results analysis: docs/MAIN_RESULTS_ANALYSIS.md' \
		'Coverage-aware analysis: docs/COVERAGE_AWARE_ANALYSIS.md' \
		'Final invitation mask analysis: docs/FINAL_INVITATION_MASK_ANALYSIS.md' \
		'Coverage B3 failure diagnosis: docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md' \
		'Invitation mask correction: docs/INVITATION_MASK_CORRECTION.md' \
		'Invitation mask correction noise: docs/INVITATION_MASK_CORRECTION_NOISE.md' \
		'Invitation mask correction noise-aware: docs/INVITATION_MASK_CORRECTION_NOISE_AWARE.md' \
		'Invitation mask rerank ablation: docs/INVITATION_MASK_RERANK_ABLATION.md' \
		'Deprecated directions: docs/DEPRECATED_DIRECTIONS.md' \
		'Experiment report: EXPERIMENT_REPORT.md'

help:
	@printf '%s\n' \
		'Core targets:' \
		'  make test                  Compile scripts and run smoke checks' \
		'  make pytest-test           Run standard pytest wrappers for project checks' \
		'  make lint                  Run high-signal Ruff checks' \
		'  make boundary-test         Check evaluator dependency boundaries' \
		'  make regression-test       Run fixed-seed mainline numerical checks' \
		'  make mainline-audit        Check existing mainline result artifacts and CSV links' \
		'  make check                 Run compile, lint, and pytest project checks' \
		'  make quick-audit           Run clean-room compile/lint/tests, artifact audit, and /tmp dry-run' \
		'  make quick-audit-dry-run   Run one tiny execution-mismatch dry-run writing only /tmp' \
		'  make smoke                 Run broader one-shot smoke experiments' \
		'  make policy-comparison     Run the main baseline table' \
		'  make policy-comparison-learning  Add SAC/Codebook-Aware SAC baselines' \
		'  make policy-comparison-static    Add Fixed/Best Fixed IRS ablations' \
		'  make runtime               Reproduce the runtime benchmark' \
		'  make docs                  Show documentation entry points' \
		'  make paper-tables          Generate paper-facing Table 1/2/3 artifacts' \
		'' \
		'Current execution-mismatch frontier:' \
		'  make sparse-topk-cost-pilot' \
		'  make sparse-topk-frontier' \
		'  make coverage-sparse-topk-pilot' \
		'  make coverage-sparse-topk-frontier' \
		'  make coverage-sparse-topk-ablation' \
		'  make coverage-sparse-power-pilot' \
		'  make coverage-sparse-power-ablation' \
		'  make coverage-budget-split-pilot' \
		'  make coverage-budget-split-selected' \
		'  make coverage-budget-split-formal' \
		'  make coverage-b3-failure-diagnosis' \
		'  make invitation-mask-correction-pilot' \
		'  make invitation-mask-correction-formal' \
		'  make invitation-mask-correction-noise-sweep' \
		'  make invitation-mask-correction-noise-aware-pilot' \
		'  make invitation-mask-correction-noise-aware-formal' \
		'  make invitation-mask-rerank-ablation' \
		'  make adaptive-sparse-topk-v2-pilot' \
		'  make execution-baseline-summary' \
		'  make main-results-analysis' \
		'  make coverage-aware-analysis' \
		'  make final-invitation-mask-analysis' \
		'' \
		'Supporting baselines:' \
		'  make partial-probing-sweep' \
		'  make limited-csi-sweep' \
		'  make execution-mismatch-sweep' \
		'  make active-probe-set-pilot' \
		'  make adaptive-sparse-topk-pilot' \
		'' \
		'Diagnostic/archive targets:' \
		'  make bandit-feedback-stress' \
		'  make adaptive-sparse-topk-v3-pilot' \
		'  make learned-sparse-shortlist-pilot' \
		'  make learned-sparse-shortlist-marginal-pilot' \
		'  make learned-set-shortlist-pilot' \
		'  make learned-execution-value-shortlist-pilot' \
		'  make learned-pairwise-shortlist-pilot'
