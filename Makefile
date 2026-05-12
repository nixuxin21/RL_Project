PYTHON ?= ./.venv/bin/python

.PHONY: smoke test py-compile policy-comparison runtime parameter-sweep noisy-feature-sweep partial-probing-sweep learned-probing probing-cost-tradeoff channel-estimation-sweep limited-csi-sweep bandit-feedback-sweep bandit-feedback-stress action-diagnostics

py-compile:
	$(PYTHON) -m py_compile \
		test_env.py \
		train_agent.py \
		train_codebook_aware_agent.py \
		train_greedy_imitation_selector.py \
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
		evaluate_bandit_feedback_ms_aircomp.py \
		evaluate_bandit_feedback_stress_sweep.py \
		diagnose_policy_actions.py \
		benchmark_policy_runtime.py \
		tests/smoke_checks.py

test: py-compile
	$(PYTHON) tests/smoke_checks.py

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

policy-comparison:
	$(PYTHON) evaluate_policy_comparison.py \
		--episodes 1000 \
		--num-seeds 5 \
		--include-codebook-aware-sac

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

probing-cost-tradeoff:
	$(PYTHON) experiments/archive/evaluate_probing_cost_tradeoff.py

channel-estimation-sweep:
	$(PYTHON) evaluate_channel_estimation_error_sweep.py

limited-csi-sweep:
	$(PYTHON) evaluate_limited_csi_ms_aircomp.py

bandit-feedback-sweep:
	$(PYTHON) evaluate_bandit_feedback_ms_aircomp.py

bandit-feedback-stress:
	$(PYTHON) evaluate_bandit_feedback_stress_sweep.py

action-diagnostics:
	$(PYTHON) diagnose_policy_actions.py --episodes 1000
