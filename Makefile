PYTHON ?= ./.venv/bin/python

.PHONY: smoke test py-compile policy-comparison runtime parameter-sweep action-diagnostics

py-compile:
	$(PYTHON) -m py_compile \
		test_env.py \
		train_agent.py \
		train_codebook_aware_agent.py \
		train_greedy_imitation_selector.py \
		evaluate_agent.py \
		evaluate_batch.py \
		evaluate_random_irs_baseline.py \
		evaluate_policy_comparison.py \
		evaluate_parameter_sweep.py \
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

action-diagnostics:
	$(PYTHON) diagnose_policy_actions.py --episodes 1000
