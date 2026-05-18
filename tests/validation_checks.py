"""CLI validation regression checks for experiment entry points."""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_expect_failure(command, expected_snippet):
    """Run a command that should fail during argument validation."""
    result = subprocess.run(
        [sys.executable, *command],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(command)}")
    combined = result.stdout + result.stderr
    if expected_snippet not in combined:
        raise AssertionError(
            f"command {' '.join(command)} did not report {expected_snippet!r}; "
            f"output was:\n{combined}"
        )


def main():
    """Run validation checks."""
    run_expect_failure(
        ["evaluate_invitation_mask_correction.py", "--episodes", "0"],
        "--episodes must be positive",
    )
    run_expect_failure(
        ["evaluate_invitation_mask_correction.py", "--channel-rho-values", "nan"],
        "--channel-rho-values must contain finite values",
    )
    run_expect_failure(
        ["evaluate_execution_channel_mismatch.py", "--decision-error-std-values", "nan"],
        "--decision-error-std-values must contain finite values",
    )
    run_expect_failure(
        ["evaluate_execution_channel_mismatch.py", "--adaptive-risk-error-ref", "nan"],
        "--adaptive-risk-error-ref must contain finite values",
    )
    run_expect_failure(
        ["evaluate_execution_channel_mismatch.py", "--num-codebook-states", "1"],
        "--num-codebook-states must be greater than 1",
    )


if __name__ == "__main__":
    main()
