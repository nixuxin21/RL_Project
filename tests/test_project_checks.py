"""Pytest entry points for the existing script-style project checks."""

from tests import (
    dependency_boundary_checks,
    mainline_artifact_checks,
    mainline_regression_checks,
    smoke_checks,
    validation_checks,
)


def assert_success(status):
    """Treat script-style ``None`` and shell-style ``0`` as success."""
    assert status in (None, 0)


def test_smoke_checks():
    assert_success(smoke_checks.main())


def test_dependency_boundary_checks():
    assert_success(dependency_boundary_checks.main())


def test_mainline_regression_checks():
    assert_success(mainline_regression_checks.main())


def test_mainline_artifact_checks():
    assert_success(mainline_artifact_checks.main())


def test_validation_checks():
    assert_success(validation_checks.main())
