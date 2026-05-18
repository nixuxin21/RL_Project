"""测试入口包装层，把现有脚本式检查统一接入标准 pytest 测试入口。"""

from tests import (
    dependency_boundary_checks,
    mainline_artifact_checks,
    mainline_regression_checks,
    smoke_checks,
    validation_checks,
)


def assert_success(status):
    """处理assert、成功相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
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
