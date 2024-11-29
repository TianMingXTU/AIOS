"""
AIOS测试配置
"""
import os
import pytest
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_dir():
    """创建测试目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def env_setup():
    """设置环境变量"""
    os.environ["AIOS_JWT_SECRET"] = "test-secret-key"
    os.environ["AIOS_ADMIN_PASSWORD"] = "test-admin-password"
    yield
    del os.environ["AIOS_JWT_SECRET"]
    del os.environ["AIOS_ADMIN_PASSWORD"]
