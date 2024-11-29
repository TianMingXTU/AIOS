"""
AIOS安全层测试套件
"""
import os
import pytest
import time
from datetime import datetime
from aios.security.access_control import AccessManager
from aios.security.isolation import ResourceIsolator
from aios.security.threat_detection import ThreatDetector, ThreatEvent

@pytest.fixture
def access_manager():
    """创建AccessManager实例"""
    return AccessManager()

@pytest.fixture
def resource_isolator():
    """创建ResourceIsolator实例"""
    return ResourceIsolator()

@pytest.fixture
def threat_detector():
    """创建ThreatDetector实例"""
    return ThreatDetector()

class TestAccessControl:
    """访问控制测试"""
    
    def test_user_authentication(self, access_manager):
        """测试用户认证"""
        # 测试管理员登录
        assert access_manager.authenticate("admin", os.getenv("AIOS_ADMIN_PASSWORD"))
        
        # 测试错误密码
        assert not access_manager.authenticate("admin", "wrong_password")
        
        # 测试不存在的用户
        assert not access_manager.authenticate("nonexistent", "password")
    
    def test_role_permissions(self, access_manager):
        """测试角色权限"""
        # 测试管理员权限
        admin_token = access_manager.create_token("admin")
        assert access_manager.check_permission(admin_token, "system_manage")
        assert access_manager.check_permission(admin_token, "file_write")
        
        # 测试普通用户权限
        user_token = access_manager.create_token("user")
        assert access_manager.check_permission(user_token, "file_read")
        assert not access_manager.check_permission(user_token, "system_manage")
        
        # 测试访客权限
        guest_token = access_manager.create_token("guest")
        assert access_manager.check_permission(guest_token, "file_read")
        assert not access_manager.check_permission(guest_token, "file_write")
    
    def test_session_management(self, access_manager):
        """测试会话管理"""
        # 创建会话
        token = access_manager.create_token("admin")
        assert access_manager.validate_token(token)
        
        # 测试过期token
        expired_token = access_manager.create_token("admin", expires_in=-1)
        assert not access_manager.validate_token(expired_token)
        
        # 测试会话撤销
        access_manager.revoke_token(token)
        assert not access_manager.validate_token(token)

class TestResourceIsolation:
    """资源隔离测试"""
    
    def test_process_isolation(self, resource_isolator):
        """测试进程隔离"""
        # 创建隔离进程
        process_id = resource_isolator.create_isolated_process("echo test")
        assert process_id > 0
        
        # 检查进程状态
        assert resource_isolator.get_process_status(process_id) == "running"
        
        # 终止进程
        assert resource_isolator.terminate_process(process_id)
        assert resource_isolator.get_process_status(process_id) == "terminated"
    
    def test_filesystem_isolation(self, resource_isolator):
        """测试文件系统隔离"""
        # 创建隔离目录
        mount_point = resource_isolator.create_isolated_fs()
        assert os.path.exists(mount_point)
        
        # 写入文件
        test_file = os.path.join(mount_point, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        # 检查隔离性
        assert not os.path.exists("test.txt")
        assert os.path.exists(test_file)
        
        # 清理
        resource_isolator.remove_isolated_fs(mount_point)
        assert not os.path.exists(mount_point)
    
    def test_network_isolation(self, resource_isolator):
        """测试网络隔离"""
        # 创建网络命名空间
        ns_name = resource_isolator.create_network_ns()
        assert resource_isolator.network_ns_exists(ns_name)
        
        # 配置网络
        assert resource_isolator.configure_network(ns_name)
        
        # 检查连接性
        assert resource_isolator.check_network_isolation(ns_name)
        
        # 清理
        resource_isolator.remove_network_ns(ns_name)
        assert not resource_isolator.network_ns_exists(ns_name)

class TestThreatDetection:
    """威胁检测测试"""
    
    def test_system_monitoring(self, threat_detector):
        """测试系统监控"""
        # 收集指标
        metrics = threat_detector._collect_metrics()
        assert "system" in metrics
        assert "processes" in metrics
        assert "network" in metrics
        
        # 检查系统指标
        assert "cpu_percent" in metrics["system"]
        assert "memory_percent" in metrics["system"]
        assert "disk_percent" in metrics["system"]
    
    def test_anomaly_detection(self, threat_detector):
        """测试异常检测"""
        # 训练模型
        threat_detector.train_anomaly_detectors()
        
        # 正常指标
        normal_metrics = {
            "system": {
                "cpu_percent": 20,
                "memory_percent": 50,
                "disk_percent": 60
            },
            "processes": [],
            "network": {
                "connections": 10
            }
        }
        assert threat_detector.check_anomaly(normal_metrics) < 0.5
        
        # 异常指标
        abnormal_metrics = {
            "system": {
                "cpu_percent": 95,
                "memory_percent": 95,
                "disk_percent": 95
            },
            "processes": [],
            "network": {
                "connections": 1000
            }
        }
        assert threat_detector.check_anomaly(abnormal_metrics) > 0.5
    
    def test_threat_response(self, threat_detector):
        """测试威胁响应"""
        # 创建威胁事件
        threat_detector._create_threat_event(
            "high_cpu_usage",
            "High CPU usage detected",
            {"cpu_percent": 95}
        )
        
        # 检查事件记录
        events = threat_detector.get_recent_events(hours=1)
        assert len(events) > 0
        assert events[0].rule_name == "high_cpu_usage"
        assert events[0].severity >= 3

def test_integration(access_manager, resource_isolator, threat_detector):
    """集成测试"""
    # 1. 创建用户并授权
    token = access_manager.create_token("admin")
    assert access_manager.validate_token(token)
    
    # 2. 创建隔离环境
    process_id = resource_isolator.create_isolated_process("stress --cpu 8")
    assert process_id > 0
    
    # 3. 等待威胁检测
    time.sleep(5)
    events = threat_detector.get_recent_events(hours=1, min_severity=3)
    assert len(events) > 0
    
    # 4. 响应威胁
    resource_isolator.terminate_process(process_id)
    assert resource_isolator.get_process_status(process_id) == "terminated"
