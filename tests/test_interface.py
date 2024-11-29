"""
AIOS接口层测试套件
"""
import pytest
import json
from fastapi.testclient import TestClient
from aios.interface.cli import CLI
from aios.interface.api import app
from aios.interface.nlp_interface import NLPInterface

@pytest.fixture
def cli():
    """创建CLI实例"""
    return CLI()

@pytest.fixture
def api_client():
    """创建API测试客户端"""
    return TestClient(app)

@pytest.fixture
def nlp_interface():
    """创建NLP接口实例"""
    return NLPInterface()

class TestCLI:
    """命令行界面测试"""
    
    def test_basic_commands(self, cli):
        """测试基本命令"""
        # 测试help命令
        result = cli.execute("help")
        assert result["status"] == "success"
        assert "Available commands" in result["output"]
        
        # 测试status命令
        result = cli.execute("status")
        assert result["status"] == "success"
        assert "System Status" in result["output"]
        
        # 测试ls命令
        result = cli.execute("ls .")
        assert result["status"] == "success"
        assert isinstance(result["output"], list)
    
    def test_command_completion(self, cli):
        """测试命令补全"""
        # 测试部分命令补全
        completions = cli.complete("sta")
        assert "status" in completions
        
        # 测试参数补全
        completions = cli.complete("ls ")
        assert len(completions) > 0
        assert all(isinstance(c, str) for c in completions)
    
    def test_error_handling(self, cli):
        """测试错误处理"""
        # 测试不存在的命令
        result = cli.execute("nonexistent")
        assert result["status"] == "error"
        assert "Unknown command" in result["error"]
        
        # 测试参数错误
        result = cli.execute("ls --invalid")
        assert result["status"] == "error"
        assert "Invalid argument" in result["error"]

class TestAPI:
    """API接口测试"""
    
    def test_authentication(self, api_client):
        """测试认证"""
        # 测试无token访问
        response = api_client.get("/status")
        assert response.status_code == 401
        
        # 测试token获取
        response = api_client.post("/token", data={
            "username": "admin",
            "password": "admin"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        
        # 测试使用token访问
        response = api_client.get(
            "/status",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
    
    def test_endpoints(self, api_client):
        """测试端点"""
        # 获取token
        response = api_client.post("/token", data={
            "username": "admin",
            "password": "admin"
        })
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 测试状态端点
        response = api_client.get("/status", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "cpu_usage" in data
        assert "memory_usage" in data
        assert "disk_usage" in data
        assert "process_count" in data
        
        # 测试进程列表端点
        response = api_client.get("/processes", headers=headers)
        assert response.status_code == 200
        processes = response.json()
        assert isinstance(processes, list)
        if processes:  # 如果有进程
            process = processes[0]
            assert "pid" in process
            assert "name" in process
            assert "status" in process
            assert "cpu_percent" in process
            assert "memory_percent" in process
        
        # 测试文件系统端点
        response = api_client.get("/fs/list/.", headers=headers)
        assert response.status_code == 200
        files = response.json()
        assert isinstance(files, list)
        if files:  # 如果有文件
            file = files[0]
            assert "name" in file
            assert "type" in file
            assert "size" in file
            assert "modified" in file
        
        # 测试命令执行端点
        command_request = {
            "command": "echo",
            "args": ["test"]
        }
        response = api_client.post("/exec", headers=headers, json=command_request)
        assert response.status_code == 200
        result = response.json()
        assert "success" in result
        assert "output" in result
    
    def test_error_handling(self, api_client):
        """测试错误处理"""
        # 测试无效token
        response = api_client.get(
            "/status",
            headers={"Authorization": "Bearer invalid"}
        )
        assert response.status_code == 401
        
        # 测试无效端点
        response = api_client.get("/nonexistent")
        assert response.status_code == 404
        
        # 测试无效进程ID
        response = api_client.post("/token", data={
            "username": "admin",
            "password": "admin"
        })
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        response = api_client.delete("/processes/999999", headers=headers)
        assert response.status_code == 404

class TestNLPInterface:
    """NLP接口测试"""
    
    def test_intent_recognition(self, nlp_interface):
        """测试意图识别"""
        # 测试系统状态意图
        intent = nlp_interface.recognize_intent("显示系统状态")
        assert intent["intent"] == "get_status"
        assert intent["confidence"] > 0.8
        
        # 测试文件操作意图
        intent = nlp_interface.recognize_intent("列出当前目录下的文件")
        assert intent["intent"] == "list_files"
        assert intent["confidence"] > 0.8
    
    def test_entity_extraction(self, nlp_interface):
        """测试实体提取"""
        # 测试文件路径提取
        entities = nlp_interface.extract_entities("打开/home/user/test.txt文件")
        assert "file_path" in entities
        assert entities["file_path"] == "/home/user/test.txt"
        
        # 测试进程ID提取
        entities = nlp_interface.extract_entities("终止进程1234")
        assert "process_id" in entities
        assert entities["process_id"] == "1234"
    
    def test_context_management(self, nlp_interface):
        """测试上下文管理"""
        # 设置上下文
        nlp_interface.set_context({"current_dir": "/home/user"})
        
        # 测试上下文相关的命令
        response = nlp_interface.process("列出文件")
        assert response["status"] == "success"
        assert "files" in response
        
        # 测试上下文继承
        response = nlp_interface.process("打开test.txt")
        assert response["status"] == "success"
        assert response["file_path"] == "/home/user/test.txt"

def test_integration(cli, api_client, nlp_interface):
    """集成测试"""
    # 1. CLI命令执行
    cli_result = cli.execute("status")
    assert cli_result["status"] == "success"
    
    # 2. API状态获取
    response = api_client.post("/token", data={
        "username": "admin",
        "password": "admin"
    })
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    api_result = api_client.get("/status", headers=headers)
    assert api_result.status_code == 200
    
    # 3. NLP命令处理
    nlp_result = nlp_interface.process("显示系统状态")
    assert nlp_result["status"] == "success"
    
    # 4. 结果比较
    cli_status = json.loads(cli_result["output"])
    api_status = api_result.json()
    nlp_status = nlp_result["system_status"]
    
    # 验证三个接口返回的系统状态是否一致
    assert cli_status["cpu_percent"] == api_status["cpu_percent"]
    assert api_status["cpu_percent"] == nlp_status["cpu_percent"]
