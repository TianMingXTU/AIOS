"""
Process Context Manager
进程上下文管理器，负责管理进程的上下文信息和状态
"""
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import json
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class ProcessContext:
    """进程上下文信息"""
    process_id: str
    state: Dict[str, Any]  # 进程状态
    resources: Dict[str, float]  # 资源使用
    environment: Dict[str, str]  # 环境变量
    timestamps: Dict[str, datetime]  # 时间戳
    metrics: Dict[str, float]  # 性能指标
    dependencies: Dict[str, Any]  # 依赖关系

class ContextManager:
    """
    进程上下文管理器负责：
    1. 上下文信息管理
    2. 状态追踪
    3. 上下文切换优化
    4. 依赖关系管理
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # 上下文存储
        self.contexts: Dict[str, ProcessContext] = {}
        
        # 上下文切换历史
        self.switch_history = []
        
        # 性能指标
        self.metrics = {
            "switch_time": [],
            "state_changes": [],
            "resource_usage": []
        }

    def create_context(self, process_id: str, initial_state: Dict[str, Any]) -> ProcessContext:
        """
        创建进程上下文
        :param process_id: 进程ID
        :param initial_state: 初始状态
        :return: 创建的上下文
        """
        context = ProcessContext(
            process_id=process_id,
            state=initial_state,
            resources={
                "cpu": 0.0,
                "memory": 0.0,
                "io": 0.0
            },
            environment={},
            timestamps={
                "created": datetime.now(),
                "last_active": datetime.now()
            },
            metrics={
                "cpu_time": 0.0,
                "memory_usage": 0.0,
                "io_operations": 0
            },
            dependencies={}
        )
        
        with self._lock:
            self.contexts[process_id] = context
            
        return context

    def get_context(self, process_id: str) -> Optional[ProcessContext]:
        """
        获取进程上下文
        :param process_id: 进程ID
        :return: 进程上下文
        """
        with self._lock:
            return self.contexts.get(process_id)

    def update_context(self, process_id: str, updates: Dict[str, Any]):
        """
        更新进程上下文
        :param process_id: 进程ID
        :param updates: 更新内容
        """
        with self._lock:
            if process_id in self.contexts:
                context = self.contexts[process_id]
                
                # 更新状态
                if "state" in updates:
                    context.state.update(updates["state"])
                
                # 更新资源使用
                if "resources" in updates:
                    context.resources.update(updates["resources"])
                
                # 更新环境变量
                if "environment" in updates:
                    context.environment.update(updates["environment"])
                
                # 更新时间戳
                context.timestamps["last_active"] = datetime.now()
                
                # 更新性能指标
                if "metrics" in updates:
                    context.metrics.update(updates["metrics"])
                
                # 更新依赖关系
                if "dependencies" in updates:
                    context.dependencies.update(updates["dependencies"])

    def switch_context(self, from_pid: str, to_pid: str) -> float:
        """
        进行上下文切换
        :param from_pid: 源进程ID
        :param to_pid: 目标进程ID
        :return: 切换耗时（毫秒）
        """
        start_time = datetime.now()
        
        with self._lock:
            # 保存源进程上下文
            if from_pid in self.contexts:
                self._save_context_state(from_pid)
            
            # 加载目标进程上下文
            if to_pid in self.contexts:
                self._load_context_state(to_pid)
            
            # 记录切换时间
            switch_time = (datetime.now() - start_time).total_seconds() * 1000
            self.switch_history.append({
                "from": from_pid,
                "to": to_pid,
                "time": switch_time,
                "timestamp": datetime.now()
            })
            
            # 更新指标
            self.metrics["switch_time"].append(switch_time)
            
            # 限制历史记录大小
            if len(self.switch_history) > 1000:
                self.switch_history = self.switch_history[-1000:]
            
            return switch_time

    def _save_context_state(self, process_id: str):
        """
        保存进程上下文状态
        :param process_id: 进程ID
        """
        context = self.contexts[process_id]
        # 这里可以添加更多的状态保存逻辑
        context.timestamps["last_saved"] = datetime.now()

    def _load_context_state(self, process_id: str):
        """
        加载进程上下文状态
        :param process_id: 进程ID
        """
        context = self.contexts[process_id]
        # 这里可以添加更多的状态加载逻辑
        context.timestamps["last_loaded"] = datetime.now()

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self._lock:
            return {
                "average_switch_time": np.mean(self.metrics["switch_time"])
                if self.metrics["switch_time"] else 0,
                "total_switches": len(self.switch_history),
                "recent_switches": self.switch_history[-10:]
            }

    def cleanup_context(self, process_id: str):
        """
        清理进程上下文
        :param process_id: 进程ID
        """
        with self._lock:
            if process_id in self.contexts:
                # 保存最终状态
                self._save_context_state(process_id)
                # 删除上下文
                del self.contexts[process_id]

    def serialize_context(self, process_id: str) -> str:
        """
        序列化进程上下文
        :param process_id: 进程ID
        :return: 序列化的上下文数据
        """
        with self._lock:
            if process_id in self.contexts:
                context = self.contexts[process_id]
                # 转换datetime对象为ISO格式字符串
                context_dict = asdict(context)
                for key, value in context_dict["timestamps"].items():
                    if isinstance(value, datetime):
                        context_dict["timestamps"][key] = value.isoformat()
                return json.dumps(context_dict)
        return "{}"

    def deserialize_context(self, process_id: str, context_data: str):
        """
        反序列化进程上下文
        :param process_id: 进程ID
        :param context_data: 序列化的上下文数据
        """
        try:
            context_dict = json.loads(context_data)
            # 转换ISO格式字符串为datetime对象
            for key, value in context_dict["timestamps"].items():
                if isinstance(value, str):
                    context_dict["timestamps"][key] = datetime.fromisoformat(value)
            
            context = ProcessContext(**context_dict)
            
            with self._lock:
                self.contexts[process_id] = context
                
        except Exception as e:
            self.logger.error(f"上下文反序列化失败: {str(e)}")
            raise
