"""
Resource Manager
智能资源管理器，负责系统资源的分配和优化
"""
import logging
import threading
import psutil
import ray
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ResourceAllocation:
    """资源分配记录"""
    resource_type: str
    amount: float
    process_id: str
    priority: float
    allocated_at: datetime
    expires_at: Optional[datetime] = None

class ResourceManager:
    """
    智能资源管理器负责：
    1. 系统资源监控
    2. 智能资源分配
    3. 负载均衡
    4. 资源使用优化
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = False
        self._lock = threading.Lock()
        
        # 资源分配记录
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # 资源使用历史
        self.usage_history = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": []
        }
        
        # 负载预测模型
        self.load_predictions = {}
        
        # 资源限制
        self.resource_limits = {
            "cpu": psutil.cpu_count(),
            "memory": psutil.virtual_memory().total,
            "disk": psutil.disk_usage('/').total
        }

    def start(self):
        """启动资源管理器"""
        with self._lock:
            if self.running:
                return
                
            try:
                self.logger.info("正在启动资源管理器...")
                self._init_monitoring()
                self.running = True
                self.logger.info("资源管理器启动成功")
                
            except Exception as e:
                self.logger.error(f"资源管理器启动失败: {str(e)}")
                raise

    def stop(self):
        """停止资源管理器"""
        with self._lock:
            if not self.running:
                return
                
            try:
                self.logger.info("正在停止资源管理器...")
                self.running = False
                self._cleanup()
                self.logger.info("资源管理器已停止")
                
            except Exception as e:
                self.logger.error(f"资源管理器停止失败: {str(e)}")
                raise

    def assess_resources(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估资源请求
        :param request: 资源请求信息
        :return: 评估结果
        """
        if not self.running:
            raise RuntimeError("资源管理器未运行")
            
        try:
            process_id = request.get("process_id")
            resource_type = request.get("type", "cpu")
            amount = request.get("amount", 0.0)
            priority = request.get("priority", 0.0)
            
            # 检查资源可用性
            available = self._check_resource_availability(resource_type, amount)
            if not available:
                # 尝试资源回收
                self._reclaim_resources(resource_type, amount)
                available = self._check_resource_availability(resource_type, amount)
                if not available:
                    return {
                        "status": "rejected",
                        "reason": "资源不足"
                    }
            
            # 预测资源使用
            prediction = self._predict_resource_usage(process_id, resource_type)
            
            # 评估分配建议
            recommendation = self._get_allocation_recommendation(
                resource_type, amount, priority, prediction
            )
            
            return {
                "status": "approved" if recommendation["allocate"] else "rejected",
                "recommendation": recommendation
            }
            
        except Exception as e:
            self.logger.error(f"资源评估失败: {str(e)}")
            raise

    def allocate_resources(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        分配资源
        :param request: 资源请求
        :return: 分配结果
        """
        assessment = self.assess_resources(request)
        if assessment["status"] == "rejected":
            return assessment
            
        try:
            allocation = ResourceAllocation(
                resource_type=request.get("type", "cpu"),
                amount=request.get("amount", 0.0),
                process_id=request.get("process_id"),
                priority=request.get("priority", 0.0),
                allocated_at=datetime.now()
            )
            
            allocation_id = f"alloc_{len(self.allocations)}"
            
            with self._lock:
                self.allocations[allocation_id] = allocation
                
            return {
                "status": "success",
                "allocation_id": allocation_id,
                "details": assessment["recommendation"]
            }
            
        except Exception as e:
            self.logger.error(f"资源分配失败: {str(e)}")
            raise

    def release_resources(self, allocation_id: str):
        """
        释放资源
        :param allocation_id: 分配ID
        """
        with self._lock:
            if allocation_id in self.allocations:
                del self.allocations[allocation_id]

    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
            "network": psutil.net_io_counters()._asdict()
        }

    def _init_monitoring(self):
        """初始化资源监控"""
        @ray.remote
        def monitor_resources():
            while self.running:
                try:
                    usage = self.get_resource_usage()
                    with self._lock:
                        for resource_type in self.usage_history:
                            if resource_type in usage:
                                self.usage_history[resource_type].append(
                                    usage[resource_type]
                                )
                                # 限制历史记录大小
                                if len(self.usage_history[resource_type]) > 1000:
                                    self.usage_history[resource_type] = \
                                        self.usage_history[resource_type][-1000:]
                except Exception as e:
                    self.logger.error(f"资源监控错误: {str(e)}")
                    
                ray.sleep(1)  # 每秒更新一次
                
        # 启动监控
        ray.get(monitor_resources.remote())

    def _cleanup(self):
        """清理资源"""
        with self._lock:
            self.allocations.clear()

    def _check_resource_availability(self, resource_type: str, amount: float) -> bool:
        """检查资源可用性"""
        current_usage = self.get_resource_usage()
        if resource_type == "cpu":
            return psutil.cpu_percent() + amount <= 100
        elif resource_type == "memory":
            return psutil.virtual_memory().available >= amount
        elif resource_type == "disk":
            return psutil.disk_usage('/').free >= amount
        return False

    def _reclaim_resources(self, resource_type: str, amount: float):
        """回收资源"""
        with self._lock:
            # 按优先级排序分配
            allocations = sorted(
                self.allocations.items(),
                key=lambda x: x[1].priority
            )
            
            # 从低优先级开始回收
            reclaimed = 0
            for alloc_id, alloc in allocations:
                if alloc.resource_type == resource_type:
                    reclaimed += alloc.amount
                    self.release_resources(alloc_id)
                    if reclaimed >= amount:
                        break

    def _predict_resource_usage(self, process_id: str, resource_type: str) -> Dict[str, float]:
        """预测资源使用"""
        if resource_type in self.usage_history and len(self.usage_history[resource_type]) > 0:
            recent_usage = self.usage_history[resource_type][-10:]  # 最近10次记录
            predicted_usage = np.mean([float(usage.get("percent", 0)) for usage in recent_usage])
            return {
                "predicted_usage": predicted_usage,
                "confidence": 0.8 if len(recent_usage) >= 10 else 0.5
            }
        return {
            "predicted_usage": 0.0,
            "confidence": 0.0
        }

    def _get_allocation_recommendation(self, resource_type: str, amount: float,
                                     priority: float, prediction: Dict[str, float]) -> Dict[str, Any]:
        """获取分配建议"""
        current_usage = self.get_resource_usage()
        predicted_total = prediction["predicted_usage"] + amount
        
        if predicted_total > 90:  # 如果预测总使用率超过90%
            if priority > 0.8:  # 高优先级任务
                return {
                    "allocate": True,
                    "reason": "高优先级任务，尽管资源紧张",
                    "warning": "系统负载可能较高"
                }
            else:
                return {
                    "allocate": False,
                    "reason": "资源紧张，建议稍后重试",
                    "suggested_wait_time": 300  # 建议等待5分钟
                }
        
        return {
            "allocate": True,
            "reason": "资源充足"
        }
