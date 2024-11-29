"""
Smart Process Scheduler
智能进程调度器，使用AI技术进行进程调度决策
"""
import logging
import threading
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import ray
from queue import PriorityQueue

@dataclass
class Process:
    """进程信息"""
    id: str
    name: str
    priority: float
    state: str  # ready, running, blocked, terminated
    cpu_usage: float
    memory_usage: float
    created_at: datetime
    context: Dict[str, Any]
    dependencies: List[str]

class SmartScheduler:
    """
    智能进程调度器负责：
    1. 进程优先级动态调整
    2. 上下文感知调度
    3. 负载预测和平衡
    4. 资源使用优化
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = False
        self._lock = threading.Lock()
        
        # 进程管理
        self.processes: Dict[str, Process] = {}
        self.ready_queue = PriorityQueue()
        self.blocked_queue = []
        
        # 调度策略
        self.quantum = 100  # 时间片大小（毫秒）
        self.max_priority = 10.0
        
        # 性能指标
        self.metrics = {
            "throughput": [],
            "response_time": [],
            "waiting_time": []
        }
        
        # 负载历史
        self.load_history = []

    def start(self):
        """启动调度器"""
        with self._lock:
            if self.running:
                return
                
            try:
                self.logger.info("正在启动智能进程调度器...")
                self.running = True
                self._start_scheduler_loop()
                self.logger.info("智能进程调度器启动成功")
                
            except Exception as e:
                self.logger.error(f"调度器启动失败: {str(e)}")
                raise

    def stop(self):
        """停止调度器"""
        with self._lock:
            if not self.running:
                return
                
            try:
                self.logger.info("正在停止调度器...")
                self.running = False
                self.logger.info("调度器已停止")
                
            except Exception as e:
                self.logger.error(f"调度器停止失败: {str(e)}")
                raise

    def create_process(self, process_info: Dict[str, Any]) -> Process:
        """
        创建新进程
        :param process_info: 进程信息
        :return: 创建的进程
        """
        process = Process(
            id=f"proc_{len(self.processes)}",
            name=process_info.get("name", "unknown"),
            priority=process_info.get("priority", 1.0),
            state="ready",
            cpu_usage=0.0,
            memory_usage=0.0,
            created_at=datetime.now(),
            context=process_info.get("context", {}),
            dependencies=process_info.get("dependencies", [])
        )
        
        with self._lock:
            self.processes[process.id] = process
            # 优先级队列使用负优先级值，这样高优先级的进程会先被处理
            self.ready_queue.put((-process.priority, process.id))
            
        return process

    def terminate_process(self, process_id: str):
        """
        终止进程
        :param process_id: 进程ID
        """
        with self._lock:
            if process_id in self.processes:
                process = self.processes[process_id]
                process.state = "terminated"
                del self.processes[process_id]

    def block_process(self, process_id: str, reason: str):
        """
        阻塞进程
        :param process_id: 进程ID
        :param reason: 阻塞原因
        """
        with self._lock:
            if process_id in self.processes:
                process = self.processes[process_id]
                process.state = "blocked"
                process.context["blocked_reason"] = reason
                self.blocked_queue.append(process_id)

    def unblock_process(self, process_id: str):
        """
        解除进程阻塞
        :param process_id: 进程ID
        """
        with self._lock:
            if process_id in self.processes:
                process = self.processes[process_id]
                if process.state == "blocked":
                    process.state = "ready"
                    self.ready_queue.put((-process.priority, process_id))
                    self.blocked_queue.remove(process_id)

    def adjust_priority(self, process_id: str, delta: float):
        """
        调整进程优先级
        :param process_id: 进程ID
        :param delta: 优先级调整值
        """
        with self._lock:
            if process_id in self.processes:
                process = self.processes[process_id]
                new_priority = max(0.0, min(self.max_priority,
                                          process.priority + delta))
                process.priority = new_priority

    def get_process_info(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        获取进程信息
        :param process_id: 进程ID
        :return: 进程信息
        """
        with self._lock:
            if process_id in self.processes:
                process = self.processes[process_id]
                return {
                    "id": process.id,
                    "name": process.name,
                    "priority": process.priority,
                    "state": process.state,
                    "cpu_usage": process.cpu_usage,
                    "memory_usage": process.memory_usage,
                    "created_at": process.created_at,
                    "context": process.context,
                    "dependencies": process.dependencies
                }
        return None

    @ray.remote
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                self._schedule_next_process()
                self._update_metrics()
                self._adjust_scheduling_parameters()
                time.sleep(self.quantum / 1000)  # 转换为秒
                
            except Exception as e:
                self.logger.error(f"调度循环错误: {str(e)}")

    def _start_scheduler_loop(self):
        """启动调度循环"""
        ray.get(self._scheduler_loop.remote())

    def _schedule_next_process(self):
        """选择下一个要执行的进程"""
        with self._lock:
            if self.ready_queue.empty():
                return
                
            try:
                # 获取最高优先级的进程
                _, process_id = self.ready_queue.get_nowait()
                if process_id in self.processes:
                    process = self.processes[process_id]
                    
                    # 检查依赖
                    if self._check_dependencies(process):
                        process.state = "running"
                        self._execute_process(process)
                    else:
                        # 如果依赖未满足，将进程阻塞
                        self.block_process(process_id, "waiting_for_dependencies")
                        
            except Exception as e:
                self.logger.error(f"进程调度失败: {str(e)}")

    def _check_dependencies(self, process: Process) -> bool:
        """检查进程依赖是否满足"""
        for dep_id in process.dependencies:
            if dep_id in self.processes and \
               self.processes[dep_id].state != "terminated":
                return False
        return True

    def _execute_process(self, process: Process):
        """
        执行进程
        这里是模拟执行，实际实现需要与真实的进程执行机制集成
        """
        try:
            # 更新进程状态
            process.cpu_usage = np.random.uniform(0, 100)
            process.memory_usage = np.random.uniform(0, 100)
            
            # 模拟执行时间
            time.sleep(self.quantum / 1000)
            
            # 完成后重新加入就绪队列
            process.state = "ready"
            self.ready_queue.put((-process.priority, process.id))
            
        except Exception as e:
            self.logger.error(f"进程执行错误: {str(e)}")
            process.state = "terminated"

    def _update_metrics(self):
        """更新性能指标"""
        with self._lock:
            # 计算吞吐量
            completed_processes = len([p for p in self.processes.values()
                                    if p.state == "terminated"])
            self.metrics["throughput"].append(completed_processes)
            
            # 计算响应时间
            current_time = datetime.now()
            response_times = [(current_time - p.created_at).total_seconds()
                            for p in self.processes.values()]
            if response_times:
                self.metrics["response_time"].append(np.mean(response_times))
            
            # 限制历史记录大小
            for metric in self.metrics:
                if len(self.metrics[metric]) > 1000:
                    self.metrics[metric] = self.metrics[metric][-1000:]

    def _adjust_scheduling_parameters(self):
        """动态调整调度参数"""
        # 基于性能指标调整时间片大小
        if len(self.metrics["response_time"]) > 0:
            avg_response_time = np.mean(self.metrics["response_time"][-10:])
            if avg_response_time > 1.0:  # 如果平均响应时间超过1秒
                self.quantum = max(50, self.quantum - 10)  # 减小时间片
            else:
                self.quantum = min(200, self.quantum + 10)  # 增加时间片
