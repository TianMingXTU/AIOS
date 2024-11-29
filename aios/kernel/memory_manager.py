"""
Memory Manager
智能内存管理器，负责内存的智能分配、回收和优化
"""
import logging
import threading
import psutil
import numpy as np
from typing import Dict, List, Any, Optional
import torch
from collections import defaultdict

class MemoryBlock:
    """内存块"""
    def __init__(self, size: int, process_id: str, priority: float):
        self.size = size
        self.process_id = process_id
        self.priority = priority
        self.allocated_at = torch.cuda.Event().record()
        self.last_accessed = torch.cuda.Event().record()
        self.access_count = 0

class MemoryManager:
    """
    智能内存管理器负责：
    1. 智能内存分配
    2. 内存使用预测
    3. 内存回收和优化
    4. 内存访问模式学习
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = False
        self._lock = threading.Lock()
        
        # 内存块管理
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        
        # 内存使用统计
        self.usage_stats = defaultdict(list)
        
        # 访问模式学习
        self.access_patterns = defaultdict(lambda: np.zeros((24, 60)))  # 24小时 x 60分钟
        
        # GPU内存管理（如果可用）
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_memory = defaultdict(dict)

    def start(self):
        """启动内存管理器"""
        with self._lock:
            if self.running:
                return
                
            try:
                self.logger.info("正在启动内存管理器...")
                self._init_memory_monitoring()
                self.running = True
                self.logger.info("内存管理器启动成功")
                
            except Exception as e:
                self.logger.error(f"内存管理器启动失败: {str(e)}")
                raise

    def stop(self):
        """停止内存管理器"""
        with self._lock:
            if not self.running:
                return
                
            try:
                self.logger.info("正在停止内存管理器...")
                self.running = False
                self._cleanup()
                self.logger.info("内存管理器已停止")
                
            except Exception as e:
                self.logger.error(f"内存管理器停止失败: {str(e)}")
                raise

    def allocate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        智能内存分配
        :param request: 包含进程ID、所需内存大小和优先级的请求
        :return: 分配结果
        """
        if not self.running:
            raise RuntimeError("内存管理器未运行")
            
        try:
            process_id = request.get("process_id")
            size = request.get("size", 0)
            priority = request.get("priority", 0.0)
            
            # 检查可用内存
            available_memory = psutil.virtual_memory().available
            if size > available_memory:
                # 尝试内存回收
                self._collect_garbage()
                available_memory = psutil.virtual_memory().available
                if size > available_memory:
                    raise MemoryError("内存不足")
            
            # 创建内存块
            block = MemoryBlock(size, process_id, priority)
            block_id = f"block_{len(self.memory_blocks)}"
            
            with self._lock:
                self.memory_blocks[block_id] = block
                
            # 更新使用统计
            self._update_usage_stats(process_id, size)
            
            return {
                "block_id": block_id,
                "size": size,
                "status": "allocated"
            }
            
        except Exception as e:
            self.logger.error(f"内存分配失败: {str(e)}")
            raise

    def deallocate(self, block_id: str):
        """
        释放内存块
        :param block_id: 内存块ID
        """
        with self._lock:
            if block_id in self.memory_blocks:
                block = self.memory_blocks[block_id]
                # 更新使用统计
                self._update_usage_stats(block.process_id, -block.size)
                del self.memory_blocks[block_id]

    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存使用信息"""
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        info = {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "free": vm.free,
            "percent": vm.percent,
            "swap": {
                "total": swap.total,
                "used": swap.used,
                "free": swap.free,
                "percent": swap.percent
            }
        }
        
        if self.gpu_available:
            info["gpu"] = {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved()
            }
            
        return info

    def predict_memory_needs(self, process_id: str) -> Dict[str, float]:
        """
        预测进程的内存需求
        :param process_id: 进程ID
        :return: 预测的内存需求
        """
        if process_id in self.usage_stats:
            usage_history = self.usage_stats[process_id]
            if len(usage_history) > 0:
                # 使用简单的移动平均预测
                recent_usage = usage_history[-10:]  # 最近10次使用记录
                predicted_usage = sum(recent_usage) / len(recent_usage)
                
                # 考虑访问模式
                current_pattern = self.access_patterns[process_id]
                pattern_factor = np.mean(current_pattern)
                
                return {
                    "predicted_size": predicted_usage * (1 + pattern_factor),
                    "confidence": 0.8 if len(recent_usage) >= 10 else 0.5
                }
        
        return {
            "predicted_size": 0,
            "confidence": 0.0
        }

    def _init_memory_monitoring(self):
        """初始化内存监控"""
        # 这里可以添加更多的监控初始化逻辑
        pass

    def _cleanup(self):
        """清理资源"""
        with self._lock:
            self.memory_blocks.clear()
            if self.gpu_available:
                torch.cuda.empty_cache()

    def _collect_garbage(self):
        """垃圾回收"""
        with self._lock:
            # 按优先级排序内存块
            blocks = sorted(
                self.memory_blocks.items(),
                key=lambda x: x[1].priority
            )
            
            # 从低优先级开始回收
            for block_id, block in blocks:
                if psutil.virtual_memory().available >= block.size:
                    break
                self.deallocate(block_id)

    def _update_usage_stats(self, process_id: str, size_delta: int):
        """更新使用统计"""
        self.usage_stats[process_id].append(size_delta)
        # 限制历史记录大小
        if len(self.usage_stats[process_id]) > 1000:
            self.usage_stats[process_id] = self.usage_stats[process_id][-1000:]
