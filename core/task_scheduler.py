"""
Task Scheduler Module
负责管理和调度系统任务
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import threading
import queue

@dataclass
class Task:
    id: str
    name: str
    priority: int
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[any] = None

class TaskScheduler:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_queue = queue.PriorityQueue()
        self.lock = threading.Lock()
        self._running = False
        self.worker_thread = None

    def create_task(self, name: str, priority: int = 1) -> Task:
        """创建新任务"""
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            priority=priority,
            status="pending",
            created_at=datetime.now()
        )
        with self.lock:
            self.tasks[task.id] = task
            self.task_queue.put((-priority, task.id))
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务信息"""
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[Task]:
        """列出所有任务"""
        return list(self.tasks.values())

    def start(self):
        """启动任务调度器"""
        if self._running:
            return
        self._running = True
        self.worker_thread = threading.Thread(target=self._process_tasks)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop(self):
        """停止任务调度器"""
        self._running = False
        if self.worker_thread:
            self.worker_thread.join()

    def _process_tasks(self):
        """处理任务队列"""
        while self._running:
            try:
                priority, task_id = self.task_queue.get(timeout=1)
                task = self.tasks[task_id]
                
                # 更新任务状态
                with self.lock:
                    task.status = "running"
                    task.started_at = datetime.now()

                # TODO: 实际的任务执行逻辑
                # 这里将来会与AI引擎集成

                # 更新任务完成状态
                with self.lock:
                    task.status = "completed"
                    task.completed_at = datetime.now()
                
            except queue.Empty:
                continue
            except Exception as e:
                if task_id in self.tasks:
                    self.tasks[task_id].status = "failed"
                    self.tasks[task_id].result = str(e)
