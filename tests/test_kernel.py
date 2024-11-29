"""
内核层测试
"""
import unittest
import asyncio
from aios.kernel import AIKernel, CognitiveEngine, MemoryManager, ResourceManager

class TestAIKernel(unittest.TestCase):
    """测试AI内核"""
    
    def setUp(self):
        self.kernel = AIKernel()
    
    def test_kernel_initialization(self):
        """测试内核初始化"""
        self.assertIsNotNone(self.kernel)
        self.assertTrue(hasattr(self.kernel, 'cognitive_engine'))
        self.assertTrue(hasattr(self.kernel, 'memory_manager'))
        self.assertTrue(hasattr(self.kernel, 'resource_manager'))
    
    def test_kernel_start_stop(self):
        """测试内核启动和停止"""
        self.kernel.start()
        self.assertTrue(self.kernel.is_running())
        self.kernel.stop()
        self.assertFalse(self.kernel.is_running())
    
    def test_task_scheduling(self):
        """测试任务调度"""
        self.kernel.start()
        task = {"name": "test_task", "priority": 1}
        task_id = self.kernel.schedule_task(task)
        self.assertIsNotNone(task_id)
        task_status = self.kernel.get_task_status(task_id)
        self.assertIn(task_status, ['pending', 'running', 'completed'])
        self.kernel.stop()

class TestCognitiveEngine(unittest.TestCase):
    """测试认知引擎"""
    
    def setUp(self):
        self.engine = CognitiveEngine()
    
    def test_intent_recognition(self):
        """测试意图识别"""
        intent = self.engine.recognize_intent("打开文件test.txt")
        self.assertEqual(intent['action'], 'open')
        self.assertEqual(intent['target'], 'file')
        self.assertEqual(intent['params']['filename'], 'test.txt')
    
    def test_context_understanding(self):
        """测试上下文理解"""
        context = {
            'current_dir': '/home/user',
            'last_command': 'ls',
            'system_load': 0.5
        }
        understanding = self.engine.understand_context(context)
        self.assertIsInstance(understanding, dict)
        self.assertIn('current_state', understanding)
        self.assertIn('predicted_needs', understanding)

class TestMemoryManager(unittest.TestCase):
    """测试内存管理器"""
    
    def setUp(self):
        self.manager = MemoryManager()
    
    def test_memory_allocation(self):
        """测试内存分配"""
        allocation = self.manager.allocate(1024)  # 1KB
        self.assertIsNotNone(allocation)
        self.assertTrue(self.manager.is_allocated(allocation))
        self.manager.free(allocation)
        self.assertFalse(self.manager.is_allocated(allocation))
    
    def test_memory_prediction(self):
        """测试内存使用预测"""
        prediction = self.manager.predict_usage(timeframe=60)  # 60秒
        self.assertIsInstance(prediction, dict)
        self.assertIn('predicted_usage', prediction)
        self.assertIn('confidence', prediction)

class TestResourceManager(unittest.TestCase):
    """测试资源管理器"""
    
    def setUp(self):
        self.manager = ResourceManager()
    
    def test_resource_monitoring(self):
        """测试资源监控"""
        status = self.manager.get_system_status()
        self.assertIn('cpu_usage', status)
        self.assertIn('memory_usage', status)
        self.assertIn('disk_usage', status)
    
    def test_resource_optimization(self):
        """测试资源优化"""
        suggestions = self.manager.optimize_resources()
        self.assertIsInstance(suggestions, list)
        for suggestion in suggestions:
            self.assertIn('type', suggestion)
            self.assertIn('action', suggestion)
            self.assertIn('expected_benefit', suggestion)

if __name__ == '__main__':
    unittest.main()
