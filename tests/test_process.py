"""
进程管理层测试
"""
import unittest
import asyncio
from aios.process import SmartScheduler, ContextManager, LoadPredictor

class TestSmartScheduler(unittest.TestCase):
    """测试智能调度器"""
    
    def setUp(self):
        self.scheduler = SmartScheduler()
    
    def test_process_creation(self):
        """测试进程创建"""
        process = self.scheduler.create_process({
            "name": "test_process",
            "priority": 1.0,
            "resources": {"cpu": 0.5, "memory": 512}
        })
        self.assertIsNotNone(process)
        self.assertEqual(process.name, "test_process")
        self.assertEqual(process.priority, 1.0)
    
    def test_priority_adjustment(self):
        """测试优先级调整"""
        process = self.scheduler.create_process({
            "name": "test_process",
            "priority": 1.0
        })
        self.scheduler.adjust_priority(process.id, 2.0)
        updated_process = self.scheduler.get_process(process.id)
        self.assertEqual(updated_process.priority, 2.0)
    
    def test_load_balancing(self):
        """测试负载均衡"""
        processes = []
        for i in range(5):
            process = self.scheduler.create_process({
                "name": f"process_{i}",
                "priority": 1.0,
                "resources": {"cpu": 0.2, "memory": 256}
            })
            processes.append(process)
        
        balance_result = self.scheduler.balance_load()
        self.assertTrue(balance_result['balanced'])
        self.assertLess(balance_result['variance'], 0.1)

class TestContextManager(unittest.TestCase):
    """测试上下文管理器"""
    
    def setUp(self):
        self.context_manager = ContextManager()
    
    def test_context_tracking(self):
        """测试上下文追踪"""
        context = self.context_manager.create_context({
            "process_id": "123",
            "state": "running",
            "resources": {"cpu": 0.5, "memory": 512}
        })
        self.assertIsNotNone(context)
        tracked_context = self.context_manager.get_context("123")
        self.assertEqual(tracked_context['state'], "running")
    
    def test_context_switching(self):
        """测试上下文切换"""
        context1 = self.context_manager.create_context({
            "process_id": "123",
            "state": "running"
        })
        context2 = self.context_manager.create_context({
            "process_id": "456",
            "state": "ready"
        })
        
        switch_result = self.context_manager.switch_context("123", "456")
        self.assertTrue(switch_result['success'])
        self.assertLess(switch_result['switch_time'], 0.1)
    
    def test_dependency_management(self):
        """测试依赖关系管理"""
        self.context_manager.add_dependency("123", "456")
        dependencies = self.context_manager.get_dependencies("123")
        self.assertIn("456", dependencies)
        
        self.context_manager.remove_dependency("123", "456")
        dependencies = self.context_manager.get_dependencies("123")
        self.assertNotIn("456", dependencies)

class TestLoadPredictor(unittest.TestCase):
    """测试负载预测器"""
    
    def setUp(self):
        self.predictor = LoadPredictor()
    
    def test_data_collection(self):
        """测试数据收集"""
        data = {
            "timestamp": "2023-01-01T12:00:00",
            "cpu_usage": 0.5,
            "memory_usage": 0.6,
            "process_count": 10
        }
        self.predictor.collect_data(data)
        history = self.predictor.get_history()
        self.assertGreater(len(history), 0)
    
    def test_load_prediction(self):
        """测试负载预测"""
        # 收集一些历史数据
        for i in range(10):
            data = {
                "timestamp": f"2023-01-01T{12+i}:00:00",
                "cpu_usage": 0.5 + i * 0.02,
                "memory_usage": 0.6 + i * 0.01,
                "process_count": 10 + i
            }
            self.predictor.collect_data(data)
        
        prediction = self.predictor.predict_load(timeframe=3600)  # 1小时
        self.assertIsInstance(prediction, dict)
        self.assertIn('predicted_cpu', prediction)
        self.assertIn('predicted_memory', prediction)
        self.assertIn('confidence', prediction)
    
    def test_optimization_suggestions(self):
        """测试优化建议"""
        suggestions = self.predictor.get_optimization_suggestions()
        self.assertIsInstance(suggestions, list)
        for suggestion in suggestions:
            self.assertIn('type', suggestion)
            self.assertIn('description', suggestion)
            self.assertIn('expected_impact', suggestion)

if __name__ == '__main__':
    unittest.main()
