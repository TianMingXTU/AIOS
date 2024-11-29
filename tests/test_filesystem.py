"""
文件系统层测试
"""
import unittest
import asyncio
import os
import tempfile
from aios.filesystem import AIFS, ContentAnalyzer, SmartCache, IOPredictor

class TestAIFS(unittest.TestCase):
    """测试AI文件系统"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fs = AIFS(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_file_operations(self):
        """测试文件操作"""
        # 写入文件
        content = b"Hello, AIOS!"
        self.fs.write_file("test.txt", content)
        
        # 读取文件
        read_content = self.fs.read_file("test.txt")
        self.assertEqual(read_content, content)
        
        # 获取文件元数据
        metadata = self.fs.get_metadata("test.txt")
        self.assertIn('size', metadata)
        self.assertIn('created_at', metadata)
        self.assertIn('content_type', metadata)
    
    def test_content_aware_storage(self):
        """测试内容感知存储"""
        # 写入不同类型的内容
        text_content = b"This is a text file"
        image_content = b"Fake image content"
        
        self.fs.write_file("doc.txt", text_content)
        self.fs.write_file("image.jpg", image_content)
        
        # 检查存储优化
        text_metadata = self.fs.get_metadata("doc.txt")
        image_metadata = self.fs.get_metadata("image.jpg")
        
        self.assertNotEqual(
            text_metadata['storage_strategy'],
            image_metadata['storage_strategy']
        )
    
    def test_version_control(self):
        """测试版本控制"""
        # 创建文件并进行多次修改
        self.fs.write_file("versioned.txt", b"Version 1")
        self.fs.write_file("versioned.txt", b"Version 2")
        self.fs.write_file("versioned.txt", b"Version 3")
        
        # 获取版本历史
        history = self.fs.get_version_history("versioned.txt")
        self.assertEqual(len(history), 3)
        
        # 恢复到之前的版本
        self.fs.restore_version("versioned.txt", history[0]['version'])
        content = self.fs.read_file("versioned.txt")
        self.assertEqual(content, b"Version 1")

class TestContentAnalyzer(unittest.TestCase):
    """测试内容分析器"""
    
    def setUp(self):
        self.analyzer = ContentAnalyzer()
    
    def test_file_type_detection(self):
        """测试文件类型检测"""
        text_content = b"This is a text file"
        result = self.analyzer.detect_type(text_content)
        self.assertEqual(result['type'], 'text/plain')
    
    def test_content_understanding(self):
        """测试内容理解"""
        content = "This is a document about artificial intelligence"
        understanding = self.analyzer.understand_content(content)
        self.assertIn('topics', understanding)
        self.assertIn('keywords', understanding)
        self.assertIn('importance_score', understanding)
    
    def test_similarity_analysis(self):
        """测试相似度分析"""
        content1 = "Document about AI"
        content2 = "Article about artificial intelligence"
        content3 = "Recipe for chocolate cake"
        
        similarity = self.analyzer.compare_content(content1, content2)
        self.assertGreater(similarity, 0.5)
        
        similarity = self.analyzer.compare_content(content1, content3)
        self.assertLess(similarity, 0.3)

class TestSmartCache(unittest.TestCase):
    """测试智能缓存"""
    
    def setUp(self):
        self.cache = SmartCache(max_size=1024)
    
    def test_cache_operations(self):
        """测试缓存操作"""
        self.cache.put("key1", b"value1")
        self.assertTrue(self.cache.contains("key1"))
        value = self.cache.get("key1")
        self.assertEqual(value, b"value1")
    
    def test_cache_eviction(self):
        """测试缓存淘汰"""
        # 填充缓存
        for i in range(100):
            self.cache.put(f"key{i}", b"x" * 20)
        
        # 检查缓存大小限制
        self.assertLessEqual(self.cache.current_size, self.cache.max_size)
    
    def test_access_pattern_learning(self):
        """测试访问模式学习"""
        # 模拟访问模式
        for _ in range(10):
            self.cache.get("key1")
            self.cache.get("key2")
            self.cache.get("key3")
        
        patterns = self.cache.get_access_patterns()
        self.assertTrue(any(p['key'] == "key1" for p in patterns))
        self.assertGreater(patterns[0]['frequency'], patterns[-1]['frequency'])

class TestIOPredictor(unittest.TestCase):
    """测试IO预测器"""
    
    def setUp(self):
        self.predictor = IOPredictor()
    
    def test_pattern_collection(self):
        """测试模式收集"""
        event = {
            "operation": "read",
            "path": "test.txt",
            "size": 1024,
            "timestamp": "2023-01-01T12:00:00"
        }
        self.predictor.record_io(event)
        history = self.predictor.io_history
        self.assertGreater(len(history), 0)
    
    def test_access_prediction(self):
        """测试访问预测"""
        # 记录一些IO事件
        for i in range(10):
            event = {
                "operation": "read",
                "path": f"file{i}.txt",
                "size": 1024,
                "timestamp": f"2023-01-01T{12+i}:00:00"
            }
            self.predictor.record_io(event)
        
        # 获取预测
        context = {"current_file": "file0.txt"}
        predictions = self.predictor.predict_access_patterns(context)
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)
    
    def test_optimization_suggestions(self):
        """测试优化建议"""
        # 记录一些IO事件以生成建议
        for i in range(20):
            event = {
                "operation": "read" if i % 2 == 0 else "write",
                "path": f"file{i % 5}.txt",
                "size": 1024,
                "timestamp": f"2023-01-01T{12+i}:00:00"
            }
            self.predictor.record_io(event)
        
        suggestions = self.predictor.get_optimization_suggestions()
        self.assertIsInstance(suggestions, list)
        for suggestion in suggestions:
            self.assertIn('type', suggestion)
            self.assertIn('message', suggestion)
            self.assertIn('suggestion', suggestion)

if __name__ == '__main__':
    unittest.main()
