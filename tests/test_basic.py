"""
基本功能测试
"""
import unittest
import os
import tempfile
import shutil

class TestBasicFunctionality(unittest.TestCase):
    """测试系统基本功能"""
    
    def setUp(self):
        """测试准备"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir)
    
    def test_file_operations(self):
        """测试基本文件操作"""
        # 创建测试文件
        test_file = os.path.join(self.test_dir, "test.txt")
        content = "Hello, AIOS!"
        
        # 写入文件
        with open(test_file, "w") as f:
            f.write(content)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(test_file))
        
        # 读取文件
        with open(test_file, "r") as f:
            read_content = f.read()
        
        # 验证内容
        self.assertEqual(content, read_content)
    
    def test_directory_operations(self):
        """测试目录操作"""
        # 创建子目录
        sub_dir = os.path.join(self.test_dir, "subdir")
        os.makedirs(sub_dir)
        
        # 验证目录存在
        self.assertTrue(os.path.isdir(sub_dir))
        
        # 在子目录中创建文件
        test_file = os.path.join(sub_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test content")
        
        # 列出目录内容
        dir_contents = os.listdir(sub_dir)
        self.assertEqual(len(dir_contents), 1)
        self.assertEqual(dir_contents[0], "test.txt")
    
    def test_memory_operations(self):
        """测试内存操作"""
        # 创建大数组
        import numpy as np
        array_size = 1000000  # 1M元素
        
        # 分配内存
        array = np.zeros(array_size, dtype=np.float32)
        
        # 验证内存分配
        self.assertEqual(array.size, array_size)
        self.assertEqual(array.dtype, np.float32)
        
        # 写入数据
        array.fill(1.0)
        
        # 验证数据
        self.assertTrue(np.all(array == 1.0))
    
    def test_process_operations(self):
        """测试进程操作"""
        import psutil
        import subprocess
        import contextlib
        
        # 获取当前进程
        current_process = psutil.Process()
        
        # 验证进程存在
        self.assertTrue(current_process.is_running())
        
        # 使用上下文管理器启动和管理子进程
        with subprocess.Popen(
            ["python", "-c", "import time; time.sleep(1)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ) as proc:
            # 验证子进程
            self.assertIsNotNone(proc.pid)
            
            # 等待子进程结束
            proc.wait()
            
            # 验证进程已结束
            self.assertEqual(proc.returncode, 0)
            
            # 确保所有输出都被读取
            with contextlib.suppress(Exception):
                stdout, stderr = proc.communicate(timeout=1)
    
if __name__ == '__main__':
    unittest.main()
