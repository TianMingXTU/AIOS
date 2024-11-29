"""
AIOS System Test
"""
import logging
from utils.helpers import setup_logging
from core.task_scheduler import TaskScheduler
from core.resource_manager import ResourceManager
from core.ai_engine import AIEngine
from ai.nlp import NLPProcessor
from ai.decision import DecisionEngine
from ai.executor import TaskExecutor
import time

# 设置日志
logger = setup_logging()

def test_task_scheduler():
    logger.info("=== 测试任务调度器 ===")
    scheduler = TaskScheduler()
    scheduler.start()
    
    try:
        # 创建测试任务
        task1 = scheduler.create_task("测试任务1", 1)
        task2 = scheduler.create_task("测试任务2", 2)
        
        logger.info("已创建的任务：")
        for task in scheduler.list_tasks():
            logger.info(f"任务ID: {task.id}, 名称: {task.name}, 状态: {task.status}")
        
        time.sleep(2)  # 等待任务处理
    except Exception as e:
        logger.error(f"任务调度器测试失败: {str(e)}")
    finally:
        scheduler.stop()

def test_resource_manager():
    logger.info("=== 测试资源管理器 ===")
    manager = ResourceManager()
    manager.start_monitoring()
    
    try:
        # 获取系统资源状态
        resources = manager.get_system_resources()
        logger.info("系统资源状态：")
        logger.info(f"CPU使用率: {resources['cpu_percent']}%")
        logger.info(f"内存使用率: {resources['memory']['percent']}%")
        logger.info(f"磁盘使用率: {resources['disk']['percent']}%")
        
        time.sleep(2)
    except Exception as e:
        logger.error(f"资源管理器测试失败: {str(e)}")
    finally:
        manager.stop_monitoring()

def test_ai_components():
    logger.info("=== 测试AI组件 ===")
    
    # 测试NLP处理器
    try:
        nlp = NLPProcessor()
        result = nlp.process_command("你好，请帮我创建一个新任务")
        logger.info(f"NLP处理结果：{result}")
    except Exception as e:
        logger.error(f"NLP测试出错：{str(e)}")
    
    # 测试决策引擎
    try:
        decision = DecisionEngine()
        state = {
            "type": "task_management",
            "pending_tasks": 5,
            "system_load": 60
        }
        result = decision.predict({"state": state})
        logger.info(f"决策引擎结果：{result}")
    except Exception as e:
        logger.error(f"决策引擎测试失败: {str(e)}")
    
    # 测试任务执行器
    try:
        executor = TaskExecutor()
        def test_action(data):
            logger.info(f"执行测试任务: {data}")
            return "完成"
            
        executor.register_action("test", test_action)
        result = executor.predict({
            "type": "test",
            "data": "测试任务数据"
        })
        logger.info(f"任务执行结果：{result}")
    except Exception as e:
        logger.error(f"任务执行器测试失败: {str(e)}")

def main():
    logger.info("开始AIOS系统测试...")
    
    try:
        # 测试各个组件
        test_task_scheduler()
        test_resource_manager()
        test_ai_components()
        
        logger.info("所有测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
