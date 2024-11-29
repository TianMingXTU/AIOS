"""
Helper functions for AIOS
"""
import logging
import sys
from rich.logging import RichHandler

def setup_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    return logger

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def safe_import(module_name):
    """安全导入模块"""
    try:
        return __import__(module_name)
    except ImportError as e:
        logging.warning(f"无法导入模块 {module_name}: {str(e)}")
        return None
