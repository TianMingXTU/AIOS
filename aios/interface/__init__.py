"""
AIOS智能接口层
提供多种交互方式：CLI、API和自然语言处理
"""

from .cli import CLI
from .api import app
from .nlp_interface import NLPInterface

__all__ = ['CLI', 'app', 'NLPInterface']
