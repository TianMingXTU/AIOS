"""
Command Line Interface for AIOS
提供交互式命令行界面
"""
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel
import sys
import cmd

from core.task_scheduler import TaskScheduler
from core.resource_manager import ResourceManager
from core.ai_engine import AIEngine

class CLI(cmd.Cmd):
    intro = """
    欢迎使用 AIOS (AI Operating System)
    输入 'help' 或 '?' 查看可用命令
    输入 'exit' 退出系统
    """
    prompt = 'AIOS> '

    def __init__(self):
        super().__init__()
        self.console = Console()
        self.task_scheduler = TaskScheduler()
        self.resource_manager = ResourceManager()
        self.ai_engine = AIEngine()
        
        # 启动核心服务
        self.task_scheduler.start()
        self.resource_manager.start_monitoring()

    def do_status(self, arg):
        """显示系统状态"""
        resources = self.resource_manager.get_system_resources()
        
        # 创建资源使用表格
        table = Table(title="系统状态")
        table.add_column("资源", style="cyan")
        table.add_column("使用情况", style="magenta")
        
        table.add_row("CPU 使用率", f"{resources['cpu_percent']}%")
        table.add_row("内存使用率", f"{resources['memory']['percent']}%")
        table.add_row("磁盘使用率", f"{resources['disk']['percent']}%")
        
        self.console.print(table)

    def do_tasks(self, arg):
        """显示所有任务"""
        tasks = self.task_scheduler.list_tasks()
        
        if not tasks:
            self.console.print("当前没有任务", style="yellow")
            return
            
        table = Table(title="任务列表")
        table.add_column("ID", style="cyan")
        table.add_column("名称", style="green")
        table.add_column("状态", style="magenta")
        table.add_column("优先级", style="blue")
        
        for task in tasks:
            table.add_row(
                str(task.id),
                task.name,
                task.status,
                str(task.priority)
            )
            
        self.console.print(table)

    def do_create_task(self, arg):
        """创建新任务"""
        name = Prompt.ask("任务名称")
        priority = int(Prompt.ask("优先级 (1-10)", default="1"))
        
        task = self.task_scheduler.create_task(name, priority)
        self.console.print(f"创建任务成功: {task.id}", style="green")

    def do_ai(self, arg):
        """AI模型管理"""
        models = self.ai_engine.list_models()
        
        if not models:
            self.console.print("当前没有可用的AI模型", style="yellow")
            return
            
        table = Table(title="AI模型列表")
        table.add_column("名称", style="cyan")
        table.add_column("类型", style="green")
        
        for name, model_type in models.items():
            table.add_row(name, model_type)
            
        self.console.print(table)

    def do_exit(self, arg):
        """退出系统"""
        self.task_scheduler.stop()
        self.resource_manager.stop_monitoring()
        self.console.print("正在关闭系统...", style="yellow")
        return True

    def default(self, line):
        """处理未知命令"""
        self.console.print(f"未知命令: {line}", style="red")
        self.console.print("输入 'help' 查看可用命令", style="yellow")

def main():
    try:
        cli = CLI()
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n正在退出...")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        sys.exit(0)

if __name__ == '__main__':
    main()
