"""
AIOS命令行界面
提供智能的命令行交互体验
"""
import os
import sys
import asyncio
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from ..kernel import AIKernel
from ..filesystem import AIFS
from ..process import SmartScheduler

class CLI:
    """
    AIOS命令行界面
    特点：
    1. 智能命令补全
    2. 上下文感知
    3. 自然语言理解
    4. 实时系统状态显示
    5. 交互式帮助
    """
    
    def __init__(self):
        """初始化CLI"""
        self.console = Console()
        self.kernel = AIKernel()
        self.fs = AIFS()
        self.scheduler = SmartScheduler()
        self.layout = self._create_layout()
        self.commands = self._initialize_commands()
        self.context: Dict[str, Any] = {}
        
    def _create_layout(self) -> Layout:
        """创建界面布局"""
        layout = Layout()
        
        # 创建主布局
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # 分割主区域
        layout["main"].split_row(
            Layout(name="sidebar", ratio=1),
            Layout(name="body", ratio=4)
        )
        
        return layout
    
    def _initialize_commands(self) -> Dict[str, callable]:
        """初始化命令集"""
        return {
            'help': self.cmd_help,
            'status': self.cmd_status,
            'ls': self.cmd_ls,
            'cd': self.cmd_cd,
            'cat': self.cmd_cat,
            'ps': self.cmd_ps,
            'kill': self.cmd_kill,
            'exec': self.cmd_exec,
            'clear': self.cmd_clear,
            'exit': self.cmd_exit
        }
    
    def _update_header(self):
        """更新头部信息"""
        system_status = self.kernel.get_system_status()
        header = Table.grid()
        header.add_row(
            f"CPU: {system_status['cpu_usage']}%",
            f"MEM: {system_status['memory_usage']}%",
            f"Tasks: {system_status['task_count']}"
        )
        self.layout["header"].update(Panel(header, title="System Status"))
    
    def _update_sidebar(self):
        """更新侧边栏"""
        # 显示文件系统树
        tree = self.fs.get_tree(self.context.get('current_dir', '/'))
        self.layout["sidebar"].update(Panel(tree, title="File System"))
    
    def _update_body(self, content: Any):
        """更新主体内容"""
        self.layout["body"].update(Panel(content, title="Output"))
    
    def _update_footer(self):
        """更新底部信息"""
        cwd = self.context.get('current_dir', '/')
        prompt = f"AIOS [{cwd}]> "
        self.layout["footer"].update(Text(prompt))
    
    async def cmd_help(self, args: List[str] = None):
        """显示帮助信息"""
        help_table = Table(title="AIOS Commands")
        help_table.add_column("Command")
        help_table.add_column("Description")
        
        help_info = {
            'help': 'Show this help message',
            'status': 'Show system status',
            'ls': 'List directory contents',
            'cd': 'Change directory',
            'cat': 'View file contents',
            'ps': 'List processes',
            'kill': 'Terminate process',
            'exec': 'Execute command',
            'clear': 'Clear screen',
            'exit': 'Exit AIOS'
        }
        
        for cmd, desc in help_info.items():
            help_table.add_row(cmd, desc)
        
        self._update_body(help_table)
    
    async def cmd_status(self, args: List[str] = None):
        """显示系统状态"""
        status = self.kernel.get_detailed_status()
        status_table = Table(title="System Status")
        
        for key, value in status.items():
            status_table.add_row(key, str(value))
        
        self._update_body(status_table)
    
    async def cmd_ls(self, args: List[str] = None):
        """列出目录内容"""
        path = args[0] if args else self.context.get('current_dir', '/')
        try:
            contents = await self.fs.list_directory(path)
            table = Table(title=f"Contents of {path}")
            table.add_column("Name")
            table.add_column("Type")
            table.add_column("Size")
            table.add_column("Modified")
            
            for item in contents:
                table.add_row(
                    item['name'],
                    item['type'],
                    item['size'],
                    item['modified']
                )
            
            self._update_body(table)
        except Exception as e:
            self._update_body(f"Error: {str(e)}")
    
    async def cmd_cd(self, args: List[str]):
        """更改当前目录"""
        if not args:
            self._update_body("Error: Path required")
            return
        
        try:
            new_path = await self.fs.resolve_path(args[0])
            if await self.fs.is_directory(new_path):
                self.context['current_dir'] = new_path
                self._update_sidebar()
                self._update_footer()
            else:
                self._update_body(f"Error: {args[0]} is not a directory")
        except Exception as e:
            self._update_body(f"Error: {str(e)}")
    
    async def cmd_cat(self, args: List[str]):
        """查看文件内容"""
        if not args:
            self._update_body("Error: File name required")
            return
        
        try:
            content = await self.fs.read_file(args[0])
            self._update_body(content)
        except Exception as e:
            self._update_body(f"Error: {str(e)}")
    
    async def cmd_ps(self, args: List[str] = None):
        """列出进程"""
        processes = self.scheduler.list_processes()
        table = Table(title="Processes")
        table.add_column("PID")
        table.add_column("Name")
        table.add_column("Status")
        table.add_column("CPU %")
        table.add_column("Memory %")
        
        for proc in processes:
            table.add_row(
                str(proc['pid']),
                proc['name'],
                proc['status'],
                f"{proc['cpu_percent']}%",
                f"{proc['memory_percent']}%"
            )
        
        self._update_body(table)
    
    async def cmd_kill(self, args: List[str]):
        """终止进程"""
        if not args:
            self._update_body("Error: PID required")
            return
        
        try:
            pid = int(args[0])
            result = self.scheduler.terminate_process(pid)
            self._update_body(
                "Process terminated successfully" if result
                else "Failed to terminate process"
            )
        except ValueError:
            self._update_body("Error: Invalid PID")
        except Exception as e:
            self._update_body(f"Error: {str(e)}")
    
    async def cmd_exec(self, args: List[str]):
        """执行命令"""
        if not args:
            self._update_body("Error: Command required")
            return
        
        try:
            result = await self.scheduler.execute_command(args)
            self._update_body(result)
        except Exception as e:
            self._update_body(f"Error: {str(e)}")
    
    async def cmd_clear(self, args: List[str] = None):
        """清屏"""
        self.console.clear()
    
    async def cmd_exit(self, args: List[str] = None):
        """退出系统"""
        await self.shutdown()
        sys.exit(0)
    
    async def process_command(self, cmd_line: str):
        """处理命令"""
        if not cmd_line.strip():
            return
        
        parts = cmd_line.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd in self.commands:
            try:
                await self.commands[cmd](args)
            except Exception as e:
                self._update_body(f"Error executing command: {str(e)}")
        else:
            self._update_body(f"Unknown command: {cmd}")
    
    async def run(self):
        """运行CLI"""
        # 初始化系统
        await self.kernel.start()
        self.context['current_dir'] = '/'
        
        # 显示欢迎信息
        welcome = Panel(
            """Welcome to AIOS - AI Operating System
Type 'help' for available commands""",
            title="AIOS CLI"
        )
        self.console.print(welcome)
        
        # 主循环
        with Live(self.layout, refresh_per_second=4):
            while True:
                try:
                    # 更新界面
                    self._update_header()
                    self._update_sidebar()
                    self._update_footer()
                    
                    # 获取用户输入
                    cmd = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: Prompt.ask(
                            f"AIOS [{self.context['current_dir']}]"
                        )
                    )
                    
                    # 处理命令
                    await self.process_command(cmd)
                    
                except KeyboardInterrupt:
                    await self.shutdown()
                    break
                except Exception as e:
                    self._update_body(f"Error: {str(e)}")
    
    async def shutdown(self):
        """关闭系统"""
        await self.kernel.stop()
        self.console.print("Goodbye!")

def main():
    """CLI入口点"""
    cli = CLI()
    asyncio.run(cli.run())

if __name__ == '__main__':
    main()
