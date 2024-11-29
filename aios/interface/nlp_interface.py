"""
AIOS自然语言处理接口
提供自然语言交互能力
"""
from typing import Dict, List, Any, Optional
import asyncio
from transformers import pipeline
import spacy
import numpy as np

from ..kernel import AIKernel
from ..filesystem import AIFS
from ..process import SmartScheduler

class NLPInterface:
    """
    AIOS自然语言处理接口
    特点：
    1. 自然语言理解
    2. 意图识别
    3. 实体提取
    4. 上下文管理
    5. 对话管理
    """
    
    def __init__(self):
        """初始化NLP接口"""
        self.kernel = AIKernel()
        self.fs = AIFS()
        self.scheduler = SmartScheduler()
        
        # 加载NLP模型
        self.nlp = spacy.load("en_core_web_sm")
        self.intent_classifier = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli"
        )
        
        # 初始化意图映射
        self.intent_handlers = {
            'system_status': self._handle_status_intent,
            'file_operation': self._handle_file_intent,
            'process_operation': self._handle_process_intent,
            'help': self._handle_help_intent,
            'exit': self._handle_exit_intent
        }
        
        # 初始化上下文
        self.context: Dict[str, Any] = {}
    
    async def _classify_intent(self, text: str) -> str:
        """分类用户意图"""
        # 定义可能的意图
        candidate_intents = [
            "system_status",
            "file_operation",
            "process_operation",
            "help",
            "exit"
        ]
        
        # 使用模型进行分类
        results = self.intent_classifier(
            text,
            candidate_labels=candidate_intents
        )
        
        return results[0]['label']
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体"""
        doc = self.nlp(text)
        entities = {}
        
        # 提取命名实体
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities
    
    async def _handle_status_intent(
        self,
        text: str,
        entities: Dict[str, List[str]]
    ) -> str:
        """处理系统状态相关意图"""
        status = self.kernel.get_system_status()
        
        # 根据实体确定具体查询
        if 'RESOURCE' in entities:
            resource = entities['RESOURCE'][0].lower()
            if 'cpu' in resource:
                return f"CPU usage is {status['cpu_usage']}%"
            elif 'memory' in resource or 'ram' in resource:
                return f"Memory usage is {status['memory_usage']}%"
            elif 'disk' in resource:
                return f"Disk usage is {status['disk_usage']}%"
        
        # 默认返回所有状态
        return (
            f"System Status:\n"
            f"CPU: {status['cpu_usage']}%\n"
            f"Memory: {status['memory_usage']}%\n"
            f"Disk: {status['disk_usage']}%\n"
            f"Processes: {status['process_count']}"
        )
    
    async def _handle_file_intent(
        self,
        text: str,
        entities: Dict[str, List[str]]
    ) -> str:
        """处理文件操作相关意图"""
        # 提取文件路径
        path = (
            entities.get('PATH', ['.'])[0]
            if 'PATH' in entities
            else '.'
        )
        
        # 确定操作类型
        if 'list' in text.lower() or 'show' in text.lower():
            try:
                contents = await self.fs.list_directory(path)
                return "\n".join(
                    f"{item['name']} ({item['type']}, {item['size']})"
                    for item in contents
                )
            except Exception as e:
                return f"Error listing directory: {str(e)}"
        
        elif 'read' in text.lower() or 'cat' in text.lower():
            try:
                content = await self.fs.read_file(path)
                return f"Content of {path}:\n{content}"
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        return "Sorry, I don't understand the file operation"
    
    async def _handle_process_intent(
        self,
        text: str,
        entities: Dict[str, List[str]]
    ) -> str:
        """处理进程操作相关意图"""
        # 列出进程
        if 'list' in text.lower() or 'show' in text.lower():
            processes = self.scheduler.list_processes()
            return "\n".join(
                f"{proc['pid']}: {proc['name']} ({proc['status']})"
                for proc in processes
            )
        
        # 终止进程
        elif 'kill' in text.lower() or 'terminate' in text.lower():
            if 'NUMBER' in entities:
                try:
                    pid = int(entities['NUMBER'][0])
                    if self.scheduler.terminate_process(pid):
                        return f"Process {pid} terminated"
                    return f"Failed to terminate process {pid}"
                except ValueError:
                    return "Invalid process ID"
            return "No process ID specified"
        
        return "Sorry, I don't understand the process operation"
    
    async def _handle_help_intent(
        self,
        text: str,
        entities: Dict[str, List[str]]
    ) -> str:
        """处理帮助相关意图"""
        help_topics = {
            'general': """
            I can help you with:
            - Checking system status
            - Managing files
            - Managing processes
            Just ask in natural language!
            """,
            'status': """
            Ask about system status:
            - "How's the system doing?"
            - "What's the CPU usage?"
            - "Show me memory usage"
            """,
            'files': """
            File operations:
            - "List files in directory"
            - "Show content of file.txt"
            - "What's in this folder?"
            """,
            'processes': """
            Process operations:
            - "Show running processes"
            - "Kill process 1234"
            - "What processes are running?"
            """
        }
        
        # 确定具体帮助主题
        topic = 'general'
        for ent in entities.get('TOPIC', []):
            ent_lower = ent.lower()
            if ent_lower in help_topics:
                topic = ent_lower
                break
        
        return help_topics[topic].strip()
    
    async def _handle_exit_intent(
        self,
        text: str,
        entities: Dict[str, List[str]]
    ) -> str:
        """处理退出相关意图"""
        await self.shutdown()
        return "Goodbye!"
    
    async def process_input(self, text: str) -> str:
        """处理用户输入"""
        try:
            # 分类意图
            intent = await self._classify_intent(text)
            
            # 提取实体
            entities = self._extract_entities(text)
            
            # 更新上下文
            self.context.update({
                'last_input': text,
                'intent': intent,
                'entities': entities
            })
            
            # 处理意图
            if intent in self.intent_handlers:
                response = await self.intent_handlers[intent](text, entities)
            else:
                response = "I'm not sure how to help with that"
            
            # 更新上下文
            self.context['last_response'] = response
            
            return response
            
        except Exception as e:
            return f"Error processing input: {str(e)}"
    
    async def run(self):
        """运行NLP接口"""
        print("AIOS NLP Interface")
        print("Type 'exit' to quit")
        
        await self.kernel.start()
        
        while True:
            try:
                # 获取用户输入
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    input,
                    "You: "
                )
                
                if not user_input.strip():
                    continue
                
                # 处理输入
                response = await self.process_input(user_input)
                print(f"AIOS: {response}")
                
                # 检查是否退出
                if self.context.get('intent') == 'exit':
                    break
                
            except KeyboardInterrupt:
                await self.shutdown()
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    async def shutdown(self):
        """关闭系统"""
        await self.kernel.stop()
        print("System shutdown complete")

def main():
    """NLP接口入口点"""
    interface = NLPInterface()
    asyncio.run(interface.run())
