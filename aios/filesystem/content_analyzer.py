"""
Content Analyzer
内容分析器，负责分析和理解文件内容
"""
import logging
import threading
import os
import platform
from typing import Dict, List, Any, Optional
import mimetypes
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import json

class ContentAnalyzer:
    """
    内容分析器负责：
    1. 文件类型识别
    2. 内容理解
    3. 重要性评估
    4. 内容相似度分析
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # 文本内容分析模型
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # 内容向量缓存
        self.content_vectors: Dict[str, np.ndarray] = {}
        
        # 相似度阈值
        self.similarity_threshold = 0.8

    def analyze_content(self, content: bytes) -> str:
        """
        分析文件内容
        :param content: 文件内容
        :return: 内容类型
        """
        try:
            # 检测MIME类型
            mime_type = self.get_mime_type("temp_file")
            
            # 对于文本文件，进行更深入的分析
            if mime_type.startswith("text/"):
                text_content = content.decode('utf-8', errors='ignore')
                if self._is_source_code(text_content):
                    mime_type = f"text/x-{self._detect_programming_language(text_content)}"
                elif self._is_config_file(text_content):
                    mime_type = "text/x-config"
            
            return mime_type
            
        except Exception as e:
            self.logger.error(f"内容分析失败: {str(e)}")
            return "application/octet-stream"

    def calculate_importance(self, content: bytes) -> float:
        """
        计算内容重要性
        :param content: 文件内容
        :return: 重要性分数 (0-1)
        """
        try:
            mime_type = self.analyze_content(content)
            importance = 0.5  # 默认重要性
            
            # 基于文件类型的重要性
            if mime_type.startswith("text/"):
                importance = self._calculate_text_importance(content)
            elif mime_type.startswith("image/"):
                importance = self._calculate_image_importance(content)
            elif mime_type.startswith("application/"):
                importance = self._calculate_application_importance(content)
            
            # 考虑文件大小
            size_factor = min(1.0, len(content) / (10 * 1024 * 1024))  # 10MB作为参考
            importance = 0.7 * importance + 0.3 * size_factor
            
            return importance
            
        except Exception as e:
            self.logger.error(f"重要性计算失败: {str(e)}")
            return 0.5

    def find_similar_content(self, content: bytes,
                           max_results: int = 5) -> List[Dict[str, Any]]:
        """
        查找相似内容
        :param content: 目标内容
        :param max_results: 最大结果数
        :return: 相似内容列表
        """
        try:
            # 获取内容向量
            vector = self._get_content_vector(content)
            
            # 计算相似度
            similarities = []
            for path, cached_vector in self.content_vectors.items():
                similarity = self._calculate_similarity(vector, cached_vector)
                if similarity > self.similarity_threshold:
                    similarities.append({
                        "path": path,
                        "similarity": similarity
                    })
            
            # 按相似度排序
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:max_results]
            
        except Exception as e:
            self.logger.error(f"相似内容查找失败: {str(e)}")
            return []

    def _is_source_code(self, content: str) -> bool:
        """判断是否为源代码"""
        # 检查常见的编程语言特征
        code_indicators = [
            "def ", "class ", "function", "import ", "from ", "#include",
            "public ", "private ", "protected ", "var ", "let ", "const "
        ]
        return any(indicator in content for indicator in code_indicators)

    def _detect_programming_language(self, content: str) -> str:
        """检测编程语言"""
        # 简单的特征匹配
        if "def " in content or "import " in content:
            return "python"
        elif "function" in content or "var " in content:
            return "javascript"
        elif "#include" in content:
            return "cpp"
        elif "public class" in content:
            return "java"
        return "unknown"

    def _is_config_file(self, content: str) -> bool:
        """判断是否为配置文件"""
        try:
            # 尝试解析为JSON
            json.loads(content)
            return True
        except:
            # 检查其他配置文件特征
            config_indicators = [
                "=", ":", "[section]", "<!-- ", "<?xml"
            ]
            return any(indicator in content for indicator in config_indicators)

    def _calculate_text_importance(self, content: bytes) -> float:
        """计算文本重要性"""
        try:
            text = content.decode('utf-8', errors='ignore')
            
            # 基于文本长度
            length_score = min(1.0, len(text) / 10000)  # 10000字符作为参考
            
            # 基于关键词
            important_keywords = [
                "password", "secret", "api", "key", "token", "config",
                "important", "critical", "urgent", "backup"
            ]
            keyword_count = sum(1 for keyword in important_keywords
                              if keyword in text.lower())
            keyword_score = min(1.0, keyword_count / 5)  # 5个关键词作为参考
            
            # 基于结构复杂度
            structure_score = 0.5
            if self._is_source_code(text):
                # 计算代码复杂度
                lines = text.split('\n')
                indentation_levels = [len(line) - len(line.lstrip())
                                    for line in lines if line.strip()]
                if indentation_levels:
                    structure_score = min(1.0, max(indentation_levels) / 16)
            
            return (0.4 * length_score +
                   0.4 * keyword_score +
                   0.2 * structure_score)
            
        except Exception as e:
            self.logger.error(f"文本重要性计算失败: {str(e)}")
            return 0.5

    def _calculate_image_importance(self, content: bytes) -> float:
        """计算图像重要性"""
        try:
            # 基于图像大小
            size_score = min(1.0, len(content) / (5 * 1024 * 1024))  # 5MB作为参考
            
            # 这里可以添加更多图像分析逻辑，如：
            # - 图像分辨率
            # - 图像质量
            # - 图像内容识别
            
            return size_score
            
        except Exception as e:
            self.logger.error(f"图像重要性计算失败: {str(e)}")
            return 0.5

    def _calculate_application_importance(self, content: bytes) -> float:
        """计算应用程序文件重要性"""
        try:
            # 基于文件大小
            size_score = min(1.0, len(content) / (50 * 1024 * 1024))  # 50MB作为参考
            
            # 检查是否为可执行文件
            is_executable = content.startswith(b'MZ') or content.startswith(b'\x7fELF')
            executable_score = 0.8 if is_executable else 0.5
            
            return 0.6 * size_score + 0.4 * executable_score
            
        except Exception as e:
            self.logger.error(f"应用程序重要性计算失败: {str(e)}")
            return 0.5

    def _get_content_vector(self, content: bytes) -> np.ndarray:
        """获取内容的向量表示"""
        try:
            # 初始化模型（如果需要）
            if not self.model_loaded:
                self._load_model()
            
            # 对于文本内容，使用BERT生成向量
            if self.analyze_content(content).startswith("text/"):
                text = content.decode('utf-8', errors='ignore')
                inputs = self.tokenizer(text, return_tensors="pt",
                                      max_length=512, truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).numpy()
            
            # 对于非文本内容，使用简单的统计特征
            return np.array([
                len(content),  # 大小
                sum(content) / len(content),  # 平均字节值
                np.std(list(content))  # 字节值标准差
            ])
            
        except Exception as e:
            self.logger.error(f"内容向量生成失败: {str(e)}")
            return np.zeros(768)  # BERT向量维度

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算向量相似度"""
        try:
            # 使用余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0
            
        except Exception as e:
            self.logger.error(f"相似度计算失败: {str(e)}")
            return 0.0

    def _load_model(self):
        """加载BERT模型"""
        try:
            model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model_loaded = True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise

    def get_mime_type(self, file_path: str) -> str:
        """获取文件的MIME类型"""
        try:
            # 使用 mimetypes 模块（跨平台支持）
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                return mime_type
            
            # 如果无法识别，尝试通过文件头识别
            with open(file_path, 'rb') as f:
                header = f.read(2048)
                
            # 常见文件头识别
            if header.startswith(b'%PDF'):
                return 'application/pdf'
            elif header.startswith(b'\x89PNG'):
                return 'image/png'
            elif header.startswith(b'\xFF\xD8'):
                return 'image/jpeg'
            elif header.startswith(b'PK'):
                return 'application/zip'
            elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                return 'image/gif'
            elif header.startswith(b'\x1F\x8B'):
                return 'application/gzip'
            
            # 检查是否为文本文件
            try:
                header.decode('utf-8')
                return 'text/plain'
            except UnicodeDecodeError:
                pass
                
            return 'application/octet-stream'
            
        except Exception as e:
            logging.error(f"Error getting MIME type for {file_path}: {e}")
            return 'application/octet-stream'
