"""
AI File System Core
AI文件系统核心实现
"""
import os
import logging
import threading
from typing import Dict, List, Any, Optional, BinaryIO
from datetime import datetime
import aiofiles
import asyncio
from pathlib import Path
import shutil
import hashlib
from dataclasses import dataclass

@dataclass
class FileMetadata:
    """文件元数据"""
    path: str
    size: int
    created_at: datetime
    modified_at: datetime
    accessed_at: datetime
    content_type: str
    tags: List[str]
    importance: float
    checksum: str
    version: int
    is_compressed: bool
    compression_ratio: float
    access_count: int
    last_access_pattern: str

class AIFS:
    """
    AI文件系统核心类，提供：
    1. 智能文件操作
    2. 内容感知存储
    3. 预测性IO
    4. 自适应压缩
    5. 版本控制
    """
    
    def __init__(self, root_path: str):
        self.logger = logging.getLogger(__name__)
        self.root_path = Path(root_path)
        self._lock = threading.Lock()
        
        # 确保根目录存在
        self.root_path.mkdir(parents=True, exist_ok=True)
        
        # 元数据存储
        self.metadata: Dict[str, FileMetadata] = {}
        
        # 组件初始化
        from .content_analyzer import ContentAnalyzer
        from .smart_cache import SmartCache
        from .io_predictor import IOPredictor
        
        self.content_analyzer = ContentAnalyzer()
        self.smart_cache = SmartCache()
        self.io_predictor = IOPredictor()
        
        # 加载现有文件的元数据
        self._load_existing_files()

    async def write_file(self, relative_path: str, content: bytes,
                        tags: Optional[List[str]] = None) -> FileMetadata:
        """
        写入文件
        :param relative_path: 相对路径
        :param content: 文件内容
        :param tags: 文件标签
        :return: 文件元数据
        """
        abs_path = self.root_path / relative_path
        
        try:
            # 创建父目录
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 分析内容
            content_type = self.content_analyzer.analyze_content(content)
            importance = self.content_analyzer.calculate_importance(content)
            
            # 决定是否压缩
            should_compress = self._should_compress(content, content_type)
            if should_compress:
                content = self._compress_content(content)
            
            # 异步写入文件
            async with aiofiles.open(abs_path, 'wb') as f:
                await f.write(content)
            
            # 创建元数据
            metadata = FileMetadata(
                path=str(abs_path),
                size=len(content),
                created_at=datetime.now(),
                modified_at=datetime.now(),
                accessed_at=datetime.now(),
                content_type=content_type,
                tags=tags or [],
                importance=importance,
                checksum=self._calculate_checksum(content),
                version=1,
                is_compressed=should_compress,
                compression_ratio=1.0 if not should_compress else len(content) / len(content),
                access_count=0,
                last_access_pattern="write"
            )
            
            # 存储元数据
            with self._lock:
                self.metadata[str(abs_path)] = metadata
            
            # 更新缓存
            self.smart_cache.update(str(abs_path), content)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"文件写入失败 {abs_path}: {str(e)}")
            raise

    async def read_file(self, relative_path: str) -> tuple[bytes, FileMetadata]:
        """
        读取文件
        :param relative_path: 相对路径
        :return: (文件内容, 元数据)
        """
        abs_path = self.root_path / relative_path
        
        try:
            # 检查缓存
            cached_content = self.smart_cache.get(str(abs_path))
            if cached_content is not None:
                self._update_access_stats(str(abs_path), "cache_hit")
                return cached_content, self.metadata[str(abs_path)]
            
            # 异步读取文件
            async with aiofiles.open(abs_path, 'rb') as f:
                content = await f.read()
            
            # 如果文件被压缩，解压缩
            if self.metadata[str(abs_path)].is_compressed:
                content = self._decompress_content(content)
            
            # 更新访问统计
            self._update_access_stats(str(abs_path), "read")
            
            # 更新缓存
            self.smart_cache.update(str(abs_path), content)
            
            return content, self.metadata[str(abs_path)]
            
        except Exception as e:
            self.logger.error(f"文件读取失败 {abs_path}: {str(e)}")
            raise

    async def delete_file(self, relative_path: str):
        """
        删除文件
        :param relative_path: 相对路径
        """
        abs_path = self.root_path / relative_path
        
        try:
            # 删除文件
            if abs_path.exists():
                abs_path.unlink()
            
            # 清理元数据和缓存
            with self._lock:
                if str(abs_path) in self.metadata:
                    del self.metadata[str(abs_path)]
                self.smart_cache.remove(str(abs_path))
            
        except Exception as e:
            self.logger.error(f"文件删除失败 {abs_path}: {str(e)}")
            raise

    async def move_file(self, src_path: str, dst_path: str):
        """
        移动文件
        :param src_path: 源路径
        :param dst_path: 目标路径
        """
        abs_src = self.root_path / src_path
        abs_dst = self.root_path / dst_path
        
        try:
            # 创建目标目录
            abs_dst.parent.mkdir(parents=True, exist_ok=True)
            
            # 移动文件
            shutil.move(str(abs_src), str(abs_dst))
            
            # 更新元数据
            with self._lock:
                if str(abs_src) in self.metadata:
                    metadata = self.metadata[str(abs_src)]
                    metadata.path = str(abs_dst)
                    metadata.modified_at = datetime.now()
                    self.metadata[str(abs_dst)] = metadata
                    del self.metadata[str(abs_src)]
            
            # 更新缓存
            self.smart_cache.move(str(abs_src), str(abs_dst))
            
        except Exception as e:
            self.logger.error(f"文件移动失败 {abs_src} -> {abs_dst}: {str(e)}")
            raise

    async def copy_file(self, src_path: str, dst_path: str):
        """
        复制文件
        :param src_path: 源路径
        :param dst_path: 目标路径
        """
        abs_src = self.root_path / src_path
        abs_dst = self.root_path / dst_path
        
        try:
            # 创建目标目录
            abs_dst.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            shutil.copy2(str(abs_src), str(abs_dst))
            
            # 复制元数据
            with self._lock:
                if str(abs_src) in self.metadata:
                    metadata = self.metadata[str(abs_src)]
                    new_metadata = FileMetadata(
                        path=str(abs_dst),
                        size=metadata.size,
                        created_at=datetime.now(),
                        modified_at=datetime.now(),
                        accessed_at=datetime.now(),
                        content_type=metadata.content_type,
                        tags=metadata.tags.copy(),
                        importance=metadata.importance,
                        checksum=metadata.checksum,
                        version=1,
                        is_compressed=metadata.is_compressed,
                        compression_ratio=metadata.compression_ratio,
                        access_count=0,
                        last_access_pattern="copy"
                    )
                    self.metadata[str(abs_dst)] = new_metadata
            
            # 更新缓存
            cached_content = self.smart_cache.get(str(abs_src))
            if cached_content is not None:
                self.smart_cache.update(str(abs_dst), cached_content)
            
        except Exception as e:
            self.logger.error(f"文件复制失败 {abs_src} -> {abs_dst}: {str(e)}")
            raise

    def search_files(self, query: Dict[str, Any]) -> List[FileMetadata]:
        """
        搜索文件
        :param query: 搜索条件
        :return: 匹配的文件元数据列表
        """
        results = []
        
        try:
            for metadata in self.metadata.values():
                if self._matches_query(metadata, query):
                    results.append(metadata)
            
            # 按相关性排序
            results.sort(key=lambda x: x.importance, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"文件搜索失败: {str(e)}")
            return []

    def get_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        获取文件操作建议
        :param context: 上下文信息
        :return: 建议列表
        """
        suggestions = []
        
        try:
            # 基于访问模式的建议
            access_patterns = self.io_predictor.predict_access_patterns(context)
            for pattern in access_patterns:
                suggestions.append({
                    "type": "access_pattern",
                    "message": f"预计将访问: {pattern['files']}",
                    "confidence": pattern["confidence"]
                })
            
            # 基于存储优化的建议
            storage_suggestions = self._get_storage_suggestions()
            suggestions.extend(storage_suggestions)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"生成建议失败: {str(e)}")
            return []

    def _load_existing_files(self):
        """加载现有文件的元数据"""
        try:
            for file_path in self.root_path.rglob("*"):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        content = f.read()
                        
                    metadata = FileMetadata(
                        path=str(file_path),
                        size=file_path.stat().st_size,
                        created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                        modified_at=datetime.fromtimestamp(file_path.stat().st_mtime),
                        accessed_at=datetime.fromtimestamp(file_path.stat().st_atime),
                        content_type=self.content_analyzer.analyze_content(content),
                        tags=[],
                        importance=self.content_analyzer.calculate_importance(content),
                        checksum=self._calculate_checksum(content),
                        version=1,
                        is_compressed=False,
                        compression_ratio=1.0,
                        access_count=0,
                        last_access_pattern="load"
                    )
                    
                    self.metadata[str(file_path)] = metadata
                    
        except Exception as e:
            self.logger.error(f"加载现有文件失败: {str(e)}")

    def _calculate_checksum(self, content: bytes) -> str:
        """计算内容校验和"""
        return hashlib.sha256(content).hexdigest()

    def _should_compress(self, content: bytes, content_type: str) -> bool:
        """判断是否应该压缩文件"""
        # 根据文件类型和大小决定是否压缩
        if content_type.startswith(("image/", "video/", "audio/")):
            return False  # 多媒体文件通常已经压缩
        return len(content) > 1024  # 大于1KB的文件考虑压缩

    def _compress_content(self, content: bytes) -> bytes:
        """压缩内容"""
        import zlib
        return zlib.compress(content)

    def _decompress_content(self, content: bytes) -> bytes:
        """解压缩内容"""
        import zlib
        return zlib.decompress(content)

    def _update_access_stats(self, file_path: str, access_type: str):
        """更新访问统计"""
        with self._lock:
            if file_path in self.metadata:
                metadata = self.metadata[file_path]
                metadata.access_count += 1
                metadata.accessed_at = datetime.now()
                metadata.last_access_pattern = access_type

    def _matches_query(self, metadata: FileMetadata, query: Dict[str, Any]) -> bool:
        """检查元数据是否匹配查询条件"""
        for key, value in query.items():
            if key == "content_type" and value not in metadata.content_type:
                return False
            elif key == "tags" and not set(value).issubset(set(metadata.tags)):
                return False
            elif key == "min_size" and metadata.size < value:
                return False
            elif key == "max_size" and metadata.size > value:
                return False
            elif key == "modified_after" and metadata.modified_at < value:
                return False
        return True

    def _get_storage_suggestions(self) -> List[Dict[str, Any]]:
        """获取存储优化建议"""
        suggestions = []
        
        # 检查大文件
        large_files = [m for m in self.metadata.values() if m.size > 100*1024*1024]
        if large_files:
            suggestions.append({
                "type": "storage",
                "message": f"发现{len(large_files)}个大文件，建议考虑压缩或归档",
                "files": [f.path for f in large_files]
            })
        
        # 检查低访问频率的文件
        inactive_files = [m for m in self.metadata.values()
                         if (datetime.now() - m.accessed_at).days > 30]
        if inactive_files:
            suggestions.append({
                "type": "storage",
                "message": f"发现{len(inactive_files)}个超过30天未访问的文件",
                "files": [f.path for f in inactive_files]
            })
        
        return suggestions
