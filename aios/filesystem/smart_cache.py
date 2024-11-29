"""
Smart Cache
智能缓存系统，使用机器学习优化缓存策略
"""
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
from collections import OrderedDict
import joblib
from sklearn.ensemble import RandomForestClassifier

class CacheEntry:
    """缓存条目"""
    def __init__(self, content: bytes, size: int):
        self.content = content
        self.size = size
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 1
        self.access_pattern = []  # 访问时间间隔模式

class SmartCache:
    """
    智能缓存系统负责：
    1. 智能缓存替换
    2. 预测性缓存
    3. 访问模式学习
    4. 内存使用优化
    """
    
    def __init__(self, max_size: int = 100 * 1024 * 1024):  # 默认100MB
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # 缓存配置
        self.max_size = max_size
        self.current_size = 0
        
        # 缓存存储
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # 访问统计
        self.access_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
        # 机器学习模型
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_trained = False
        
        # 特征历史
        self.feature_history = []
        self.label_history = []

    def get(self, key: str) -> Optional[bytes]:
        """
        获取缓存内容
        :param key: 缓存键
        :return: 缓存内容
        """
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                self._update_access_stats(entry, "hit")
                self.access_stats["hits"] += 1
                return entry.content
            
            self.access_stats["misses"] += 1
            return None

    def update(self, key: str, content: bytes):
        """
        更新缓存
        :param key: 缓存键
        :param content: 缓存内容
        """
        with self._lock:
            size = len(content)
            
            # 如果内容太大，直接跳过
            if size > self.max_size:
                self.logger.warning(f"内容大小({size})超过缓存最大限制({self.max_size})")
                return
            
            # 如果键已存在，更新它
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size -= old_entry.size
                
            # 确保有足够空间
            while self.current_size + size > self.max_size:
                self._evict()
            
            # 添加新条目
            entry = CacheEntry(content, size)
            self.cache[key] = entry
            self.current_size += size
            
            # 收集训练数据
            self._collect_training_data(entry)
            
            # 定期训练模型
            if len(self.feature_history) >= 100 and not self.model_trained:
                self._train_model()

    def remove(self, key: str):
        """
        移除缓存项
        :param key: 缓存键
        """
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                self.current_size -= entry.size
                del self.cache[key]

    def move(self, old_key: str, new_key: str):
        """
        移动缓存项
        :param old_key: 原缓存键
        :param new_key: 新缓存键
        """
        with self._lock:
            if old_key in self.cache:
                entry = self.cache[old_key]
                del self.cache[old_key]
                self.cache[new_key] = entry

    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.current_size = 0
            self.access_stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0
            }

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self.access_stats["hits"] + self.access_stats["misses"]
            hit_rate = (self.access_stats["hits"] / total_requests
                       if total_requests > 0 else 0)
            
            return {
                "size": self.current_size,
                "max_size": self.max_size,
                "item_count": len(self.cache),
                "hit_rate": hit_rate,
                "evictions": self.access_stats["evictions"],
                "model_trained": self.model_trained
            }

    def _evict(self):
        """驱逐缓存项"""
        with self._lock:
            if not self.cache:
                return
            
            if self.model_trained:
                # 使用模型预测最应该驱逐的项
                scores = []
                for key, entry in self.cache.items():
                    features = self._extract_features(entry)
                    score = self.model.predict_proba([features])[0][1]  # 保留概率
                    scores.append((key, score))
                
                # 选择得分最低的项驱逐
                key_to_evict = min(scores, key=lambda x: x[1])[0]
                
            else:
                # 使用LRU策略
                key_to_evict = next(iter(self.cache))
            
            entry = self.cache[key_to_evict]
            self.current_size -= entry.size
            del self.cache[key_to_evict]
            self.access_stats["evictions"] += 1

    def _update_access_stats(self, entry: CacheEntry, access_type: str):
        """更新访问统计"""
        now = datetime.now()
        time_since_last = (now - entry.last_accessed).total_seconds()
        
        entry.access_count += 1
        entry.access_pattern.append(time_since_last)
        entry.last_accessed = now
        
        # 保持模式列表在合理大小
        if len(entry.access_pattern) > 100:
            entry.access_pattern = entry.access_pattern[-100:]

    def _extract_features(self, entry: CacheEntry) -> list:
        """提取特征"""
        now = datetime.now()
        return [
            entry.size,  # 大小
            entry.access_count,  # 访问次数
            (now - entry.created_at).total_seconds(),  # 存在时间
            (now - entry.last_accessed).total_seconds(),  # 上次访问后的时间
            np.mean(entry.access_pattern) if entry.access_pattern else 0,  # 平均访问间隔
            np.std(entry.access_pattern) if entry.access_pattern else 0  # 访问间隔标准差
        ]

    def _collect_training_data(self, entry: CacheEntry):
        """收集训练数据"""
        features = self._extract_features(entry)
        self.feature_history.append(features)
        
        # 标签：1表示应该保留，0表示可以驱逐
        label = 1 if entry.access_count > 5 else 0
        self.label_history.append(label)
        
        # 限制训练数据大小
        max_history = 10000
        if len(self.feature_history) > max_history:
            self.feature_history = self.feature_history[-max_history:]
            self.label_history = self.label_history[-max_history:]

    def _train_model(self):
        """训练模型"""
        try:
            X = np.array(self.feature_history)
            y = np.array(self.label_history)
            
            # 训练模型
            self.model.fit(X, y)
            self.model_trained = True
            
            # 保存模型
            joblib.dump(self.model, "cache_model.joblib")
            
            self.logger.info("缓存预测模型训练完成")
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")

    def optimize(self):
        """优化缓存配置"""
        with self._lock:
            if not self.model_trained:
                return
            
            try:
                # 评估所有缓存项
                scores = []
                for key, entry in self.cache.items():
                    features = self._extract_features(entry)
                    score = self.model.predict_proba([features])[0][1]
                    scores.append((key, score, entry.size))
                
                # 按得分排序
                scores.sort(key=lambda x: x[1], reverse=True)
                
                # 重新组织缓存
                new_cache = OrderedDict()
                current_size = 0
                
                for key, score, size in scores:
                    if current_size + size <= self.max_size:
                        new_cache[key] = self.cache[key]
                        current_size += size
                    else:
                        break
                
                self.cache = new_cache
                self.current_size = current_size
                
            except Exception as e:
                self.logger.error(f"缓存优化失败: {str(e)}")

    def preload(self, keys: list):
        """
        预加载可能需要的内容
        :param keys: 要预加载的键列表
        """
        if not self.model_trained:
            return
        
        try:
            # 预测每个键的重要性
            predictions = []
            for key in keys:
                if key in self.cache:
                    entry = self.cache[key]
                    features = self._extract_features(entry)
                    score = self.model.predict_proba([features])[0][1]
                    predictions.append((key, score))
            
            # 按预测分数排序
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # 优先保留高分项
            for key, _ in predictions:
                if key in self.cache:
                    entry = self.cache[key]
                    # 将项移到最近使用位置
                    self.cache.move_to_end(key)
                    
        except Exception as e:
            self.logger.error(f"预加载失败: {str(e)}")

    def get_recommendations(self) -> Dict[str, Any]:
        """获取缓存优化建议"""
        with self._lock:
            recommendations = {
                "actions": [],
                "stats": self.get_stats()
            }
            
            # 检查缓存使用率
            usage_ratio = self.current_size / self.max_size
            if usage_ratio > 0.9:
                recommendations["actions"].append({
                    "type": "warning",
                    "message": "缓存使用率超过90%",
                    "suggestion": "考虑增加缓存大小或清理低价值内容"
                })
            
            # 检查命中率
            hit_rate = (self.access_stats["hits"] /
                       (self.access_stats["hits"] + self.access_stats["misses"])
                       if (self.access_stats["hits"] + self.access_stats["misses"]) > 0
                       else 0)
            if hit_rate < 0.5:
                recommendations["actions"].append({
                    "type": "optimization",
                    "message": "缓存命中率低于50%",
                    "suggestion": "建议调整缓存策略或预加载机制"
                })
            
            # 检查驱逐率
            if self.access_stats["evictions"] > 100:
                recommendations["actions"].append({
                    "type": "info",
                    "message": f"频繁的缓存驱逐({self.access_stats['evictions']}次)",
                    "suggestion": "考虑增加缓存大小或优化缓存策略"
                })
            
            return recommendations
