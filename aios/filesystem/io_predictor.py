"""
IO Predictor
IO行为预测器，预测文件系统的访问模式
"""
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

class LSTMPredictor(nn.Module):
    """LSTM预测模型"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class IOPredictor:
    """
    IO预测器负责：
    1. 收集IO模式
    2. 预测文件访问
    3. 优化IO操作
    4. 提供访问建议
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # IO历史记录
        self.io_history: List[Dict[str, Any]] = []
        
        # 访问模式统计
        self.access_patterns = defaultdict(list)
        
        # 文件关联分析
        self.file_correlations = defaultdict(lambda: defaultdict(float))
        
        # LSTM模型配置
        self.input_size = 10  # 特征数量
        self.hidden_size = 64
        self.num_layers = 2
        self.sequence_length = 20
        
        # 模型和数据处理
        self.model = LSTMPredictor(self.input_size, self.hidden_size, self.num_layers)
        self.scaler = StandardScaler()
        self.model_trained = False

    def record_io(self, io_event: Dict[str, Any]):
        """
        记录IO事件
        :param io_event: IO事件信息
        """
        with self._lock:
            # 添加时间戳
            io_event["timestamp"] = datetime.now()
            self.io_history.append(io_event)
            
            # 更新访问模式
            self._update_access_pattern(io_event)
            
            # 更新文件关联
            self._update_file_correlations(io_event)
            
            # 限制历史记录大小
            if len(self.io_history) > 10000:
                self.io_history = self.io_history[-10000:]
            
            # 定期训练模型
            if len(self.io_history) >= 1000 and not self.model_trained:
                self._train_model()

    def predict_access_patterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        预测访问模式
        :param context: 上下文信息
        :return: 预测的访问模式
        """
        predictions = []
        
        try:
            if not self.model_trained:
                return self._get_simple_predictions(context)
            
            # 准备输入数据
            features = self._extract_features(context)
            features_scaled = self.scaler.transform([features])
            
            # 转换为PyTorch张量
            x = torch.FloatTensor(features_scaled).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                output = self.model(x)
            
            # 转换预测结果
            predicted_features = self.scaler.inverse_transform(
                output.numpy()
            )[0]
            
            # 解释预测结果
            predictions = self._interpret_predictions(predicted_features, context)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"访问模式预测失败: {str(e)}")
            return self._get_simple_predictions(context)

    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """获取IO优化建议"""
        suggestions = []
        
        try:
            # 分析IO模式
            patterns = self._analyze_io_patterns()
            
            # 检查高频访问
            if patterns["high_frequency_files"]:
                suggestions.append({
                    "type": "caching",
                    "message": "发现高频访问文件",
                    "files": patterns["high_frequency_files"],
                    "suggestion": "建议将这些文件预加载到缓存"
                })
            
            # 检查顺序访问模式
            if patterns["sequential_access"]:
                suggestions.append({
                    "type": "prefetch",
                    "message": "检测到顺序访问模式",
                    "files": patterns["sequential_access"],
                    "suggestion": "建议启用预读取优化"
                })
            
            # 检查随机访问模式
            if patterns["random_access"]:
                suggestions.append({
                    "type": "optimization",
                    "message": "检测到随机访问模式",
                    "files": patterns["random_access"],
                    "suggestion": "考虑优化文件布局或使用更适合的存储介质"
                })
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"生成优化建议失败: {str(e)}")
            return []

    def _update_access_pattern(self, io_event: Dict[str, Any]):
        """更新访问模式统计"""
        file_path = io_event.get("path", "")
        operation = io_event.get("operation", "")
        
        if file_path and operation:
            pattern = {
                "timestamp": io_event["timestamp"],
                "operation": operation,
                "size": io_event.get("size", 0)
            }
            self.access_patterns[file_path].append(pattern)
            
            # 限制每个文件的模式记录数
            if len(self.access_patterns[file_path]) > 1000:
                self.access_patterns[file_path] = \
                    self.access_patterns[file_path][-1000:]

    def _update_file_correlations(self, io_event: Dict[str, Any]):
        """更新文件关联统计"""
        current_file = io_event.get("path", "")
        if not current_file:
            return
            
        # 查找最近访问的其他文件
        recent_files = set()
        recent_time = io_event["timestamp"] - timedelta(seconds=10)
        
        for event in reversed(self.io_history[-100:]):
            if event["timestamp"] < recent_time:
                break
            if event.get("path") != current_file:
                recent_files.add(event.get("path"))
        
        # 更新关联度
        for other_file in recent_files:
            self.file_correlations[current_file][other_file] += 1
            self.file_correlations[other_file][current_file] += 1

    def _extract_features(self, context: Dict[str, Any]) -> List[float]:
        """提取特征"""
        features = []
        
        # 时间特征
        now = datetime.now()
        features.extend([
            float(now.hour) / 24,  # 小时
            float(now.weekday()) / 7,  # 星期
            float(now.day) / 31  # 日期
        ])
        
        # IO统计特征
        recent_events = [e for e in self.io_history[-100:]
                        if (now - e["timestamp"]).seconds <= 3600]
        
        features.extend([
            len(recent_events),  # 最近事件数
            sum(e.get("size", 0) for e in recent_events),  # 总IO大小
            len(set(e.get("path", "") for e in recent_events)),  # 不同文件数
            len([e for e in recent_events if e.get("operation") == "read"]),  # 读操作数
            len([e for e in recent_events if e.get("operation") == "write"])  # 写操作数
        ])
        
        # 上下文特征
        features.extend([
            context.get("cpu_usage", 0) / 100,  # CPU使用率
            context.get("memory_usage", 0) / 100  # 内存使用率
        ])
        
        return features

    def _train_model(self):
        """训练预测模型"""
        try:
            # 准备训练数据
            X, y = self._prepare_training_data()
            
            if len(X) < self.sequence_length:
                return
            
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 创建序列数据
            sequences = []
            targets = []
            
            for i in range(len(X_scaled) - self.sequence_length):
                seq = X_scaled[i:i + self.sequence_length]
                target = y[i + self.sequence_length]
                sequences.append(seq)
                targets.append(target)
            
            # 转换为PyTorch张量
            X_train = torch.FloatTensor(sequences)
            y_train = torch.FloatTensor(targets)
            
            # 训练模型
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters())
            
            for epoch in range(100):
                outputs = self.model(X_train)
                loss = criterion(outputs, y_train)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.model_trained = True
            
            # 保存模型
            torch.save(self.model.state_dict(), "io_predictor_model.pth")
            joblib.dump(self.scaler, "io_predictor_scaler.joblib")
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")

    def _prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        X = []
        y = []
        
        for i in range(len(self.io_history) - 1):
            features = self._extract_features({
                "timestamp": self.io_history[i]["timestamp"]
            })
            next_features = self._extract_features({
                "timestamp": self.io_history[i + 1]["timestamp"]
            })
            
            X.append(features)
            y.append(next_features)
        
        return np.array(X), np.array(y)

    def _get_simple_predictions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取简单的预测结果"""
        predictions = []
        
        # 基于最近的访问历史
        recent_files = [e.get("path") for e in self.io_history[-10:]
                       if e.get("path")]
        
        if recent_files:
            predictions.append({
                "type": "recent_access",
                "files": list(set(recent_files)),
                "confidence": 0.6
            })
        
        # 基于文件关联
        for file_path in recent_files:
            correlations = self.file_correlations[file_path]
            if correlations:
                related_files = sorted(
                    correlations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                predictions.append({
                    "type": "correlation",
                    "files": [f[0] for f in related_files],
                    "confidence": 0.4
                })
        
        return predictions

    def _interpret_predictions(self, features: np.ndarray,
                             context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解释预测结果"""
        predictions = []
        
        # 预测的IO强度
        io_intensity = features[3] + features[4]  # 读写操作数之和
        if io_intensity > 10:
            predictions.append({
                "type": "io_intensity",
                "message": "预计IO活动将增加",
                "confidence": min(0.9, io_intensity / 20)
            })
        
        # 预测的文件访问模式
        if features[5] > 5:  # 不同文件数
            predictions.append({
                "type": "access_pattern",
                "message": "预计将有大量文件访问",
                "confidence": min(0.9, features[5] / 10)
            })
        
        # 添加基于文件关联的预测
        recent_files = [e.get("path") for e in self.io_history[-5:]
                       if e.get("path")]
        for file_path in recent_files:
            correlations = self.file_correlations[file_path]
            if correlations:
                related_files = sorted(
                    correlations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                predictions.append({
                    "type": "related_files",
                    "files": [f[0] for f in related_files],
                    "confidence": 0.7
                })
        
        return predictions

    def _analyze_io_patterns(self) -> Dict[str, List[str]]:
        """分析IO模式"""
        patterns = {
            "high_frequency_files": [],
            "sequential_access": [],
            "random_access": []
        }
        
        # 分析每个文件的访问模式
        for file_path, events in self.access_patterns.items():
            if len(events) < 10:
                continue
            
            # 计算访问频率
            time_span = (events[-1]["timestamp"] -
                        events[0]["timestamp"]).total_seconds()
            frequency = len(events) / time_span if time_span > 0 else 0
            
            if frequency > 1:  # 每秒超过1次访问
                patterns["high_frequency_files"].append(file_path)
            
            # 分析访问顺序性
            if len(events) >= 3:
                sequential_count = 0
                random_count = 0
                
                for i in range(len(events) - 1):
                    if events[i + 1]["timestamp"] - events[i]["timestamp"] < \
                       timedelta(seconds=1):
                        sequential_count += 1
                    else:
                        random_count += 1
                
                if sequential_count > random_count:
                    patterns["sequential_access"].append(file_path)
                else:
                    patterns["random_access"].append(file_path)
        
        return patterns
