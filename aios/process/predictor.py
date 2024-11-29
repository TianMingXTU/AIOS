"""
Load Predictor
负载预测器，使用机器学习预测系统负载和资源使用
"""
import logging
import threading
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class LoadPredictor:
    """
    负载预测器负责：
    1. 收集历史负载数据
    2. 训练预测模型
    3. 预测未来负载
    4. 提供负载优化建议
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # 数据存储
        self.load_history: List[Dict[str, Any]] = []
        self.feature_history: List[List[float]] = []
        self.target_history: List[float] = []
        
        # 机器学习模型
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # 模型状态
        self.model_trained = False
        self.model_path = model_path
        
        # 如果存在已保存的模型，加载它
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def add_load_data(self, load_data: Dict[str, Any]):
        """
        添加负载数据
        :param load_data: 负载数据
        """
        with self._lock:
            self.load_history.append({
                **load_data,
                "timestamp": datetime.now()
            })
            
            # 提取特征
            features = self._extract_features(load_data)
            self.feature_history.append(features)
            
            # 提取目标值（CPU使用率）
            self.target_history.append(load_data.get("cpu_usage", 0.0))
            
            # 限制历史记录大小
            max_history = 10000
            if len(self.load_history) > max_history:
                self.load_history = self.load_history[-max_history:]
                self.feature_history = self.feature_history[-max_history:]
                self.target_history = self.target_history[-max_history:]

    def train_model(self) -> bool:
        """
        训练预测模型
        :return: 训练是否成功
        """
        with self._lock:
            if len(self.feature_history) < 100:  # 需要足够的数据来训练
                self.logger.warning("训练数据不足")
                return False
            
            try:
                # 准备训练数据
                X = np.array(self.feature_history)
                y = np.array(self.target_history)
                
                # 标准化特征
                X_scaled = self.scaler.fit_transform(X)
                
                # 训练模型
                self.model.fit(X_scaled, y)
                self.model_trained = True
                
                # 保存模型
                if self.model_path:
                    self.save_model(self.model_path)
                
                return True
                
            except Exception as e:
                self.logger.error(f"模型训练失败: {str(e)}")
                return False

    def predict_load(self, current_state: Dict[str, Any],
                    horizon_minutes: int = 30) -> List[Dict[str, Any]]:
        """
        预测未来负载
        :param current_state: 当前系统状态
        :param horizon_minutes: 预测时间范围（分钟）
        :return: 预测结果列表
        """
        if not self.model_trained:
            self.logger.warning("模型未训练")
            return []
            
        try:
            predictions = []
            current_time = datetime.now()
            
            # 生成预测时间点
            for i in range(horizon_minutes):
                # 提取特征
                features = self._extract_features(current_state)
                features_scaled = self.scaler.transform([features])
                
                # 预测
                predicted_load = self.model.predict(features_scaled)[0]
                
                # 创建预测结果
                prediction_time = current_time + timedelta(minutes=i)
                prediction = {
                    "timestamp": prediction_time,
                    "predicted_cpu_usage": predicted_load,
                    "confidence": self._calculate_confidence(features)
                }
                predictions.append(prediction)
                
                # 更新状态用于下一次预测
                current_state = self._update_state_for_next_prediction(
                    current_state, predicted_load
                )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"负载预测失败: {str(e)}")
            return []

    def get_optimization_suggestions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于预测结果提供优化建议
        :param predictions: 预测结果列表
        :return: 优化建议列表
        """
        suggestions = []
        
        try:
            # 分析预测结果
            cpu_usage_trend = [p["predicted_cpu_usage"] for p in predictions]
            avg_usage = np.mean(cpu_usage_trend)
            max_usage = np.max(cpu_usage_trend)
            
            # 生成建议
            if max_usage > 90:
                suggestions.append({
                    "type": "warning",
                    "message": "预计CPU使用率将超过90%",
                    "action": "考虑启动额外的资源或推迟非关键任务"
                })
            
            if avg_usage > 70:
                suggestions.append({
                    "type": "suggestion",
                    "message": "预计平均CPU使用率较高",
                    "action": "建议优化资源密集型任务的调度"
                })
            
            # 检测负载波动
            usage_std = np.std(cpu_usage_trend)
            if usage_std > 20:
                suggestions.append({
                    "type": "optimization",
                    "message": "负载波动较大",
                    "action": "建议实施负载平衡策略"
                })
            
        except Exception as e:
            self.logger.error(f"生成优化建议失败: {str(e)}")
            
        return suggestions

    def save_model(self, path: str):
        """
        保存模型
        :param path: 保存路径
        """
        try:
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "trained_at": datetime.now()
            }
            joblib.dump(model_data, path)
            self.logger.info(f"模型已保存到: {path}")
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {str(e)}")

    def load_model(self, path: str):
        """
        加载模型
        :param path: 模型路径
        """
        try:
            model_data = joblib.load(path)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.model_trained = True
            self.logger.info(f"模型已从 {path} 加载")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")

    def _extract_features(self, state: Dict[str, Any]) -> List[float]:
        """
        从系统状态提取特征
        :param state: 系统状态
        :return: 特征列表
        """
        features = [
            state.get("cpu_usage", 0.0),
            state.get("memory_usage", 0.0),
            state.get("io_usage", 0.0),
            state.get("process_count", 0),
            state.get("active_connections", 0),
            float(datetime.now().hour),  # 时间特征
            float(datetime.now().weekday())  # 星期特征
        ]
        return features

    def _calculate_confidence(self, features: List[float]) -> float:
        """
        计算预测置信度
        :param features: 特征列表
        :return: 置信度分数
        """
        try:
            # 使用随机森林的预测方差作为不确定性度量
            predictions = []
            for estimator in self.model.estimators_:
                predictions.append(
                    estimator.predict(self.scaler.transform([features]))[0]
                )
            
            # 计算预测的标准差
            std = np.std(predictions)
            # 将标准差转换为0-1范围的置信度分数
            confidence = 1.0 / (1.0 + std)
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"置信度计算失败: {str(e)}")
            return 0.5

    def _update_state_for_next_prediction(self, current_state: Dict[str, Any],
                                        predicted_load: float) -> Dict[str, Any]:
        """
        更新状态用于下一次预测
        :param current_state: 当前状态
        :param predicted_load: 预测的负载
        :return: 更新后的状态
        """
        next_state = current_state.copy()
        next_state["cpu_usage"] = predicted_load
        
        # 可以添加更多的状态更新逻辑
        
        return next_state
