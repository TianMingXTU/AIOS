"""
AI Engine Interface
提供AI功能的核心接口
"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class AIModel(ABC):
    """AI模型的抽象基类"""
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """模型预测接口"""
        pass

    @abstractmethod
    def train(self, training_data: Any):
        """模型训练接口"""
        pass

class AIEngine:
    def __init__(self):
        self.models: Dict[str, AIModel] = {}
        self.default_model: Optional[str] = None

    def register_model(self, name: str, model: AIModel, make_default: bool = False):
        """注册AI模型"""
        self.models[name] = model
        if make_default or self.default_model is None:
            self.default_model = name

    def get_model(self, name: Optional[str] = None) -> Optional[AIModel]:
        """获取指定的AI模型"""
        if name is None:
            name = self.default_model
        return self.models.get(name)

    def process(self, input_data: Any, model_name: Optional[str] = None) -> Any:
        """
        处理输入数据
        :param input_data: 输入数据
        :param model_name: 指定使用的模型名称
        :return: 处理结果
        """
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name or 'default'} not found")
        return model.predict(input_data)

    def train_model(self, model_name: str, training_data: Any):
        """
        训练指定模型
        :param model_name: 模型名称
        :param training_data: 训练数据
        """
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        model.train(training_data)

    def list_models(self) -> Dict[str, str]:
        """列出所有可用的模型"""
        return {name: model.__class__.__name__ for name, model in self.models.items()}
