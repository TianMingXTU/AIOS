"""
AIOS威胁检测系统
提供实时威胁检测和防护
"""
import os
import time
import json
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import psutil
import numpy as np
from sklearn.ensemble import IsolationForest
import threading
import queue

@dataclass
class ThreatRule:
    """威胁规则定义"""
    name: str
    description: str
    type: str  # system, process, network, file
    condition: str
    severity: int  # 1-5
    actions: List[str]

@dataclass
class ThreatEvent:
    """威胁事件定义"""
    timestamp: datetime
    rule_name: str
    description: str
    severity: int
    source: str
    details: dict
    status: str = "detected"  # detected, investigating, mitigated, false_positive

class ThreatDetector:
    """
    威胁检测器
    特点：
    1. 实时系统监控
    2. 异常行为检测
    3. 入侵检测
    4. 自动响应
    5. 威胁情报整合
    """
    
    def __init__(self, config_dir: str = "config/security"):
        """初始化威胁检测器"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.rules: Dict[str, ThreatRule] = {}
        self.events: List[ThreatEvent] = []
        self.baselines: Dict[str, dict] = {}
        
        # 异常检测模型
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        
        # 监控队列和线程
        self.monitoring_queue = queue.Queue()
        self.should_stop = threading.Event()
        
        # 加载配置
        self._load_config()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def _load_config(self):
        """加载配置"""
        # 加载威胁规则
        rules_file = self.config_dir / "threat_rules.json"
        if rules_file.exists():
            with open(rules_file) as f:
                rules_data = json.load(f)
                for rule_data in rules_data:
                    rule = ThreatRule(**rule_data)
                    self.rules[rule.name] = rule
        else:
            # 创建默认规则
            self._create_default_rules()
        
        # 加载基线数据
        baseline_file = self.config_dir / "baselines.json"
        if baseline_file.exists():
            with open(baseline_file) as f:
                self.baselines = json.load(f)
    
    def _create_default_rules(self):
        """创建默认威胁规则"""
        default_rules = [
            ThreatRule(
                name="high_cpu_usage",
                description="CPU usage exceeds threshold",
                type="system",
                condition="cpu_percent > 90",
                severity=3,
                actions=["log", "alert", "throttle"]
            ),
            ThreatRule(
                name="memory_exhaustion",
                description="Memory usage exceeds threshold",
                type="system",
                condition="memory_percent > 90",
                severity=4,
                actions=["log", "alert", "kill_processes"]
            ),
            ThreatRule(
                name="suspicious_process",
                description="Process exhibits suspicious behavior",
                type="process",
                condition="anomaly_score > 0.8",
                severity=4,
                actions=["log", "alert", "isolate"]
            ),
            ThreatRule(
                name="file_system_abuse",
                description="Unusual file system activity",
                type="file",
                condition="io_rate > 1000",
                severity=3,
                actions=["log", "alert", "limit_io"]
            ),
            ThreatRule(
                name="network_scan",
                description="Potential network scanning activity",
                type="network",
                condition="connection_rate > 100",
                severity=4,
                actions=["log", "alert", "block"]
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.name] = rule
        
        # 保存规则
        self._save_rules()
    
    def _save_rules(self):
        """保存威胁规则"""
        rules_data = [
            {
                "name": rule.name,
                "description": rule.description,
                "type": rule.type,
                "condition": rule.condition,
                "severity": rule.severity,
                "actions": rule.actions
            }
            for rule in self.rules.values()
        ]
        
        with open(self.config_dir / "threat_rules.json", "w") as f:
            json.dump(rules_data, f, indent=2)
    
    def _monitoring_loop(self):
        """监控循环"""
        while not self.should_stop.is_set():
            try:
                # 收集系统指标
                metrics = self._collect_metrics()
                
                # 检测威胁
                self._detect_threats(metrics)
                
                # 处理监控队列中的事件
                while not self.monitoring_queue.empty():
                    event = self.monitoring_queue.get_nowait()
                    self._handle_event(event)
                
                # 等待下一个监控周期
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self) -> dict:
        """收集系统指标"""
        metrics = {
            "timestamp": datetime.now(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent
            },
            "processes": [],
            "network": {
                "connections": len(psutil.net_connections()),
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            }
        }
        
        # 收集进程信息
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                metrics["processes"].append({
                    "pid": proc.info["pid"],
                    "name": proc.info["name"],
                    "cpu_percent": proc.info["cpu_percent"]
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return metrics
    
    def _detect_threats(self, metrics: dict):
        """检测威胁"""
        # 检查系统级威胁
        if metrics["system"]["cpu_percent"] > 90:
            self._create_threat_event(
                "high_cpu_usage",
                "High CPU usage detected",
                metrics["system"]
            )
        
        if metrics["system"]["memory_percent"] > 90:
            self._create_threat_event(
                "memory_exhaustion",
                "High memory usage detected",
                metrics["system"]
            )
        
        # 检查进程异常
        for proc in metrics["processes"]:
            if proc["cpu_percent"] > 50:  # 单个进程CPU使用率过高
                self._create_threat_event(
                    "suspicious_process",
                    f"High CPU usage from process {proc['name']}",
                    proc
                )
        
        # 检查网络异常
        if len(metrics["network"]["connections"]) > 1000:
            self._create_threat_event(
                "network_scan",
                "Unusual number of network connections",
                metrics["network"]
            )
    
    def _create_threat_event(
        self,
        rule_name: str,
        description: str,
        details: dict
    ):
        """创建威胁事件"""
        if rule_name not in self.rules:
            return
        
        rule = self.rules[rule_name]
        event = ThreatEvent(
            timestamp=datetime.now(),
            rule_name=rule_name,
            description=description,
            severity=rule.severity,
            source=rule.type,
            details=details
        )
        
        # 添加到事件列表
        self.events.append(event)
        
        # 添加到监控队列
        self.monitoring_queue.put(event)
    
    def _handle_event(self, event: ThreatEvent):
        """处理威胁事件"""
        rule = self.rules.get(event.rule_name)
        if not rule:
            return
        
        # 执行响应动作
        for action in rule.actions:
            if action == "log":
                self._log_event(event)
            elif action == "alert":
                self._send_alert(event)
            elif action == "throttle":
                self._throttle_resource(event)
            elif action == "kill_processes":
                self._kill_suspicious_processes(event)
            elif action == "isolate":
                self._isolate_process(event)
            elif action == "block":
                self._block_traffic(event)
    
    def _log_event(self, event: ThreatEvent):
        """记录事件"""
        log_entry = {
            "timestamp": event.timestamp.isoformat(),
            "rule": event.rule_name,
            "description": event.description,
            "severity": event.severity,
            "source": event.source,
            "details": event.details
        }
        
        log_file = self.config_dir / "threat.log"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _send_alert(self, event: ThreatEvent):
        """发送警报"""
        # TODO: 实现警报发送机制
        print(f"ALERT: {event.description}")
    
    def _throttle_resource(self, event: ThreatEvent):
        """限制资源使用"""
        if event.source == "system":
            if "cpu_percent" in event.details:
                # 使用cgroups限制CPU
                pass
            elif "memory_percent" in event.details:
                # 限制内存使用
                pass
    
    def _kill_suspicious_processes(self, event: ThreatEvent):
        """终止可疑进程"""
        if event.source == "process" and "pid" in event.details:
            try:
                process = psutil.Process(event.details["pid"])
                process.terminate()
            except psutil.NoSuchProcess:
                pass
    
    def _isolate_process(self, event: ThreatEvent):
        """隔离进程"""
        if event.source == "process" and "pid" in event.details:
            # TODO: 实现进程隔离
            pass
    
    def _block_traffic(self, event: ThreatEvent):
        """阻止网络流量"""
        if event.source == "network":
            # TODO: 实现网络阻断
            pass
    
    def train_anomaly_detectors(self):
        """训练异常检测器"""
        # 收集训练数据
        training_data = []
        for _ in range(100):
            metrics = self._collect_metrics()
            training_data.append([
                metrics["system"]["cpu_percent"],
                metrics["system"]["memory_percent"],
                len(metrics["processes"]),
                metrics["network"]["connections"]
            ])
            time.sleep(0.1)
        
        # 训练模型
        X = np.array(training_data)
        self.anomaly_detectors["system"] = IsolationForest(
            contamination=0.1
        ).fit(X)
    
    def check_anomaly(self, metrics: dict) -> float:
        """检查异常情况"""
        if "system" not in self.anomaly_detectors:
            return 0.0
        
        X = np.array([[
            metrics["system"]["cpu_percent"],
            metrics["system"]["memory_percent"],
            len(metrics["processes"]),
            metrics["network"]["connections"]
        ]])
        
        # 预测异常分数
        score = self.anomaly_detectors["system"].score_samples(X)[0]
        return 1 - (score + 0.5)  # 转换为0-1范围
    
    def get_recent_events(
        self,
        hours: int = 24,
        min_severity: int = 1
    ) -> List[ThreatEvent]:
        """获取最近的威胁事件"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.events
            if event.timestamp >= cutoff
            and event.severity >= min_severity
        ]
    
    def stop(self):
        """停止威胁检测"""
        self.should_stop.set()
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()
