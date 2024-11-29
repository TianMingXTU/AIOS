"""
AIOS资源隔离系统
提供进程、文件系统和网络隔离
"""
import os
import platform
from typing import Dict, List, Optional
import docker
from pathlib import Path
import psutil
import logging

if platform.system() != 'Windows':
    import pwd
    import grp
    import resource
else:
    import win32security
    import win32api
    import win32con
    import win32process
    import win32job

class ResourceIsolator:
    """
    资源隔离管理器
    特点：
    1. 进程隔离
    2. 文件系统隔离
    3. 网络隔离
    4. 资源限制
    """
    def __init__(self):
        self.namespaces: Dict[str, Dict] = {}
        self.containers: Dict[str, docker.models.containers.Container] = {}
        self.client = docker.from_env()
        self.logger = logging.getLogger(__name__)

    def set_resource_limits(self, process_id: int, limits: Dict[str, int]) -> None:
        """设置进程资源限制"""
        if platform.system() != 'Windows':
            # Unix系统使用resource模块
            if 'cpu_time' in limits:
                resource.setrlimit(resource.RLIMIT_CPU, (limits['cpu_time'], limits['cpu_time']))
            if 'memory' in limits:
                resource.setrlimit(resource.RLIMIT_AS, (limits['memory'], limits['memory']))
        else:
            # Windows系统使用作业对象限制资源
            try:
                handle = win32process.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, process_id)
                if 'memory' in limits:
                    job = win32job.CreateJobObject(None, f"AIOS_Job_{process_id}")
                    info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
                    info['ProcessMemoryLimit'] = limits['memory']
                    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
                    win32job.AssignProcessToJobObject(job, handle)
            except Exception as e:
                self.logger.error(f"设置Windows资源限制失败: {str(e)}")

    def isolate_process(self, pid: int, namespace: str) -> None:
        """将进程隔离到指定的命名空间"""
        try:
            if namespace not in self.namespaces:
                self.namespaces[namespace] = {
                    "processes": set(),
                    "memory_limit": 1024,  # 默认1GB
                    "disk_quota": 1024,    # 默认1GB
                }
            
            ns = self.namespaces[namespace]
            
            if platform.system() != 'Windows':
                # Unix系统资源限制
                memory_bytes = ns["memory_limit"] * 1024 * 1024
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (memory_bytes, memory_bytes)
                )
                
                resource.setrlimit(
                    resource.RLIMIT_FSIZE,
                    (ns["disk_quota"] * 1024 * 1024, -1)
                )
            else:
                # Windows系统资源限制
                self.set_resource_limits(pid, {
                    'memory': ns["memory_limit"] * 1024 * 1024
                })
            
            # 添加到命名空间
            ns["processes"].add(pid)
            
        except Exception as e:
            raise RuntimeError(f"Failed to isolate process: {e}")

    def create_namespace(
        self,
        name: str,
        cpu_limit: float = 1.0,
        memory_limit: int = 512,  # MB
        disk_quota: int = 1024,   # MB
        network_access: bool = True
    ) -> dict:
        """创建新的隔离命名空间"""
        if name in self.namespaces:
            raise ValueError(f"Namespace {name} already exists")
        
        namespace = {
            "name": name,
            "cpu_limit": cpu_limit,
            "memory_limit": memory_limit,
            "disk_quota": disk_quota,
            "network_access": network_access,
            "processes": set(),
            "mounts": set()
        }
        
        self.namespaces[name] = namespace
        return namespace
    
    def create_container(
        self,
        name: str,
        image: str,
        command: Optional[str] = None,
        cpu_limit: float = 1.0,
        memory_limit: int = 512,  # MB
        network_mode: str = "bridge",
        volumes: Optional[Dict[str, dict]] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> docker.models.containers.Container:
        """创建Docker容器"""
        try:
            container = self.client.containers.run(
                image=image,
                name=name,
                command=command,
                cpu_period=100000,
                cpu_quota=int(cpu_limit * 100000),
                mem_limit=f"{memory_limit}m",
                network_mode=network_mode,
                volumes=volumes,
                environment=environment,
                detach=True
            )
            
            self.containers[name] = container
            return container
            
        except docker.errors.APIError as e:
            raise RuntimeError(f"Failed to create container: {e}")
    
    def create_chroot(
        self,
        path: str,
        namespace: str
    ):
        """创建chroot环境"""
        if namespace not in self.namespaces:
            raise ValueError(f"Namespace {namespace} does not exist")
        
        root_path = Path(path)
        if not root_path.exists():
            root_path.mkdir(parents=True)
        
        # 创建基本目录结构
        for dir_name in ["bin", "lib", "lib64", "usr", "etc", "tmp"]:
            (root_path / dir_name).mkdir(exist_ok=True)
        
        # 复制必要的系统文件
        essential_files = [
            "/bin/bash",
            "/bin/ls",
            "/bin/cat",
            "/bin/cp",
            "/bin/mv"
        ]
        
        for file_path in essential_files:
            if os.path.exists(file_path):
                dest = root_path / file_path[1:]
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.system(f"cp -P {file_path} {dest}")
        
        # 添加到命名空间
        self.namespaces[namespace]["mounts"].add(str(root_path))
    
    def setup_network_isolation(
        self,
        namespace: str,
        interfaces: List[str] = None,
        bandwidth_limit: int = None  # KB/s
    ):
        """设置网络隔离"""
        if namespace not in self.namespaces:
            raise ValueError(f"Namespace {namespace} does not exist")
        
        if not self.namespaces[namespace]["network_access"]:
            raise ValueError(
                f"Network access is disabled for namespace {namespace}"
            )
        
        try:
            # 创建网络命名空间
            os.system(f"ip netns add {namespace}")
            
            if interfaces:
                for interface in interfaces:
                    # 将网络接口移动到命名空间
                    os.system(
                        f"ip link set {interface} netns {namespace}"
                    )
            
            if bandwidth_limit:
                # 使用tc限制带宽
                os.system(
                    f"tc qdisc add dev {interfaces[0]} root tbf "
                    f"rate {bandwidth_limit}kbit latency 50ms burst 1540"
                )
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup network isolation: {e}")
    
    def cleanup_namespace(self, namespace: str):
        """清理命名空间"""
        if namespace not in self.namespaces:
            return
        
        ns = self.namespaces[namespace]
        
        # 终止进程
        for pid in ns["processes"]:
            try:
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=5)
            except psutil.NoSuchProcess:
                continue
            except psutil.TimeoutExpired:
                process.kill()
        
        # 卸载文件系统
        for mount in ns["mounts"]:
            try:
                os.system(f"umount {mount}")
            except Exception:
                pass
        
        # 清理网络命名空间
        os.system(f"ip netns delete {namespace}")
        
        # 移除命名空间
        del self.namespaces[namespace]
    
    def cleanup_container(self, name: str):
        """清理容器"""
        if name in self.containers:
            try:
                container = self.containers[name]
                container.stop()
                container.remove()
                del self.containers[name]
            except Exception as e:
                raise RuntimeError(f"Failed to cleanup container: {e}")
    
    def cleanup_all(self):
        """清理所有资源"""
        # 清理命名空间
        for namespace in list(self.namespaces.keys()):
            self.cleanup_namespace(namespace)
        
        # 清理容器
        for name in list(self.containers.keys()):
            self.cleanup_container(name)
    
    def _get_user_info(self) -> Dict:
        """获取当前用户信息"""
        if platform.system() != 'Windows':
            pw = pwd.getpwuid(os.getuid())
            return {
                'name': pw.pw_name,
                'uid': pw.pw_uid,
                'gid': pw.pw_gid,
                'home': pw.pw_dir
            }
        else:
            token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
            sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
            name = win32security.LookupAccountSid(None, sid)[0]
            return {
                'name': name,
                'sid': str(sid),
                'home': os.path.expanduser('~')
            }
