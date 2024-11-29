"""
AIOS访问控制系统
提供细粒度的权限管理和访问控制
"""
from typing import Dict, List, Optional, Set
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import jwt
from passlib.hash import bcrypt
from fastapi import HTTPException, status

@dataclass
class Permission:
    """权限定义"""
    name: str
    description: str
    resource_type: str  # file, process, network, system
    access_level: str   # read, write, execute, admin

@dataclass
class Role:
    """角色定义"""
    name: str
    description: str
    permissions: Set[str]

@dataclass
class User:
    """用户定义"""
    username: str
    password_hash: str
    roles: Set[str]
    is_active: bool = True
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

class AccessManager:
    """
    访问控制管理器
    特点：
    1. 基于角色的访问控制（RBAC）
    2. 细粒度的权限管理
    3. 用户认证和授权
    4. 会话管理
    5. 审计日志
    """
    
    def __init__(self, config_dir: str = "config/security"):
        """初始化访问控制管理器"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, dict] = {}
        
        # JWT配置
        self.jwt_secret = os.getenv("AIOS_JWT_SECRET", "your-secret-key")
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = timedelta(hours=1)
        
        # 加载默认配置
        self._load_default_config()
        
        # 加载持久化数据
        self._load_data()
    
    def _load_default_config(self):
        """加载默认配置"""
        # 默认权限
        default_permissions = [
            Permission("file_read", "Read file contents", "file", "read"),
            Permission("file_write", "Write to files", "file", "write"),
            Permission("file_execute", "Execute files", "file", "execute"),
            Permission("process_view", "View processes", "process", "read"),
            Permission("process_manage", "Manage processes", "process", "write"),
            Permission("system_view", "View system status", "system", "read"),
            Permission("system_manage", "Manage system", "system", "admin"),
            Permission("network_access", "Access network", "network", "read"),
            Permission("network_manage", "Manage network", "network", "write")
        ]
        
        for perm in default_permissions:
            self.permissions[perm.name] = perm
        
        # 默认角色
        default_roles = [
            Role("admin", "System administrator", {
                "file_read", "file_write", "file_execute",
                "process_view", "process_manage",
                "system_view", "system_manage",
                "network_access", "network_manage"
            }),
            Role("user", "Regular user", {
                "file_read", "file_write",
                "process_view",
                "system_view",
                "network_access"
            }),
            Role("guest", "Guest user", {
                "file_read",
                "process_view",
                "system_view"
            })
        ]
        
        for role in default_roles:
            self.roles[role.name] = role
        
        # 默认管理员用户
        if not self.users:
            admin_password = os.getenv("AIOS_ADMIN_PASSWORD", "admin")
            self.create_user("admin", admin_password, {"admin"})
    
    def _load_data(self):
        """从持久化存储加载数据"""
        try:
            # 加载用户数据
            users_file = self.config_dir / "users.json"
            if users_file.exists():
                with open(users_file) as f:
                    users_data = json.load(f)
                    for username, data in users_data.items():
                        self.users[username] = User(**data)
            
            # 加载会话数据
            sessions_file = self.config_dir / "sessions.json"
            if sessions_file.exists():
                with open(sessions_file) as f:
                    self.sessions = json.load(f)
                    
        except Exception as e:
            print(f"Error loading security data: {e}")
    
    def _save_data(self):
        """保存数据到持久化存储"""
        try:
            # 保存用户数据
            users_data = {
                username: asdict(user)
                for username, user in self.users.items()
            }
            with open(self.config_dir / "users.json", "w") as f:
                json.dump(users_data, f, indent=2, default=str)
            
            # 保存会话数据
            with open(self.config_dir / "sessions.json", "w") as f:
                json.dump(self.sessions, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving security data: {e}")
    
    def create_user(
        self,
        username: str,
        password: str,
        roles: Set[str],
        save: bool = True
    ) -> User:
        """创建新用户"""
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        # 验证角色
        for role in roles:
            if role not in self.roles:
                raise ValueError(f"Invalid role: {role}")
        
        # 创建用户
        user = User(
            username=username,
            password_hash=bcrypt.hash(password),
            roles=roles
        )
        self.users[username] = user
        
        if save:
            self._save_data()
        
        return user
    
    def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[User]:
        """认证用户"""
        user = self.users.get(username)
        if not user:
            return None
        
        # 检查账户锁定
        if user.locked_until and user.locked_until > datetime.now():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is locked"
            )
        
        # 验证密码
        if not bcrypt.verify(password, user.password_hash):
            user.failed_attempts += 1
            
            # 锁定账户
            if user.failed_attempts >= 5:
                user.locked_until = datetime.now() + timedelta(minutes=15)
            
            self._save_data()
            return None
        
        # 重置失败尝试
        user.failed_attempts = 0
        user.last_login = datetime.now()
        self._save_data()
        
        return user
    
    def create_access_token(self, username: str) -> str:
        """创建访问令牌"""
        user = self.users.get(username)
        if not user:
            raise ValueError(f"User {username} not found")
        
        # 创建JWT
        expiration = datetime.utcnow() + self.jwt_expiration
        token_data = {
            "sub": username,
            "roles": list(user.roles),
            "exp": expiration
        }
        
        return jwt.encode(
            token_data,
            self.jwt_secret,
            algorithm=self.jwt_algorithm
        )
    
    def verify_access_token(self, token: str) -> dict:
        """验证访问令牌"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            username = payload.get("sub")
            if username not in self.users:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid user"
                )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def check_permission(
        self,
        username: str,
        permission: str
    ) -> bool:
        """检查用户权限"""
        user = self.users.get(username)
        if not user or not user.is_active:
            return False
        
        # 检查用户角色中的权限
        user_permissions = set()
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role:
                user_permissions.update(role.permissions)
        
        return permission in user_permissions
    
    def get_user_permissions(self, username: str) -> Set[str]:
        """获取用户所有权限"""
        user = self.users.get(username)
        if not user or not user.is_active:
            return set()
        
        permissions = set()
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role:
                permissions.update(role.permissions)
        
        return permissions
    
    def audit_log(
        self,
        username: str,
        action: str,
        resource: str,
        success: bool
    ):
        """记录审计日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "action": action,
            "resource": resource,
            "success": success
        }
        
        # 写入审计日志文件
        log_file = self.config_dir / "audit.log"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    async def cleanup_sessions(self):
        """清理过期会话"""
        current_time = time.time()
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if session.get("expires_at", 0) < current_time
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            self._save_data()
