"""
AIOS API接口
提供RESTful API访问AIOS功能
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import os
from pathlib import Path

from ..kernel import AIKernel
from ..filesystem import AIFS
from ..process import SmartScheduler

# API模型定义
class SystemStatus(BaseModel):
    """系统状态模型"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    process_count: int
    
class ProcessInfo(BaseModel):
    """进程信息模型"""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_percent: float
    
class FileInfo(BaseModel):
    """文件信息模型"""
    name: str
    type: str
    size: int
    modified: str
    
class CommandRequest(BaseModel):
    """命令请求模型"""
    command: str
    args: List[str]
    
class CommandResponse(BaseModel):
    """命令响应模型"""
    success: bool
    output: str
    error: Optional[str] = None

# 创建FastAPI应用实例
app = FastAPI(
    title="AIOS API",
    description="AI Operating System REST API",
    version="1.0.0"
)

# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置认证
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 初始化核心组件
kernel = AIKernel()
fs = AIFS(root_path=str(Path.home() / "aios_root"))  # 使用用户目录下的 aios_root 作为根目录
scheduler = SmartScheduler()

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """用户登录"""
    # 简单的用户验证
    if form_data.username == "admin" and form_data.password == "admin":
        return {"access_token": "admin", "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.get("/status", response_model=SystemStatus)
async def get_status(token: str = Depends(oauth2_scheme)):
    """获取系统状态"""
    try:
        status = kernel.get_system_status()
        return SystemStatus(**status)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/processes", response_model=List[ProcessInfo])
async def list_processes(token: str = Depends(oauth2_scheme)):
    """列出所有进程"""
    try:
        processes = scheduler.list_processes()
        return [ProcessInfo(**proc) for proc in processes]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/processes/{pid}")
async def kill_process(
    pid: int,
    token: str = Depends(oauth2_scheme)
):
    """终止进程"""
    try:
        result = scheduler.terminate_process(pid)
        if result:
            return {"message": f"Process {pid} terminated"}
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Process {pid} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/fs/list/{path:path}", response_model=List[FileInfo])
async def list_directory(
    path: str,
    token: str = Depends(oauth2_scheme)
):
    """列出目录内容"""
    try:
        contents = await fs.list_directory(path)
        return [FileInfo(**item) for item in contents]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/fs/read/{path:path}")
async def read_file(
    path: str,
    token: str = Depends(oauth2_scheme)
):
    """读取文件内容"""
    try:
        content = await fs.read_file(path)
        return {"content": content}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/exec", response_model=CommandResponse)
async def execute_command(
    command: CommandRequest,
    token: str = Depends(oauth2_scheme)
):
    """执行命令"""
    try:
        output = await scheduler.execute_command(
            [command.command] + command.args
        )
        return CommandResponse(
            success=True,
            output=output
        )
    except Exception as e:
        return CommandResponse(
            success=False,
            error=str(e)
        )

async def start(host: str = "0.0.0.0", port: int = 8000):
    """启动API服务器"""
    await kernel.start()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def stop():
    """停止API服务器"""
    await kernel.stop()

def main():
    """API入口点"""
    import asyncio
    asyncio.run(start())
