from typing import Dict, Optional, Any, ClassVar
from pathlib import Path
import time

from loguru import logger
from pydantic import BaseModel, Field


class SystemPromptInfo(BaseModel):
    """
    系统提示词信息类
    """
    name: str = Field(..., description="系统提示词名称")
    file_path: str | Path = Field(..., description="系统提示词文件路径")
    content: str = Field(default="", description="系统提示词内容")
    last_modified: float = Field(default=0.0, description="最后修改时间戳")
    loaded_at: float = Field(default=0.0, description="加载时间戳")


class SystemPromptManager(BaseModel):
    """
    系统提示词管理器
    负责管理多个系统提示词，支持文件I/O读取和内存缓存
    """

    # 默认的系统提示词路径
    system_prompt_path: ClassVar[Path] = Path(__file__).parent / "system.md"

    prompts: Dict[str, SystemPromptInfo] = Field(
        default_factory=dict, 
        description="存储系统提示词信息的字典"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        # 默认初始化加载 system.md 文件
        system_md_path = self.system_prompt_path
        if system_md_path.exists():
            self.add_prompt("system", str(system_md_path))
    
    def get_prompt(self, name: str, file_path: Optional[str] = None) -> str:
        """
        获取系统提示词内容
        
        Args:
            name: 提示词名称
            file_path: 提示词文件路径，如果为None则使用已存储的路径
            
        Returns:
            系统提示词内容
        """
        # 检查是否已经存在该提示词
        if name in self.prompts:
            prompt_info = self.prompts[name]
            
            # 如果传入了新的文件路径，更新路径
            if file_path is not None:
                prompt_info.file_path = file_path
                
            # 检查文件是否已更新
            if self._is_file_updated(prompt_info):
                # 重新加载文件内容
                self._load_prompt_content(prompt_info)
        else:
            # 如果是新的提示词，创建新的SystemPromptInfo
            if file_path is None:
                # 如果不存在就默认为使用 system.md 文件
                file_path = self.system_prompt_path
                logger.error(f'提示词名称不存在, 使用系统默认的提示词, 默认的提示词路径为: {file_path}')

            prompt_info = SystemPromptInfo(name=name, file_path=file_path)
            self.prompts[name] = prompt_info
            self._load_prompt_content(prompt_info)
        
        return self.prompts[name].content
    
    def _is_file_updated(self, prompt_info: SystemPromptInfo) -> bool:
        """
        检查文件是否已更新
        
        Args:
            prompt_info: 系统提示词信息
            
        Returns:
            如果文件已更新返回True，否则返回False
        """
        file_path = Path(prompt_info.file_path)
        
        if not file_path.exists():
            return False
        
        current_modified = file_path.stat().st_mtime
        
        # 比较当前修改时间与记录的修改时间
        return current_modified > prompt_info.last_modified
    
    def _load_prompt_content(self, prompt_info: SystemPromptInfo) -> None:
        """
        从文件加载提示词内容
        
        Args:
            prompt_info: 系统提示词信息
        """
        file_path = Path(prompt_info.file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"系统提示词文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_info.content = f.read()
        
        # 更新修改时间和加载时间
        prompt_info.last_modified = file_path.stat().st_mtime
        prompt_info.loaded_at = time.time()
    
    def add_prompt(self, name: str, file_path: str) -> None:
        """
        添加一个新的系统提示词
        
        Args:
            name: 提示词名称
            file_path: 提示词文件路径
        """
        if name in self.prompts:
            raise ValueError(f"提示词名称 '{name}' 已存在")
        
        prompt_info = SystemPromptInfo(name=name, file_path=file_path)
        self.prompts[name] = prompt_info
        self._load_prompt_content(prompt_info)
    
    def update_prompt_path(self, name: str, new_file_path: str) -> None:
        """
        更新现有提示词的文件路径
        
        Args:
            name: 提示词名称
            new_file_path: 新的文件路径
        """
        if name not in self.prompts:
            raise ValueError(f"提示词名称 '{name}' 不存在")
        
        self.prompts[name].file_path = new_file_path
        # 重新加载新路径的文件内容
        self._load_prompt_content(self.prompts[name])
    
    def remove_prompt(self, name: str) -> None:
        """
        移除指定的系统提示词
        
        Args:
            name: 要移除的提示词名称
        """
        if name in self.prompts:
            del self.prompts[name]
    
    def list_prompts(self) -> list:
        """
        列出所有系统提示词的名称
        
        Returns:
            所有提示词名称的列表
        """
        return list(self.prompts.keys())
    
    def get_prompt_info(self, name: str) -> Optional[SystemPromptInfo]:
        """
        获取指定提示词的详细信息
        
        Args:
            name: 提示词名称
            
        Returns:
            系统提示词信息，如果不存在则返回None
        """
        return self.prompts.get(name)