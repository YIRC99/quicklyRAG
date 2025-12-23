# 对话记忆数据默认保存的文件位置 默认为当前项目下
from pathlib import Path

# 默认sqlite数据存储路径
default_session_db_path = Path(__file__).parent.parent.parent / 'chat.db'



