# 对话记忆数据默认保存的文件位置 默认为当前项目下
from pathlib import Path

# 获取工具类文件的根目录（向上两级目录）
default_session_db_path = Path(__file__).parent.parent / 'chat.db'


if __name__ == '__main__':
    print(default_session_db_path)

