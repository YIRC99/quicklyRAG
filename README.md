
这是我的项目名字目前叫做 QuicklyRag
初心就是能够简单的修改一下配置文件, 就能一行代码接入Rag
虽然目前市场上存在很多Rag框架, 比如coze, dify, aiflow, ragflow等等
虽然这些框架都功能强大 但是不可避免的都有一些缺点
1. 前后端一体而的大而完整的项目
2. 内存占用大
3. 成品框架, 对开发者而言, 二次开发很难
4. 对于简单的智能对话需求, 配置运行往往太过于麻烦

而`QuicklyRag`就是专门解决这些缺点的解决方案
`QuicklyRag`采用类似`HuTool`的思路, 只需要在任意的`python`项目用, 引入`QuicklyRag`的文件夹
再去配置好自己的`api_key`即可在导入`QuicklyRagAPI`快速调用ai的能力
改项目的优点如下
1. 二次开发简单, 所有的功能基于工具类的思想封装, 入门小白都可以快速调用
2. 项目整体采用`pydantic`的开发, 类型安全, 调用都提供类型提示
3. 随意接入, 因为本质就是一个文件夹, 只需将文件复制粘贴到项目里面即可
4. 根据平台不同, 可以自由选择不同的向量存储, 搜索引擎, 聊天模型, 无需修改代码

## 框架使用方法:
### 1. 下载项目到本地
```angular2html
git clone https://github.com/zhangyuanyuan/QuicklyRag.git
```
### 2. 将项目中的quickly文件夹复制到自己的项目中

### 3. 下载项目依赖
```angular2html
D:\CodeFile\python-project\quicklyRag
├── main.py  ----> 项目使用实例文件
├── pyproject.toml  ---> 如果用的uv 就复制这个到自己的项目中然后使用uv sync
├── quicklyRag  ---> 将这个文件夹复制到自己的项目中
├── README.md
├── requirements.txt   ---> 如果没用uv, 那就用 pip install -r requirements.txt
├── uv.lock   ---> 如果用的uv 下载的时候这个也要复制到自己的项目中

```
### 4. 配置自己的平台key
配置文件在`QuicklyRag/config`文件夹下，根据自己的平台配置对应的key
config中的所有的文件都可以查看一下, 除了必填的其他都可以选填

### 5. 引入QuicklyRagAPI使用
在`QuicklyRag`文件夹下有`QuicklyRagAPI.py`, 里面封装了所有功能
可以查看里面的方法使用都有注释
```angular2html
from quicklyRag.QuicklyRagAPI import * # 导入即可使用

chat = QuicklyRagAPI.llm_chat("hello world")
```

框架结构如下:
```angular2html
│  chat.db                      ---> 聊天记录数据库
│  QuicklyRagAPI.py             ---> 对外提供API
├─baseClass                     ---> 基类
│  │  documentBase.py
│  │  PlatformBase.py
│  │  searchBase.py
│  │  VectorBase.py
│
├─baseEnum                      ---> 基本枚举
│  │  PlatformEnum.py
│  │  VectorEnum.py
│
├─chat                          ---> 聊天模块
│  ├─chatRequest                ---> 聊天请求处理模块
│  │  │  chatRequestHandler.py
│  │
│  ├─message                    ---> 聊天消息模块
│  │  │  chatMessage.py
│  │  │  chatMessageManager.py
│  │  │  ChatSessionManager.py
│  │
│  └─prompt                     ---> 提示词模块
│      │  system.md
│      │  SystemPromptManager.py
│
├─config                        ---> 项目配置文件
│  │  ChatConfig.py
│  │  DocumentConfig.py
│  │  PlatformConfig.py
│  │  VectorConfig.py
│
 ├─document                      ---> 文档处理模块
│  │  loadDocument.py
│  │  spliterDocument.py
│  │  testmd.md
│
 ├─provider                      ---> 服务提供者模块
│  │  QuicklyChatModelProvider.py
│  │  QuicklyEmbeddingModelProvider.py
│  │  QuicklyRerankerProvider.py
│  │  QuicklyVectorStoreProvider.py
│
├─vector                         ---> 向量处理模块
│  ├─embedding                   ---> 向量嵌入模块
│  │  │
│  │  │  vectorEmbedding.py
│  │
│  └─store                       ---> 向量存储模块
│      │  milvusUtil.py
│      │  vectorStore.py
```