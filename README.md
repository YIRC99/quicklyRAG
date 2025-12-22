```angular2html
│  chat.db                      ---> 聊天记录数据库
│  QuicklyRagAPI.py             ---> 门面模式 对外提供API
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