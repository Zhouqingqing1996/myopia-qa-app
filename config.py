# 大模型与嵌入模型API配置 (阿里巴巴灵积 DashScope)
# 请确保您的API Key是有效的
DASHSCOPE_API_KEY = "sk-0adb2a58b9184e218d8e293e1fd350e7" # 请替换为您自己的DashScope API Key

# 用于生成答案的LLM模型名称
LLM_MODEL_NAME = "qwen-turbo"
# 用于文本嵌入的模型名称
EMBEDDING_MODEL_NAME = "text-embedding-v2"

# Redis Cache (可选, 如果不使用可以忽略)
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

# 项目路径配置
DATA_PATH = "data"  # 存放PDF文献的文件夹
VECTOR_DB_PATH = "faiss_index_v2"

# 文本处理和检索参数
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVE_K = 5  # 检索时返回的最相关文档数量
MIN_RELEVANCE_SCORE = 0.5  # 默认的最低相关性分数阈值