import os
import subprocess

# --- 核心修改：动态生成数据库路径 ---
def get_git_commit_hash():
    """获取当前Git提交的短哈希值，作为数据库的唯一标识。"""
    try:
        # 在Streamlit Cloud环境中，项目是以Git仓库形式存在的
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except Exception:
        # 如果在没有git的环境中运行，则返回一个通用名称
        return "local"

# 使用Git提交哈希值来创建一个独一无二的数据库路径
# 这样每次推送新代码，都会生成一个新的路径，强制重新创建数据库
UNIQUE_ID = get_git_commit_hash()
VECTOR_DB_PATH = f"faiss_index_{UNIQUE_ID}"
# ------------------------------------

# 大模型与嵌入模型API配置 (阿里巴巴灵积 DashScope)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") 

# 用于生成答案的LLM模型名称
LLM_MODEL_NAME = "qwen-turbo-latest"
# 用于文本嵌入的模型名称
EMBEDDING_MODEL_NAME = "text-embedding-v2"

# Redis Cache (可选, 如果不使用可以忽略)
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

# 项目路径配置
DATA_PATH = "data"

# 文本处理和检索参数
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVE_K = 5
