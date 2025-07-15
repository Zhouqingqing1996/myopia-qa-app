import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import VECTOR_STORE_PATH, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

class VectorRetriever:
    def __init__(self):
        """
        初始化检索器，加载本地的FAISS向量数据库和嵌入模型。
        """
        self.vector_store = None
        self.embeddings_model = None
        try:
            # 加载嵌入模型
            logger.info(f"正在加载嵌入模型: {EMBEDDING_MODEL_NAME}...")
            self.embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            logger.info("嵌入模型加载成功。")

            # 加载FAISS索引
            logger.info(f"正在从 {VECTOR_STORE_PATH} 加载FAISS索引...")
            self.vector_store = FAISS.load_local(VECTOR_STORE_PATH, self.embeddings_model, allow_dangerous_deserialization=True)
            logger.info("FAISS索引加载成功。")

        except Exception as e:
            logger.error(f"初始化VectorRetriever失败: {e}")
            raise

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        根据用户问题，在向量数据库中进行相似度搜索。

        :param query: 用户的问题字符串。
        :param top_k: 返回最相关的文档块数量。
        :return: 一个包含相关文档块 (Document对象) 的列表。
        """
        if not self.vector_store:
            logger.error("向量数据库未初始化，无法进行检索。")
            return []
        try:
            logger.info(f"正在为问题进行检索: '{query}'")
            # asearch: asynchronous similarity search
            relevant_docs = self.vector_store.similarity_search(query, k=top_k)
            logger.info(f"检索到 {len(relevant_docs)} 个相关文档块。")
            return relevant_docs
        except Exception as e:
            logger.error(f"检索过程中发生错误: {e}")
            return []