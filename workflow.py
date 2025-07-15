import logging
import hashlib
import json
from qa_engine import QAEngine
from vectordb_manager import VectorDBManager
from cache import QueryCache

logger = logging.getLogger(__name__)

class MyopiaControlWorkflow:
    def __init__(self, llm_client, retriever: VectorDBManager, cache: QueryCache = None):
        self.qa_engine = QAEngine(llm_client)
        self.retriever = retriever
        self.cache = cache
        logger.info("MyopiaControlWorkflow 初始化完成。")

    def _get_cache_key(self, question: str, language: str) -> str:
        """为缓存生成一个唯一的键。"""
        key_string = f"{question}|{language}"
        return f"myopia_qa:{hashlib.md5(key_string.encode('utf-8')).hexdigest()}"

    def run(self, question: str, language: str = "en") -> dict:
        """
        执行完整的问答流程。
        1. (可选) 检查缓存。
        2. 从向量数据库检索相关文献。
        3. 基于文献生成答案。
        4. (可选) 存储结果到缓存。
        """
        # --- 1. 检查缓存 ---
        if self.cache and self.cache.redis_client:
            cache_key = self._get_cache_key(question, language)
            cached_result = self.cache.redis_client.get(cache_key)
            if cached_result:
                logger.info(f"缓存命中: {question}")
                return json.loads(cached_result.decode('utf-8'))
            logger.info(f"缓存未命中: {question}")

        # --- 2. 知识检索 ---
        try:
            retrieved_docs = self.retriever.retrieve(question)
        except Exception as e:
            logger.error(f"工作流中的检索步骤失败: {e}")
            return {
                "question": question,
                "answer": "Error during knowledge retrieval step.",
                "formatted_context": "",
                "error": str(e)
            }

        # --- 3. 答案生成 ---
        qa_result = self.qa_engine.generate_answer(question, retrieved_docs, language)

        final_result = {
            "question": question,
            "answer": qa_result["answer"],
            "formatted_context": qa_result["formatted_context"],
            "error": qa_result["error"]
        }

        # --- 4. 设置缓存 ---
        if self.cache and self.cache.redis_client and not final_result["error"]:
            cache_key = self._get_cache_key(question, language)
            self.cache.redis_client.setex(cache_key, 3600, json.dumps(final_result)) # 缓存1小时
            logger.info(f"结果已存入缓存: {question}")
            
        logger.info(f"工作流完成: \"{question}\"")
        return final_result