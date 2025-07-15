import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFLoader

import config

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self):
        """初始化，配置嵌入模型。"""
        self.embeddings = DashScopeEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            dashscope_api_key=config.DASHSCOPE_API_KEY
        )
        self.db = None

    def _load_all_pdfs(self):
        """从data文件夹加载所有PDF文件。"""
        if not os.path.exists(config.DATA_PATH):
            logger.error(f"数据文件夹不存在: {config.DATA_PATH}")
            return []

        pdf_files = [f for f in os.listdir(config.DATA_PATH) if f.endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"在 {config.DATA_PATH} 中没有找到任何PDF文件。")
            return []

        logger.info(f"找到 {len(pdf_files)} 个PDF文件，开始加载...")
        all_docs = []
        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(config.DATA_PATH, pdf_file)
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                for doc in documents:
                    doc.metadata['source'] = pdf_file
                all_docs.extend(documents)
                logger.info(f"已加载: {pdf_file}")
            except Exception as e:
                logger.error(f"加载或处理文件 {pdf_file} 时出错: {e}")
        return all_docs

    def create_db(self, recreate=False):
        """
        创建并持久化一个新的Chroma向量数据库。
        :param recreate: 如果为True，则强制重新创建数据库。
        """
        if not recreate and os.path.exists(config.VECTOR_DB_PATH):
            logger.info(f"数据库目录 '{config.VECTOR_DB_PATH}' 已存在。如需重建，请设置 recreate=True。")
            return

        logger.info("开始创建新的向量数据库...")
        docs = self._load_all_pdfs()
        if not docs:
            logger.error("未能加载任何文档，数据库创建中止。")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)

        logger.info(f"所有文本被分割成 {len(splits)} 个块。正在生成嵌入并存入Chroma...")
        vector_db = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=config.VECTOR_DB_PATH,
            collection_metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"向量数据库创建完成！共加载 {len(splits)} 条文档片段。")

    def load_db(self):
        """从磁盘加载已存在的Chroma数据库。"""
        if self.db:
            return self.db
        
        if not os.path.exists(config.VECTOR_DB_PATH):
            raise FileNotFoundError(f"向量数据库路径不存在: {config.VECTOR_DB_PATH}。请先运行 create_database.py。")
        
        logger.info("正在从磁盘加载向量数据库...")
        self.db = Chroma(
            persist_directory=config.VECTOR_DB_PATH,
            embedding_function=self.embeddings
        )
        logger.info("向量数据库加载成功。")
        return self.db

    def get_retriever(self, score_threshold=None):
        """
        获取一个配置好的检索器。
        """
        db = self.load_db()
        
        if score_threshold is None:
            score_threshold = config.MIN_RELEVANCE_SCORE
            
        return db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": config.RETRIEVE_K,
                "score_threshold": score_threshold
            }
        )

    def retrieve(self, query: str) -> list:
        """
        执行一个安全的检索操作。
        首先尝试用较高的相关性阈值，如果无结果，则使用配置的默认阈值。
        """
        logger.info(f"正在为查询进行检索: '{query}'")
        try:
            high_threshold_retriever = self.get_retriever(score_threshold=0.7)
            results = high_threshold_retriever.invoke(query)
            
            if not results:
                logger.warning("在高相关性(0.7)下未找到结果，尝试使用默认阈值...")
                default_retriever = self.get_retriever()
                results = default_retriever.invoke(query)
            
            logger.info(f"检索到 {len(results)} 个相关文档块。")
            return results
        
        except Exception as e:
            logger.error(f"检索过程中发生错误: {e}")
            return []