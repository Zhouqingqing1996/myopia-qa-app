import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
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
        # ... 这部分函数内容保持不变 ...
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
            except Exception as e:
                logger.error(f"加载或处理文件 {pdf_file} 时出错: {e}")
        return all_docs


    def create_db(self, recreate=False):
        """
        创建并持久化一个新的FAISS向量数据库。
        """
        if not recreate and os.path.exists(config.VECTOR_DB_PATH):
            logger.info(f"数据库目录 '{config.VECTOR_DB_PATH}' 已存在。如需重建，请设置 recreate=True。")
            return

        logger.info("开始创建新的FAISS向量数据库...")
        docs = self._load_all_pdfs()
        if not docs:
            logger.error("未能加载任何文档，数据库创建中止。")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)

        logger.info(f"所有文本被分割成 {len(splits)} 个块。正在生成嵌入并存入FAISS...")
        # 使用FAISS创建数据库
        vector_db = FAISS.from_documents(splits, self.embeddings)
        # 保存到本地
        vector_db.save_local(config.VECTOR_DB_PATH)
        logger.info(f"FAISS向量数据库创建完成！")

    def load_db(self):
        """从磁盘加载已存在的FAISS数据库。"""
        if self.db:
            return self.db
        
        if not os.path.exists(config.VECTOR_DB_PATH):
            raise FileNotFoundError(f"向量数据库路径不存在: {config.VECTOR_DB_PATH}。请先运行 create_database.py。")
        
        logger.info("正在从磁盘加载FAISS向量数据库...")
        # 从本地加载FAISS数据库
        # allow_dangerous_deserialization=True 是FAISS加载所必需的
        self.db = FAISS.load_local(
            config.VECTOR_DB_PATH, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS向量数据库加载成功。")
        return self.db

    def retrieve(self, query: str) -> list:
        """
        执行一个检索操作。
        """
        logger.info(f"正在为查询进行检索: '{query}'")
        try:
            db = self.load_db()
            # FAISS 使用 similarity_search 来检索
            results = db.similarity_search(query, k=config.RETRIEVE_K)
            logger.info(f"检索到 {len(results)} 个相关文档块。")
            return results
        
        except Exception as e:
            logger.error(f"检索过程中发生错误: {e}")
            return []