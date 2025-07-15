import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import DATA_PATH, VECTOR_STORE_PATH

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_vector_store():
    """
    遍历data文件夹下的所有PDF，提取文本，分块，创建嵌入向量，并存入FAISS向量数据库。
    """
    if not os.path.exists(DATA_PATH):
        logger.error(f"数据文件夹不存在: {DATA_PATH}")
        return

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"在 {DATA_PATH} 中没有找到任何PDF文件。")
        return

    logger.info(f"找到 {len(pdf_files)} 个PDF文件，开始处理...")

    all_docs = []
    for pdf_file in pdf_files:
        try:
            file_path = os.path.join(DATA_PATH, pdf_file)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            # 为每个文档块添加源文件元数据
            for doc in documents:
                doc.metadata['source'] = pdf_file
            all_docs.extend(documents)
            logger.info(f"已加载: {pdf_file}")
        except Exception as e:
            logger.error(f"加载或处理文件 {pdf_file} 时出错: {e}")

    if not all_docs:
        logger.error("未能从PDF文件中加载任何内容。")
        return

    # 文本分割
    logger.info("开始进行文本分割...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(all_docs)
    logger.info(f"所有文本被分割成 {len(chunks)} 个块。")

    # 创建嵌入 (使用一个强大的多语言模型)
    logger.info("正在初始化嵌入模型 (这可能需要下载模型)...")
    # 'paraphrase-multilingual-mpnet-base-v2' 是一个很好的多语言模型选项
    embeddings_model = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-mpnet-base-v2')
    logger.info("嵌入模型加载完毕。")

    # 创建并保存FAISS向量数据库
    logger.info("正在创建FAISS向量数据库...")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings_model)
        vector_store.save_local(VECTOR_STORE_PATH)
        logger.info(f"向量数据库已成功创建并保存至: {VECTOR_STORE_PATH}")
    except Exception as e:
        logger.error(f"创建或保存FAISS索引时出错: {e}")

if __name__ == "__main__":
    create_vector_store()