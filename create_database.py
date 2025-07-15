import logging
from vectordb_manager import VectorDBManager

if __name__ == "__main__":
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("开始执行数据库创建流程...")
    
    # 实例化管理器
    manager = VectorDBManager()
    
    # 创建数据库。设置 recreate=True 会删除并重建现有数据库
    # 在第一次运行时，或者当你的PDF文档更新后，使用 recreate=True
    manager.create_db(recreate=True)
    
    logger.info("数据库创建流程执行完毕！")