import redis
import logging

logger = logging.getLogger(__name__)

class QueryCache:
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            # 增加连接超时设置
            self.redis_client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=2)
            self.redis_client.ping()
            logger.info(f"成功连接到 Redis at {host}:{port}/{db}")
        except redis.exceptions.ConnectionError as e:
            logger.warning(f"无法连接到 Redis at {host}:{port}/{db}。缓存功能将被禁用。错误: {e}")
            self.redis_client = None