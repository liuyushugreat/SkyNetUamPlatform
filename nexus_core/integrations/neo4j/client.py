"""
Neo4j 客户端封装，用于 Python 部分的图数据库操作
"""
from typing import Optional, Dict, List, Any
from neo4j import GraphDatabase, Driver, Session


class Neo4jClient:
    """Neo4j 图数据库客户端"""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        初始化 Neo4j 客户端
        
        Args:
            uri: Neo4j 连接 URI，默认从环境变量 NEO4J_URI 读取，或使用 bolt://localhost:7687
            user: 用户名，默认从环境变量 NEO4J_USER 读取，或使用 neo4j
            password: 密码，默认从环境变量 NEO4J_PASSWORD 读取，或使用 skynet123
        """
        import os
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'skynet123')
        self.driver: Optional[Driver] = None
        
    def connect(self) -> None:
        """建立数据库连接"""
        if self.driver is None:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 验证连接
            self.driver.verify_connectivity()
            print(f"[Neo4j] Connected to {self.uri}")
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            self.driver = None
            print("[Neo4j] Connection closed")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行查询并返回结果
        
        Args:
            query: Cypher 查询语句
            parameters: 查询参数
            
        Returns:
            查询结果列表
        """
        if self.driver is None:
            self.connect()
        
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行写入事务
        
        Args:
            query: Cypher 查询语句
            parameters: 查询参数
            
        Returns:
            查询结果列表
        """
        if self.driver is None:
            self.connect()
        
        with self.driver.session() as session:
            result = session.write_transaction(
                lambda tx: tx.run(query, parameters or {})
            )
            return [record.data() for record in result]
    
    def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        创建节点
        
        Args:
            label: 节点标签
            properties: 节点属性
            node_id: 节点 ID（可选，用于唯一标识）
            
        Returns:
            创建的节点信息
        """
        if node_id:
            query = f"MERGE (n:{label} {{id: $id}}) SET n += $props RETURN n"
            params = {"id": node_id, "props": properties}
        else:
            query = f"CREATE (n:{label} $props) RETURN n"
            params = {"props": properties}
        
        result = self.execute_write(query, params)
        return result[0] if result else {}
    
    def create_relationship(
        self,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建关系
        
        Args:
            from_label: 起始节点标签
            from_id: 起始节点 ID
            to_label: 目标节点标签
            to_id: 目标节点 ID
            rel_type: 关系类型
            properties: 关系属性（可选）
            
        Returns:
            创建的关系信息
        """
        props_str = ", r += $props" if properties else ""
        query = (
            f"MATCH (a:{from_label} {{id: $from_id}}), "
            f"(b:{to_label} {{id: $to_id}}) "
            f"MERGE (a)-[r:{rel_type}]{props_str}->(b) "
            f"RETURN r"
        )
        params = {
            "from_id": from_id,
            "to_id": to_id,
            "props": properties or {}
        }
        
        result = self.execute_write(query, params)
        return result[0] if result else {}


# 全局客户端实例（可选）
_global_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """获取全局 Neo4j 客户端实例"""
    global _global_client
    if _global_client is None:
        _global_client = Neo4jClient()
        _global_client.connect()
    return _global_client

