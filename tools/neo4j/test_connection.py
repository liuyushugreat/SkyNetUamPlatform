"""
测试 Neo4j 连接脚本
"""
import sys
from pathlib import Path

# Add repo root to sys.path so we can import `nexus_core.*` as a package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nexus_core.integrations.neo4j import Neo4jClient


def test_connection():
    """测试 Neo4j 连接"""
    print("正在测试 Neo4j 连接...")
    
    try:
        with Neo4jClient() as client:
            # 测试基本查询
            result = client.execute_query("RETURN 1 as test")
            print(f"OK: 连接成功。测试查询结果: {result}")
            
            # 创建测试节点
            test_node = client.create_node(
                label='TestNode',
                properties={'name': 'Test', 'created_at': '2025-12-30'},
                node_id='test-001'
            )
            print(f"OK: 测试节点创建成功: {test_node}")
            
            # 查询测试节点
            nodes = client.execute_query(
                "MATCH (n:TestNode) RETURN n LIMIT 5"
            )
            print(f"OK: 查询到 {len(nodes)} 个测试节点")
            
            # 清理测试数据
            client.execute_write("MATCH (n:TestNode) DELETE n")
            print("OK: 测试数据已清理")
            
            print("\nOK: Neo4j 连接测试通过！")
            return True
            
    except Exception as e:
        # Avoid unicode symbols for Windows GBK consoles.
        print(f"ERROR: 连接失败: {e}")
        print("\n请检查：")
        print("1. Docker 容器是否运行: docker ps")
        print("2. Neo4j 是否已启动: docker compose -f infra/docker-compose.yml up -d neo4j")
        print("3. 连接信息是否正确（URI, 用户名, 密码）")
        return False


if __name__ == '__main__':
    success = test_connection()
    sys.exit(0 if success else 1)

