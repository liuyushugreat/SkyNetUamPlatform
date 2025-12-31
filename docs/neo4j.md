# Neo4j 安装和使用指南

本项目已集成 Neo4j 图数据库，用于存储和管理复杂的关联数据。

## 快速开始

### 1. 使用 Docker Compose（推荐）

最简单的方式是使用 Docker Compose 启动 Neo4j：

```bash
docker compose -f infra/docker-compose.yml up -d neo4j
```

这将启动 Neo4j 服务，默认配置：
- **HTTP 端口**: 7474 (浏览器访问: http://localhost:7474)
- **Bolt 端口**: 7687 (应用程序连接)
- **默认用户名**: neo4j
- **默认密码**: skynet123

### 2. 访问 Neo4j Browser

启动后，在浏览器中访问 http://localhost:7474，使用以下凭据登录：
- 用户名: `neo4j`
- 密码: `skynet123`

首次登录后，系统会要求您更改密码。

### 3. 配置环境变量

创建 `.env` 文件（基于 `.env.example`）并配置 Neo4j 连接信息：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=skynet123
```

## 在代码中使用

### NestJS 后端（TypeScript）

```typescript
import { Neo4jService } from './integrations/neo4j/neo4j.service';

// 在服务中注入 Neo4jService
constructor(private readonly neo4j: Neo4jService) {}

// 执行查询
const result = await this.neo4j.executeQuery(
  'MATCH (n:Drone) RETURN n LIMIT 10'
);

// 执行写入
await this.neo4j.executeWrite(
  'CREATE (d:Drone {id: $id, name: $name})',
  { id: 'drone-001', name: 'SkyNet-001' }
);
```

### Python 后端

```python
from nexus_core.integrations.neo4j import Neo4jClient

# 使用上下文管理器（推荐）
with Neo4jClient() as client:
    # 创建节点
    client.create_node(
        label='Drone',
        properties={'id': 'drone-001', 'name': 'SkyNet-001'},
        node_id='drone-001'
    )
    
    # 创建关系
    client.create_relationship(
        from_label='Drone',
        from_id='drone-001',
        to_label='Mission',
        to_id='mission-001',
        rel_type='ASSIGNED_TO',
        properties={'assigned_at': '2025-01-01T00:00:00Z'}
    )
    
    # 执行自定义查询
    result = client.execute_query(
        'MATCH (d:Drone)-[r:ASSIGNED_TO]->(m:Mission) RETURN d, r, m'
    )
```

### 连接自检（推荐）

```bash
python tools/neo4j/test_connection.py
```

## 常用 Cypher 查询示例

### 创建无人机节点
```cypher
CREATE (d:Drone {
  id: 'drone-001',
  name: 'SkyNet-001',
  status: 'active',
  operator: 'SkyHigh Ops'
})
```

### 创建任务节点并关联
```cypher
MATCH (d:Drone {id: 'drone-001'})
CREATE (m:Mission {
  id: 'mission-001',
  type: 'delivery',
  status: 'in_progress'
})
CREATE (d)-[:ASSIGNED_TO]->(m)
```

### 查询所有无人机及其任务
```cypher
MATCH (d:Drone)-[r:ASSIGNED_TO]->(m:Mission)
RETURN d, r, m
```

### 查询特定运营商的所有无人机
```cypher
MATCH (d:Drone {operator: 'SkyHigh Ops'})
RETURN d
```

## 停止 Neo4j

```bash
docker compose -f infra/docker-compose.yml down
```

如果需要同时删除数据卷：

```bash
docker compose -f infra/docker-compose.yml down -v
```

## 生产环境注意事项

1. **更改默认密码**: 在生产环境中，务必更改默认密码
2. **使用环境变量**: 不要在代码中硬编码密码
3. **配置认证**: 考虑使用 Neo4j 的企业版功能，如 LDAP 集成
4. **备份策略**: 定期备份 Neo4j 数据
5. **性能调优**: 根据数据量调整内存配置

## 故障排除

### 连接失败
- 检查 Docker 容器是否运行: `docker ps`
- 检查端口是否被占用: `netstat -an | findstr 7687`
- 检查防火墙设置

### 内存不足
在 `infra/docker-compose.yml` 中调整内存配置：
```yaml
environment:
  - NEO4J_dbms_memory_heap_max__size=4G
  - NEO4J_dbms_memory_pagecache_size=2G
```

## 更多资源

- [Neo4j 官方文档](https://neo4j.com/docs/)
- [Cypher 查询语言参考](https://neo4j.com/docs/cypher-manual/)
- [Neo4j Python 驱动文档](https://neo4j.com/docs/python-manual/)

