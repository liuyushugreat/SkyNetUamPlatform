"""
Backward-compatible Neo4j client import path.

Historically some scripts imported:
    from nexus_core.neo4j_client import Neo4jClient

The canonical location is now:
    from nexus_core.integrations.neo4j import Neo4jClient
"""

from nexus_core.integrations.neo4j import Neo4jClient, get_neo4j_client

__all__ = ["Neo4jClient", "get_neo4j_client"]


