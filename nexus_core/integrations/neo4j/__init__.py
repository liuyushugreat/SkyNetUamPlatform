"""
Neo4j integration for nexus_core (Python).

Canonical import:
    from nexus_core.integrations.neo4j import Neo4jClient
"""

from .client import Neo4jClient, get_neo4j_client

__all__ = ["Neo4jClient", "get_neo4j_client"]


