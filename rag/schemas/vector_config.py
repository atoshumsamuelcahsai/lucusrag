from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Neo4jConfig:
    """Connection configuration for Neo4j."""

    url: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")


@dataclass(frozen=True, slots=True)
class VectorIndexConfig:
    """Configuration for vector index storage in Neo4j."""

    name: str
    dimension: int
    node_label: str
    vector_property: str
    similarity_metric: str
    neo4j_url: str
    neo4j_user: str
    neo4j_password: str

    @classmethod
    def from_env(cls) -> "VectorIndexConfig":
        """Construct configuration from environment variables."""
        return cls(
            name=os.getenv("VECTOR_INDEX_NAME", "james_code_embeddings"),
            dimension=int(os.getenv("VECTOR_DIMENSION", "1536")),
            node_label=os.getenv("NODE_LABEL", "JamesCodeElement"),
            vector_property=os.getenv("VECTOR_PROPERTY", "embedding"),
            similarity_metric=os.getenv("SIMILARITY_METRIC", "cosine"),
            neo4j_url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        )
