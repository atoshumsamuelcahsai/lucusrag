from __future__ import annotations

import logging
import random
import time
import typing as t

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

from rag.schemas import CodeElement
from rag.schemas.vector_config import Neo4jConfig, VectorIndexConfig

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def get_vector_index_config() -> VectorIndexConfig:
    return VectorIndexConfig.from_env()


class GraphDBManager:
    """Manager for Neo4j database operations."""

    def __init__(
        self,
        config: t.Optional[Neo4jConfig] = None,
        max_retries: int = 3,
        delay: float = 1.0,
    ):
        """Initialize with optional config."""
        self.config = config or Neo4jConfig()
        self._driver: t.Optional[Driver] = None
        self._max_retries = max_retries
        self._delay = delay

    @property
    def driver(self) -> Driver:
        """Lazy init Neo4j driver and validate connectivity."""
        if self._driver is None:
            for attempt in range(self._max_retries):
                try:
                    self._driver = GraphDatabase.driver(
                        self.config.url,
                        auth=(self.config.user, self.config.password),
                        max_connection_lifetime=30 * 60,
                    )
                    self._driver.verify_connectivity()
                    logger.info(f"Connected to Neo4j at {self.config.url}")
                    break
                except Exception as e:
                    if attempt == self._max_retries - 1:
                        raise ConnectionError(f"Failed to connect: {e}")
                    delay = self._delay * (2 ** (attempt))
                    time.sleep(delay + random.random())
                    logger.warning(
                        f"Retrying Neo4j connection ({attempt+1}/{self._max_retries})..."
                    )
        return self._driver

    def close(self) -> None:
        """Close the db connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    # ---------- Schema ----------
    def create_schema(self, vector_config: VectorIndexConfig) -> None:
        """Create or verify Neo4j schema with indexes and constraints."""
        try:
            with self.driver.session() as session:
                # Create base constraints and indexes
                statements = [
                    # Unique ID constraint
                    f"""
                    CREATE CONSTRAINT IF NOT EXISTS
                    FOR (n:{vector_config.node_label}) REQUIRE n.id IS UNIQUE
                    """,
                    # Name index
                    f"""
                    CREATE INDEX IF NOT EXISTS
                    FOR (n:{vector_config.node_label})
                    ON (n.name)
                    """,
                    # Type index
                    f"""
                    CREATE INDEX IF NOT EXISTS
                    FOR (n:{vector_config.node_label})
                    ON (n.type)
                    """,
                    # Normalized path index
                    f"""
                    CREATE INDEX IF NOT EXISTS
                    FOR (n:{vector_config.node_label})
                    ON (n.normalized_path)
                    """,
                    # Qualified name index
                    f"""
                    CREATE INDEX IF NOT EXISTS
                    FOR (n:{vector_config.node_label})
                    ON (n.qualified_name)
                    """,
                ]

                # Apply base constraints and indexes
                for stmt in statements:
                    session.run(stmt)

                self._create_vector_index(session, vector_config, overwrite=True)
                logger.info(
                    f"Neo4j schema ensured for label {vector_config.node_label}"
                )
        except Exception as e:
            logger.exception(f"Failed to create schema: {str(e)}")
            raise

    def _index_exist(self, session, config: VectorIndexConfig) -> bool:
        # Check if index exists
        record = session.run(
            """
                SHOW INDEXES
                YIELD name
                WHERE name = $index_name
                RETURN count(*) > 0 as exists
                """,
            {"index_name": config.name},
        ).single()
        return bool(record and record.get("exists", False))

    def _drop_index(self, session, config: VectorIndexConfig) -> None:
        # Neo4j doesn't parameterize identifiers in DROP INDEX;
        # use f-string carefully
        session.run(f"DROP INDEX {config.name} IF EXISTS")
        logger.info(f"Dropped existing vector index {config.name}")

    def _create_index(self, session, config: VectorIndexConfig) -> None:
        session.run(
            """
                CALL db.index.vector.createNodeIndex(
                    $index_name, $node_label, $vector_property, $dimension, $similarity_metric
                )
                """,
            {
                "index_name": config.name,
                "node_label": config.node_label,
                "vector_property": config.vector_property,
                "dimension": config.dimension,
                "similarity_metric": config.similarity_metric,
            },
        )
        logger.info(
            f"Created vector index {config.name} (dim={config.dimension}, sim={config.similarity_metric})"
        )

    def _create_vector_index(
        self, session, config: VectorIndexConfig, overwrite: bool = True
    ) -> None:
        """Create or recreate a vector index with given configuration. This could be very expensive"""
        exists = self._index_exist(session, config)

        if exists:
            if overwrite:
                self._drop_index(session, config)
                self._create_index(session, config)
            else:
                logger.info("Vector index %s already exists; skipping", config.name)
        else:
            self._create_index(session, config)

    # ---------- Domain upserts ----------
    def create_nodes(
        self,
        code_info: CodeElement,
        vector_config: VectorIndexConfig,
        repo_name: str = "lucus",
    ) -> None:
        """Create/update a code element node with a stable ID."""
        try:
            element_id = f"{code_info.file_path}:{code_info.type}:{code_info.name}"
            with self.driver.session() as session:
                session.run(
                    f"""
                    MERGE (n:{vector_config.node_label} {{id: $id}})
                    SET n.name = $name,
                        n.type = $type,
                        n.file_path = $file_path,
                        n.updated_at = timestamp(),
                        n.qualified_name =
                            CASE
                                WHEN $file_path CONTAINS ('/' + $repo + '/')
                                THEN replace(
                                    substring(
                                        $file_path,
                                        size('/' + $repo + '/') + size(split($file_path, '/' + $repo + '/')[0]),
                                        size($file_path) - 3
                                    ),
                                    '/',
                                    '.'
                                ) + '.' + $name
                                ELSE $name
                            END
                    """,
                    {
                        "id": element_id,
                        "name": code_info.name,
                        "type": code_info.type,
                        "file_path": code_info.file_path,
                        "repo": repo_name,
                    },
                )
            logger.debug("Upserted node for %s", code_info.name)
        except Exception:
            logger.exception("Failed to create node for %s", code_info.name)
            raise

    def _add_classes(
        self,
        session,
        vector_config: VectorIndexConfig,
        code_info: CodeElement,
        element_id: str,
    ) -> None:
        for method in code_info.methods:
            method_id = f"{code_info.file_path}:function:{method}"
            session.run(
                f"""
                            MATCH (class:{vector_config.node_label} {{id: $class_id}})
                            MATCH (method:{vector_config.node_label} {{id: $method_id}})
                            MERGE (class)-[:HAS_METHOD {{context: 'Class method'}}]->(method)
                        """,
                {"class_id": element_id, "method_id": method_id},
            )

    def _add_inheritance(
        self,
        session,
        vector_config: VectorIndexConfig,
        code_info: CodeElement,
        element_id: str,
    ) -> None:
        for base in code_info.base_classes:
            session.run(
                f"""
                            MATCH (derived:{vector_config.node_label} {{id: $derived_id}})
                            MATCH (base:{vector_config.node_label} {{name: $base_name}})
                            MERGE (derived)-[:INHERITS_FROM {{context: 'Class inheritance'}}]->(base)
                        """,
                {"derived_id": element_id, "base_name": base},
            )

    def _add_calls(
        self,
        session,
        vector_config: VectorIndexConfig,
        code_info: CodeElement,
        element_id: str,
    ) -> None:
        for call in code_info.calls:
            call_name = call.split(".")[-1]
            session.run(
                f"""
                            MATCH (caller:{vector_config.node_label} {{id: $caller_id}})
                            MATCH (called:{vector_config.node_label} {{name: $call_name}})
                            MERGE (caller)-[:CALLS {{
                                context: 'Function call',
                                call_signature: $full_call
                            }}]->(called)
                        """,
                {"caller_id": element_id, "call_name": call_name, "full_call": call},
            )

    def _add_dependencies(
        self,
        session,
        vector_config: VectorIndexConfig,
        code_info: CodeElement,
        element_id: str,
    ) -> None:
        for dep in code_info.dependencies:
            dep_name = dep.split(".")[-1]
            session.run(
                f"""
                            MATCH (source:{vector_config.node_label} {{id: $source_id}})
                            MATCH (dep:{vector_config.node_label} {{name: $dep_name}})
                            MERGE (source)-[:DEPENDS_ON {{
                                context: 'Module dependency',
                                import_path: $full_dep
                            }}]->(dep)
                        """,
                {"source_id": element_id, "dep_name": dep_name, "full_dep": dep},
            )

    def create_relationships(
        self, code_info: CodeElement, vector_config: VectorIndexConfig
    ) -> None:
        """Create relationships between nodes."""
        try:
            with self.driver.session() as session:
                element_id = f"{code_info.file_path}:{code_info.type}:{code_info.name}"

                # 1. Add class methods
                if code_info.type == "class" and code_info.methods:
                    self._add_classes(session, vector_config, code_info, element_id)

                # 2. Add inheritance
                if code_info.base_classes:  # Direct attribute access
                    self._add_inheritance(session, vector_config, code_info, element_id)

                # 3. Add calls
                if code_info.calls:  # Direct attribute access
                    self._add_calls(session, vector_config, code_info, element_id)

                # 4. Add dependencies
                if code_info.dependencies:  # Direct attribute access
                    self._add_dependencies(
                        session, vector_config, code_info, element_id
                    )

            logger.info(f"Added relationships for {code_info.name}")

        except Exception as e:
            logger.exception(
                f"Failed to add relationships for {code_info.name}: {str(e)}"
            )
            raise
