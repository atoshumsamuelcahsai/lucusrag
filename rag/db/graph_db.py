from __future__ import annotations

import logging
import random
import time
import typing as t

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase, ManagedTransaction, Session

from rag.schemas import CodeElement
from rag.schemas.vector_config import Neo4jConfig, VectorIndexConfig

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


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
        assert self._driver is not None, "Driver should be initialized by now"
        return self._driver

    def close(self) -> None:
        """Close the db connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    # Schema creation
    def create_schema(self, vector_config: VectorIndexConfig) -> None:
        """Create or verify Neo4j schema with indexes and constraints."""
        try:
            with self.driver.session() as session:
                # Create base constraints and indexes
                self._create_data_schema(session, vector_config)
                self._create_vector_index(session, vector_config, overwrite=True)
                logger.info(
                    f"Neo4j schema ensured for label {vector_config.node_label}"
                )
        except Exception as e:
            logger.exception(f"Failed to create schema: {str(e)}")
            raise

    def _create_data_schema(
        self, session: Session, vector_config: VectorIndexConfig
    ) -> None:
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
        ]

        def _run_schema(tx: ManagedTransaction) -> None:
            for stmt in statements:
                tx.run(stmt)

        session.execute_write(_run_schema)
        logger.info(f"Created data schema for label {vector_config.node_label}")

    def _index_exists(self, session: Session, config: VectorIndexConfig) -> bool:
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

    def _drop_index(self, session: Session, config: VectorIndexConfig) -> None:
        session.run(f"DROP INDEX `{config.name}` IF EXISTS")
        logger.info(f"Dropped existing vector index {config.name}")

    def _create_index(self, session: Session, config: VectorIndexConfig) -> None:
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
        self, session: Session, config: VectorIndexConfig, overwrite: bool = True
    ) -> None:
        """Create or recreate a vector index with given configuration. This could be very expensive"""
        exists = self._index_exists(session, config)

        if exists:
            if overwrite:
                self._drop_index(session, config)
                self._create_index(session, config)
            else:
                logger.info("Vector index %s already exists; skipping", config.name)
        else:
            self._create_index(session, config)

    # ---------- Domain upserts ----------
    def create_node(
        self: GraphDBManager,
        code_info: CodeElement,
        vector_config: VectorIndexConfig,
    ) -> None:
        """Create or update a full CodeElement node with all metadata (except relationships)."""
        label = vector_config.node_label

        def _upsert(tx: ManagedTransaction) -> None:
            tx.run(
                f"""
              MERGE (n:{label} {{ id: $id }})
              SET n.name = $name,
                  n.type = $type,
                  n.file_path = $file_path,
                  n.docstring = $docstring,
                  n.code = $code,
                  n.parameters = $parameters,
                  n.return_type = $return_type,
                  n.decorators = $decorators,
                  n.base_classes = $base_classes,
                  n.methods = $methods,
                  n.assignments = $assignments,
                  n.explanation = $explanation,
                  n.is_async = $is_async,
                  n.updated_at = timestamp()
              """,
                {
                    "id": code_info.id,
                    "name": code_info.name,
                    "type": code_info.type,
                    "file_path": code_info.file_path,
                    "docstring": code_info.docstring,
                    "code": code_info.code,
                    "parameters": code_info.parameters,
                    "return_type": code_info.return_type,
                    "decorators": code_info.decorators,
                    "base_classes": code_info.base_classes,
                    "methods": code_info.methods,
                    "assignments": code_info.assignments,
                    "explanation": code_info.explanation or None,
                    "is_async": code_info.is_async,
                },
            )

        try:
            with self.driver.session() as session:
                session.execute_write(_upsert)
                logger.info(f"Upserted node for {code_info.name}")
        except Exception:
            logger.exception(f"Failed to create node for {code_info.name}")
            raise

    def _add_inheritance(
        self,
        session: Session,
        vector_config: VectorIndexConfig,
        code_info: CodeElement,
    ) -> None:
        if not code_info.base_classes:
            return

        label = vector_config.node_label

        def _link(tx: ManagedTransaction) -> None:
            tx.run(
                f"""
                MATCH (derived:{label} {{ id: $derived_id }})
                UNWIND $bases AS base_name
                MATCH (base:{label} {{ name: base_name }})
                MERGE (derived)-[:INHERITS_FROM {{context: 'Class inheritance'}}]->(base)
                """,
                {
                    "derived_id": code_info.id,
                    "bases": code_info.base_classes,
                },
            )

        session.execute_write(_link)

    def _add_calls(
        self, session: Session, vector_config: VectorIndexConfig, code_info: CodeElement
    ) -> None:
        if not code_info.calls:
            return
        label = vector_config.node_label

        def _link(tx: ManagedTransaction) -> None:
            tx.run(
                f"""
                MATCH (caller:{label} {{ id: $caller_id }})
                UNWIND $calls AS callee_name          // full FQN, no split
                MATCH (called:{label} {{ name: callee_name }})
                MERGE (caller)-[:CALLS {{
                    context: 'Function call',
                    call_signature: callee_name
                }}]->(called)
                """,
                {"caller_id": code_info.id, "calls": code_info.calls},
            )

        session.execute_write(_link)

    def _add_dependencies(
        self,
        session: Session,
        vector_config: VectorIndexConfig,
        code_info: CodeElement,
    ) -> None:
        if not code_info.dependencies:
            return
        label = vector_config.node_label

        def _link(tx: ManagedTransaction) -> None:
            tx.run(
                f"""
                MATCH (source:{label} {{ id: $source_id }})
                UNWIND $dependencies AS dep_name
                MATCH (dep:{label} {{ name: dep_name }})
                MERGE (source)-[:DEPENDS_ON {{
                    context: 'Module dependency',
                    import_path: dep_name
                }}]->(dep)
                """,
                {
                    "source_id": code_info.id,
                    "dependencies": code_info.dependencies,
                },
            )

        session.execute_write(_link)

    def create_relationships(
        self, code_info: CodeElement, vector_config: VectorIndexConfig
    ) -> None:
        """Create relationships between nodes."""
        try:
            with self.driver.session() as session:
                # 1. Add inheritance
                if code_info.base_classes:  # Direct attribute access
                    self._add_inheritance(session, vector_config, code_info)

                # 2. Add calls
                if code_info.calls:  # Direct attribute access
                    self._add_calls(session, vector_config, code_info)

                # 3. Add dependencies
                if code_info.dependencies:  # Direct attribute access
                    self._add_dependencies(session, vector_config, code_info)

            logger.info(f"Added relationships for {code_info.name}")

        except Exception as e:
            logger.exception(
                f"Failed to add relationships for {code_info.name}: {str(e)}"
            )
            raise

    # ----------  upsert embeddings ----------
    def upsert_embeddings(
        self,
        rows: list[dict],
        vector_config: VectorIndexConfig,
    ) -> int:
        """
        Upsert embeddings and serialized node data.

        Each row should contain:
        - id: node identifier
        - vec: embedding vector
        - text: text content
        - metadata: dict with node metadata (will be stored as _node_content JSON string)
        """
        import json

        label = vector_config.node_label
        prop = vector_config.vector_property

        # Serialize metadata to JSON strings for Neo4j storage
        serialized_rows = []
        for row in rows:
            serialized_row = {
                "id": row["id"],
                "vec": row["vec"],
                "text": row["text"],
                "metadata_json": json.dumps(
                    row.get("metadata", {})
                ),  # Serialize as JSON string
            }
            serialized_rows.append(serialized_row)

        def _write(tx: ManagedTransaction, /) -> int:
            result = tx.run(
                f"""
                UNWIND $rows AS row
                MATCH (n:{label} {{ id: row.id }})
                SET n.{prop} = row.vec,
                    n.text = row.text,
                    n._node_content = row.metadata_json,
                    n.updated_at = timestamp()
                RETURN count(n) as updated
                """,
                {"rows": serialized_rows},
            )
            record = result.single()
            return record["updated"] if record else 0

        with self.driver.session() as session:
            updated_count = session.execute_write(_write)
        return updated_count
