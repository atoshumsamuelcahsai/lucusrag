from __future__ import annotations

import asyncio
import logging
import random
import typing as t

from dotenv import load_dotenv
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction, AsyncSession

from rag.schemas import CodeElement
from rag.schemas.vector_config import Neo4jConfig, VectorIndexConfig
import json

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
        self._driver: t.Optional[AsyncDriver] = None
        self._max_retries = max_retries
        self._delay = delay

    async def driver(self) -> AsyncDriver:
        """Lazy init Neo4j async driver and validate connectivity."""
        if self._driver is None:
            for attempt in range(self._max_retries):
                try:
                    self._driver = AsyncGraphDatabase.driver(
                        self.config.url,
                        auth=(self.config.user, self.config.password),
                        max_connection_lifetime=30 * 60,
                    )
                    await self._driver.verify_connectivity()
                    logger.info(f"Connected to Neo4j at {self.config.url}")
                    break
                except Exception as e:
                    if attempt == self._max_retries - 1:
                        raise ConnectionError(f"Failed to connect: {e}")
                    delay = self._delay * (2 ** (attempt))
                    await asyncio.sleep(delay + random.random())
                    logger.warning(
                        f"Retrying Neo4j connection ({attempt+1}/{self._max_retries})..."
                    )
        assert self._driver is not None, "Driver should be initialized by now"
        return self._driver

    async def close(self) -> None:
        """Close the db connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    # Schema creation
    async def create_schema(self, vector_config: VectorIndexConfig) -> None:
        """Create or verify Neo4j schema with indexes and constraints."""
        try:
            driver = await self.driver()
            async with driver.session() as session:
                # Create base constraints and indexes
                await self._create_data_schema(session, vector_config)
                await self._create_vector_index(session, vector_config, overwrite=True)
                logger.info(
                    f"Neo4j schema ensured for label {vector_config.node_label}"
                )
        except Exception as e:
            logger.exception(f"Failed to create schema: {str(e)}")
            raise

    async def _create_data_schema(
        self, session: AsyncSession, vector_config: VectorIndexConfig
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

        async def _run_schema(tx: AsyncManagedTransaction) -> None:
            for stmt in statements:
                await tx.run(stmt)

        await session.execute_write(_run_schema)
        logger.info(f"Created data schema for label {vector_config.node_label}")

    async def _index_exists(
        self, session: AsyncSession, config: VectorIndexConfig
    ) -> bool:
        # Check if index exists
        result = await session.run(
            """
                SHOW INDEXES
                YIELD name
                WHERE name = $index_name
                RETURN count(*) > 0 as exists
                """,
            {"index_name": config.name},
        )
        record = await result.single()
        return bool(record and record.get("exists", False))

    async def _drop_index(
        self, session: AsyncSession, config: VectorIndexConfig
    ) -> None:
        await session.run(f"DROP INDEX `{config.name}` IF EXISTS")
        logger.info(f"Dropped existing vector index {config.name}")

    async def _create_index(
        self, session: AsyncSession, config: VectorIndexConfig
    ) -> None:
        await session.run(
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

    async def _create_vector_index(
        self, session: AsyncSession, config: VectorIndexConfig, overwrite: bool = True
    ) -> None:
        """Create or recreate a vector index with given configuration. This could be very expensive"""
        exists = await self._index_exists(session, config)

        if exists:
            if overwrite:
                await self._drop_index(session, config)
                await self._create_index(session, config)
            else:
                logger.info("Vector index %s already exists; skipping", config.name)
        else:
            await self._create_index(session, config)

    # ---------- Domain upserts ----------
    async def create_node(
        self: GraphDBManager,
        code_info: CodeElement,
        vector_config: VectorIndexConfig,
    ) -> None:
        """Create or update a full CodeElement node with all metadata (except relationships)."""
        label = vector_config.node_label

        async def _upsert(tx: AsyncManagedTransaction) -> None:
            await tx.run(
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
            driver = await self.driver()
            async with driver.session() as session:
                await session.execute_write(_upsert)
                logger.info(f"Upserted node for {code_info.name}")
        except Exception:
            logger.exception(f"Failed to create node for {code_info.name}")
            raise

    async def _add_inheritance(
        self,
        session: AsyncSession,
        vector_config: VectorIndexConfig,
        code_info: CodeElement,
    ) -> None:
        if not code_info.base_classes:
            return

        label = vector_config.node_label

        async def _link(tx: AsyncManagedTransaction) -> None:
            await tx.run(
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

        await session.execute_write(_link)

    async def _add_calls(
        self,
        session: AsyncSession,
        vector_config: VectorIndexConfig,
        code_info: CodeElement,
    ) -> None:
        if not code_info.calls:
            return
        label = vector_config.node_label

        async def _link(tx: AsyncManagedTransaction) -> None:
            await tx.run(
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

        await session.execute_write(_link)

    async def _add_dependencies(
        self,
        session: AsyncSession,
        vector_config: VectorIndexConfig,
        code_info: CodeElement,
    ) -> None:
        if not code_info.dependencies:
            return
        label = vector_config.node_label

        async def _link(tx: AsyncManagedTransaction) -> None:
            await tx.run(
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

        await session.execute_write(_link)

    async def create_relationships(
        self, code_info: CodeElement, vector_config: VectorIndexConfig
    ) -> None:
        """Create relationships between nodes."""
        try:
            driver = await self.driver()
            async with driver.session() as session:
                # 1. Add inheritance
                if code_info.base_classes:  # Direct attribute access
                    await self._add_inheritance(session, vector_config, code_info)

                # 2. Add calls
                if code_info.calls:  # Direct attribute access
                    await self._add_calls(session, vector_config, code_info)

                # 3. Add dependencies
                if code_info.dependencies:  # Direct attribute access
                    await self._add_dependencies(session, vector_config, code_info)

            logger.info(f"Added relationships for {code_info.name}")

        except Exception as e:
            logger.exception(
                f"Failed to add relationships for {code_info.name}: {str(e)}"
            )
            raise

    # ----------  upsert embeddings ----------
    async def upsert_embeddings(
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

        async def _write(tx: AsyncManagedTransaction, /) -> int:
            result = await tx.run(
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
            record = await result.single()
            return record["updated"] if record else 0

        driver = await self.driver()
        async with driver.session() as session:
            updated_count = await session.execute_write(_write)
        return updated_count
