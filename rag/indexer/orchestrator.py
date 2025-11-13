from __future__ import annotations
from enum import Enum
import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine


from rag.ingestion import process_code_files
from rag.indexer.vector_indexer import (
    create_vector_index_from_existing_nodes,
    graph_configure_settings,
)
from rag.engine.engine import make_query_engine, retrieve_documents_from_engine
from rag.schemas.vector_config import VectorIndexConfig

import logging
import aiofiles  # type: ignore


logger = logging.getLogger(__name__)


class Mode(Enum):
    BUILD = "build"
    REFRESH = "refresh"


@dataclass(frozen=True)
class BuildResult:
    documents: int
    elapsed_s: float
    mode: Mode
    schema_version: int

    def __str__(self) -> str:
        return (
            f"[{self.mode.value.upper()}] "
            f"{self.documents} docs | "
            f"{self.elapsed_s:.2f}s | schema v{self.schema_version}"
        )


class CodeGraphIndexer:
    """
     end-to-end lifecycle of code graph + vector index.

    Public API:
      - build()   : cold build from AST cache
      - refresh() : incremental update (only changed/removed files)
      - query()   : run a question through the query engine
    """

    def __init__(
        self,
        ast_cache_dir: str,
        schema_version: int = 1,
        top_k: int = 5,
        manifest_path: Optional[str] = None,
    ) -> None:

        self.ast_dir = Path(ast_cache_dir)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path
            else self.ast_dir / ".rag_manifest.json"
        )
        self.schema_version = schema_version
        self.top_k = top_k

        self._index: Optional[VectorStoreIndex] = None
        self._engine: Optional[RetrieverQueryEngine] = None

    async def build(self) -> BuildResult:
        """
        Cold build: build graph structure (nodes + relationships)
        and if DB empty populate embeddings and graph
        """

        logger.info("Cold build started.....")
        t0 = time.perf_counter()

        await self._get_graph_config()
        config = VectorIndexConfig.from_env()

        # Build graph structure (nodes + relationships)
        #  and populate embeddings and graph if DB empty
        logger.info(
            "Building graph structure (nodes + relationships + populate if empty)..."
        )
        docs = await process_code_files(str(self.ast_dir))
        logger.info(f"Graph structure built with {len(docs)} nodes")

        logger.info("Getting vector index from existing nodes…")
        # Parse AST docs once and pass them down so vector_indexer can hydrate docstore cleanly

        self._index = await create_vector_index_from_existing_nodes(config, docs=docs)
        logger.info("Vector index created successfully!!!")

        await self._update_manifest(config)
        logger.info("Manifest files updated successfully!!!")

        # Wire query engine
        self._engine = make_query_engine(self._index, k=self.top_k)

        dt = time.perf_counter() - t0
        logger.info(f"Cold build successfully finished in {dt:.2f}s")
        return BuildResult(
            documents=len(docs),
            elapsed_s=dt,
            mode=Mode.BUILD,
            schema_version=self.schema_version,
        )

    async def refresh(self) -> BuildResult:
        """
        Incremental: detect changed/removed AST JSONs and rebuild.

        Uses the same 2-phase architecture as build():
        Phase 1: Rebuild graph structure (force rebuild)
        Phase 2: Recreate vector index

        NOTE: Currently rebuilds everything on any change.
        Future optimization: partial updates for changed files only.
        """
        logger.info("Refresh started........")
        t0 = time.perf_counter()
        config: VectorIndexConfig = VectorIndexConfig.from_env()

        prev = self._load_manifest()
        now = await self._snapshot_files()
        changed, removed = self._diff(
            prev, now, embed_sig=self._embed_signature(config)
        )

        if not changed and not removed:
            logger.info("No changes detected, skipping refresh")
            return BuildResult(
                documents=0,
                elapsed_s=0.0,
                mode=Mode.REFRESH,
                schema_version=self.schema_version,
            )

        logger.info(f"Detected {len(changed)} changed and {len(removed)} removed files")

        # Phase 1: Rebuild graph structure (force rebuild)
        logger.info("Rebuilding graph structure...")
        docs = await process_code_files(str(self.ast_dir), force_rebuild_graph=True)

        await self._save_manifest(now, embed_sig=self._embed_signature(config))
        logger.info("Manifest files updated successfully!!!")

        # Phase 2: Recreate vector index
        self._index = await create_vector_index_from_existing_nodes(config, docs=docs)
        self._engine = make_query_engine(self._index, k=self.top_k)
        logger.info("Vector index recreated successfully")

        dt = time.perf_counter() - t0
        logger.info(f"Refresh successfully finished in {dt:.2f}s")
        return BuildResult(
            documents=len(docs),
            elapsed_s=dt,
            mode=Mode.REFRESH,
            schema_version=self.schema_version,
        )

    async def _update_manifest(self, config: VectorIndexConfig) -> None:
        # Write manifest for future diffs
        _snap_shot_files = await self._snapshot_files()
        await self._save_manifest(
            _snap_shot_files, embed_sig=self._embed_signature(config)
        )

    def query(self, text: str) -> str:
        if not self._engine:
            raise RuntimeError("Indexer not built. Call build() or refresh() first.")
        resp = self._engine.query(text)
        return str(resp)

    async def aquery(self, text: str) -> str:
        if not self._engine:
            raise RuntimeError("Indexer not built. Call build() or refresh() first.")
        resp = await self._engine.aquery(text)
        return str(resp)

    async def retrieve_documents(self, query: str, k: int = 20) -> list[Dict[str, Any]]:
        """Retrieve top-k documents for a query without LLM processing.

        Uses the engine's retrieval function to get raw retrieval results.
        This is useful for evaluation purposes.

        Args:
            query: Query string to search for
            k: Number of documents to retrieve (default: 20)

        Returns:
            List of dicts with retrieval results (see engine.retrieve_documents_from_engine)

        Raises:
            RuntimeError: If engine is not initialized
        """
        if self._engine is None:
            raise RuntimeError("Indexer not built. Call build() or refresh() first.")

        return await retrieve_documents_from_engine(self._engine, query, k)

    async def _get_graph_config(self) -> None:
        await graph_configure_settings()

    async def _snapshot_files(self) -> Dict[str, Dict[str, Any]]:
        files: Dict[str, Dict[str, Any]] = {}
        for p in self.ast_dir.rglob("*.json"):
            try:
                stat = p.stat()
                sha = await self._sha256(p)
                files[str(p)] = {
                    "mtime": int(stat.st_mtime),
                    "sha256": sha,
                }
            except Exception:
                continue
        return files

    async def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        async with aiofiles.open(path, "rb") as f:
            while True:
                chunk = await f.read(1 << 16)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _embed_signature(self, cfg: VectorIndexConfig) -> str:
        # tie manifest to embedding choices so model/dimension change triggers rebuild
        # you can import provider/model from your Settings if you expose them
        return f"{cfg.dimension}:{cfg.similarity_metric}"

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": self.schema_version, "files": {}, "embed_sig": ""}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def _save_manifest(self, files: dict[str, Any], embed_sig: str) -> None:
        out = {"version": self.schema_version, "files": files, "embed_sig": embed_sig}
        async with aiofiles.open(self.manifest_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(out, indent=2))

    def _diff(
        self,
        prev: Dict[str, Any],
        now: Dict[str, Any],
        embed_sig: str,
    ) -> Tuple[Sequence[str], Sequence[str]]:
        # trigger rebuild if schema version or embedding signature changed
        if (
            prev.get("version") != self.schema_version
            or prev.get("embed_sig") != embed_sig
        ):
            return (
                list(now.keys()),
                list(set(prev.get("files", {}).keys()) - set(now.keys())),
            )

        prev_files = prev.get("files", {})
        changed = []
        removed = []

        # removed
        for old_path in prev_files.keys():
            if old_path not in now:
                removed.append(old_path)

        # changed (mtime or sha differs)
        for path, meta in now.items():
            pmeta = prev_files.get(path)
            if not pmeta:
                changed.append(path)
                continue
            if meta["mtime"] != pmeta.get("mtime") or meta["sha256"] != pmeta.get(
                "sha256"
            ):
                changed.append(path)

        return (changed, removed)
