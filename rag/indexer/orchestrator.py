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


from rag.ingestion.data_loader import process_code_files, get_vector_index_config
from rag.indexer.vector_indexer import (
    get_vector_index,
    graph_configure_settings,
)
from rag.engine import make_query_engine
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
        manifest_path: Optional[str] = None,
        schema_version: int = 1,
        top_k: int = 5,
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
        """Cold build: parse → index → wire engine."""
        logger.info("Cold build started.....")
        t0 = time.perf_counter()

        # global Settings.llm, embed_model, parser
        await self._get_graph_config()

        # VectorIndexConfig.from env
        config = get_vector_index_config()

        # already handles optional DB population
        docs = process_code_files(str(self.ast_dir))

        # (re)build vector/graph index
        self._index = get_vector_index(docs, config)

        # 4) wire query engine
        self._engine = make_query_engine(self._index, k=self.top_k)

        # 5) write manifest for future diffs
        _snap_shot_files = await self._snapshot_files()
        await self._save_manifest(
            _snap_shot_files, embed_sig=self._embed_signature(config)
        )

        dt = time.perf_counter() - t0
        logger.info(f"Cold build successfully finished in {dt}s")
        return BuildResult(
            documents=len(docs),
            elapsed_s=dt,
            mode=Mode.BUILD,
            schema_version=self.schema_version,
        )

    async def refresh(self) -> BuildResult:
        """
        Incremental: detect changed/removed AST JSONs
        and update only those.
        NOTE: This is a pragmatic first step — rebuild
        the index if anything changed.
        Investigate and implement real partial upserts later.
        """
        logger.info("refreshing index has started........")
        t0 = time.perf_counter()
        config: VectorIndexConfig = get_vector_index_config()

        prev = self._load_manifest()
        now = await self._snapshot_files()
        changed, removed = self._diff(
            prev, now, embed_sig=self._embed_signature(config)
        )

        if not changed and not removed:
            logger.info("No change has been detected, no need for refreshing index....")
            return BuildResult(
                documents=0,
                elapsed_s=0.0,
                mode=Mode.REFRESH,
                schema_version=self.schema_version,
            )

        # Simple, safe strategy: re-run the loader (cheap for
        # lucus repos but generally expensive).
        docs = process_code_files(str(self.ast_dir))
        self._index = get_vector_index(docs, config)
        self._engine = make_query_engine(self._index, k=self.top_k)
        await self._save_manifest(now, embed_sig=self._embed_signature(config))

        dt = time.perf_counter() - t0
        logger.info(f"Index Refresher has successfully finished in {dt}s")
        return BuildResult(
            documents=len(docs),
            elapsed_s=dt,
            mode=Mode.REFRESH,
            schema_version=self.schema_version,
        )

    def query(self, text: str) -> str:
        if not self._engine:
            raise RuntimeError("Indexer not built. Call build() or refresh() first.")
        resp = self._engine.query(text)
        return str(resp)

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
