"""
Ingestion module for processing and loading code data.

This module handles the complete data preparation pipeline:
- Phase 1: Loading AST-derived data and building graph structure
- Phase 2: Enriching graph nodes with embeddings

Phase 1 (data_loader):
- Loading AST-derived data from JSON files
- Processing code elements into LlamaIndex documents
- Populating Neo4j database with nodes and relationships

Phase 2 (embedding_loader):
- Generating embeddings for code elements
- Updating existing Neo4j nodes with embeddings
"""

from rag.ingestion.data_loader import process_code_files
from rag.ingestion.embedding_loader import populate_embeddings
from rag.schemas.vector_config import get_vector_index_config

__all__ = ["process_code_files", "populate_embeddings", "get_vector_index_config"]
