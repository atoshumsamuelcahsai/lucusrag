# LucusRAG System Architecture

## Overview

LucusRAG is a production-grade RAG system that combines structural code parsing, graph-based representations, and hybrid retrieval with an adaptive-K policy for cost-aware document selection.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          LUCUSRAG SYSTEM ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: GRAPH STRUCTURE                         │
│                      (rag/ingestion/data_loader.py)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   AST JSON Files  ──→  CodeElement Parser  ──→  Neo4j Graph             │
│                                                                           │
│   • Tree-Sitter AST           • Classes/Functions      • Node Creation   │
│   • Structure Extraction      • Metadata Extraction    • Relationships   │
│   • File Metadata            • Parameter Analysis      • CALLS edges     │
│                                                        • DEPENDS_ON      │
│                                                                           │
│   Concurrency: asyncio.Semaphore(50) for files, (20) for DB writes      │
│   Connection: Neo4j pooling with 3-level exponential backoff            │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: EMBEDDING POPULATION                       │
│                    (rag/ingestion/embedding_loader.py)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   Neo4j Nodes  ──→  Embedding Provider  ──→  Update Neo4j               │
│                                                                           │
│   • Read nodes             • Voyage AI           • Batched UNWIND       │
│   • Format text            • OpenAI              • Single transaction   │
│   • Generate embeddings    • 1536-dim vectors    • In-place updates     │
│                                                                           │
│   Processing: Sequential requests with batch payload assembly            │
│   Error Handling: Individual failures logged, batch continues            │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: VECTOR INDEX CREATION                       │
│                     (rag/indexer/vector_indexer.py)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   Neo4j Nodes  ──→  LlamaIndex Configuration  ──→  VectorStoreIndex     │
│                                                                           │
│   • Read nodes+embeddings  • Neo4jVectorStore      • Query-ready index  │
│   • Read relationships     • Neo4jGraphStore       • Retriever config   │
│   • Verify structure       • Settings config       • Read-only mode     │
│   • Docstore hydration     • Graph store attach    • Multiple retriever │
│                                                      types supported     │
│                                                                           │
│   Safety: No writes, CI/CD-safe, idempotent configuration               │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL PIPELINE (QUERY TIME)                    │
│                         (rag/engine/engine.py)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   User Query                                                              │
│       │                                                                   │
│       ├──→ [1] BM25 Retrieval (Sparse/Lexical)                          │
│       │                                                                   │
│       ├──→ [2] Vector Retrieval (Dense/Semantic)                        │
│       │                                                                   │
│       └──→ [3] Hybrid RRF Fusion                                        │
│                   │                                                       │
│                   ├──→ [4] Graph Expansion (1-hop, max 50 nodes)        │
│                   │                                                       │
│                   └──→ [5] Neural Reranking (Cross-Encoder)             │
│                               │                                           │
│                               └──→ [6] Adaptive-K Selection              │
│                                           │                               │
│                                           └──→ Context for LLM           │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      ADAPTIVE-K POLICY ALGORITHM                         │
│                       (rag/engine/adaptive_k.py)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   Input: Reranked nodes with scores                                      │
│                                                                           │
│   1. Convert scores to softmax probabilities (temperature τ = 1.0)      │
│   2. Compute cumulative probability mass P_k = Σ(p_1...p_k)             │
│   3. Select top k_min (default: 2) documents                             │
│   4. For each additional candidate:                                      │
│      • Check: cumulative mass >= target (0.70) → STOP                   │
│      • Check: cost > budget ($0.01/query) → STOP                        │
│      • Check: k >= k_max (10) → STOP                                    │
│      • Else: ADD document, continue                                      │
│                                                                           │
│   Output: Selected documents + metadata (k, cost, stop reason)          │
│                                                                           │
│   Impact: 2.5× faster, 60-64% cost reduction, maintains 8.6/10 accuracy │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE & DEVOPS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   Docker Compose Stack:                                                  │
│   ┌────────────────┐         ┌────────────────┐                        │
│   │  FastAPI App   │ ←───→   │   Neo4j DB     │                        │
│   │  Port: 8000    │         │   Port: 7687   │                        │
│   └────────────────┘         └────────────────┘                        │
│                                                                           │
│   CI/CD Pipeline:                                                        │
│   • Black (formatting)                                                   │
│   • Ruff (linting)                                                       │
│   • mypy (type checking)                                                 │
│   • Bandit (security scanning)                                           │
│   • pip-audit (dependency vulnerabilities)                               │
│   • pytest (unit + integration tests)                                    │
│                                                                           │
│   Testing Strategy:                                                      │
│   • Unit tests for each module                                           │
│   • Integration tests for pipeline                                       │
│   • Mocked external dependencies                                         │
│   • Edge case and error handling coverage                                │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. **Clean Separation of Concerns**
- **Phase 1 (Graph)**: Structure only, no embeddings
- **Phase 2 (Embeddings)**: Updates existing nodes, no new structure
- **Phase 3 (Index)**: Read-only, query preparation

### 2. **Asynchronous Processing**
- `asyncio.Semaphore` for concurrency control
- Connection pooling with retry logic
- Non-blocking I/O for scalability

### 3. **Cost-Aware Retrieval**
- Adaptive-K dynamically adjusts context size
- Probabilistic stopping based on score distribution
- Budget constraints prevent cost overruns

### 4. **Production Readiness**
- Comprehensive test coverage
- Type safety with mypy
- Security scanning (Bandit, pip-audit)
- Docker containerization
- Health checks and monitoring

### 5. **Graph-Aware Context**
- Multi-hop reasoning via Neo4j relationships
- Semantic edges (CALLS, DEPENDS_ON, INHERITS_FROM)
- Neighborhood expansion for richer context

## Module Structure

```
rag/
├── ast/                    # Tree-Sitter AST parsing
│   ├── ast_builder.py     # AST extraction
│   └── builders.py        # Builder patterns
├── db/                     # Database layer
│   └── graph_db.py        # Neo4j driver and operations
├── engine/                 # Query engine
│   ├── engine.py          # Main query engine
│   ├── retrievers.py      # Hybrid retrievers (BM25, Vector, Graph, RRF)
│   └── adaptive_k.py      # Adaptive-K policy
├── indexer/                # Indexing orchestration
│   ├── orchestrator.py    # 3-phase pipeline coordinator
│   └── vector_indexer.py  # LlamaIndex configuration
├── ingestion/              # Data ingestion
│   ├── data_loader.py     # Phase 1: Graph structure
│   └── embedding_loader.py # Phase 2: Embeddings
├── parser/                 # Document parsing
│   └── parser.py          # CodeElement → LlamaIndex nodes
├── providers/              # External services
│   ├── embeddings.py      # Embedding providers (Voyage, OpenAI)
│   └── llms.py            # LLM providers (Anthropic, OpenAI)
└── schemas/                # Data models
    ├── code_element.py    # AST element schema
    ├── vector_config.py   # Vector index configuration
    └── ...                # Other schemas
```

## Performance Characteristics

| Metric | Baseline | With Adaptive-K | Improvement |
|--------|----------|----------------|-------------|
| Average Response Time | 49.5s | 19.5s | **2.5× faster** |
| LLM Processing Time | 46.5s | 16.8s | **64% reduction** |
| Cost per Query | ~$0.025 | ~$0.01 | **60% reduction** |
| Average Accuracy | 8.4/10 | 8.6/10 | **Maintained** |
| Focused Query Accuracy | 9.0/10 | 9.2/10 | **Improved** |

## Evaluation Metrics

### Information Retrieval Metrics
- **Precision@K**: Fraction of retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents retrieved
- **F₁ Score**: Harmonic mean of precision and recall
- **NDCG@K**: Normalized Discounted Cumulative Gain (ranking quality)

### Agentic Correctness
- External LLM judge (0-10 scale)
- Evaluated on 13 representative queries
- Criteria: accuracy, completeness, clarity, relevance

## Deployment

### Local Development
```bash
# Start services
docker-compose up -d

# Access web UI
open http://localhost:8000

# Query via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "How does graph expansion work?", "ast_cache_dir": "./ast_cache"}'
```

### Production Deployment
- Docker Compose orchestration
- Health checks for Neo4j and FastAPI
- Volume mounts for data persistence
- Environment-based configuration
- Automatic service recovery

## Future Enhancements

1. **Query-Adaptive Probability Targets**: Dynamic targets based on query intent
2. **Incremental Graph Updates**: Efficient handling of code changes
3. **Multi-Modal Retrieval**: Documentation, comments, tests
4. **Query Classification**: Automatic strategy selection
5. **Extended Evaluation**: Full 30-query dataset assessment
6. **Advanced Graph Patterns**: Multi-hop reasoning, path analysis
7. **Caching Layer**: Query result caching for repeated queries
8. **Distributed Processing**: Horizontal scaling for large codebases

