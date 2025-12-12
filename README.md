# LucusRAG

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ff25ca52e8e4476388db4568687ef61b)](https://app.codacy.com/gh/atoshumsamuelcahsai/lucusrag?utm_source=github.com&utm_medium=referral&utm_content=atoshumsamuelcahsai/lucusrag&utm_campaign=Badge_Grade)
[![CI](https://github.com/atoshumsamuelcahsai/lucusrag/actions/workflows/ci.yml/badge.svg)](https://github.com/atoshumsamuelcahsai/lucusrag/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/atoshumsamuelcahsai/lucusrag/branch/master/graph/badge.svg?token=)](https://codecov.io/gh/atoshumsamuelcahsai/lucusrag)

**A Production-Grade Retrieval-Augmented Generation (RAG) System for Code Understanding**


LucusRAG is a hybrid retrieval system that combines Tree-Sitter AST parsing, graph-based program analysis, LLM-augmented semantic explanations, rrf fusion, and neural reranking, and a probabilistic Adaptive-K policy to deliver **fast, accurate, and cost-efficient code understanding**.

**Performance highlights:**
- **2.5× faster** query time
- **60–64% cheaper** per request
- **8.6/10 accuracy** on complex code queries
- Hybrid retrieval: BM25 + Vector + Graph + rrf +  Reranker

---

## TL;DR
LucusRAG builds an indexed code graph from a repository, enriches it with embeddings, and performs hybrid retrieval with neural reranking and cost-aware Adaptive-K context selection.

Ideal for:
- Engineering knowledge bases
- Large multi-file repositories
- Production RAG systems
- Cost‑efficient retrieval pipelines

---

## 🔍 Key Features
- **Hybrid Retrieval**: BM25 + dense embeddings + graph expansion
- **Graph-Aware Context**: Neo4j program graph (CALLS, INHERITS_FROM, DEPENDS_ON)
- **Neural Reranking**: Cross-encoder fine-grained scoring
- **Adaptive-K Policy**: Dynamic, budget-aware context selection
- **AST-Aware Chunking**: Tree-Sitter parsing
- **LLM Explanation Augmentation**: AI-generated semantic explanations for improved retrieval discriminability
- **Production Engineering**: Docker, CI/CD, strict mypy, pre-commit

---

##  Architecture Overview

### **Indexing Pipeline**
```
Phase 1 → AST → Neo4j Graph
Phase 2 → Embeddings → Neo4j Vector Fields
Phase 3 → Vector Index → Ready for Retrieval
```

### **Query Pipeline**
```
Query
 ↓
BM25 + Vector
 ↓
RRF Fusion
 ↓
Graph Expansion
 ↓
Cross‑Encoder Reranking
 ↓
Adaptive‑K Selection
 ↓
LLM Response
```

---

## Performance Summary
| Metric | Result |
|--------|--------|
| Average Accuracy | **8.6/10** |
| Focused Queries | **9.2/10** |
| Response Time | **2.5× faster** |
| Cost per Query | **~$0.01 (↓60%)** |

Full results: `docs/LUCUSRAG_PAPER.pdf`

---

## ⚙️ Quick Start

### 1. Clone & Setup
```
git clone <repo-url>
cd lucusrag

conda create -n lucusrag python=3.11
conda activate lucusrag
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Install Git hooks:
```
pre-commit install --hook-type pre-commit --hook-type pre-push
```

### 2. Start Neo4j
```
docker-compose up -d
```

### 3. Build AST Files
```
python -m rag.ast.ast_builder --root-dir ./rag --output-dir ./ast_cache --parser-type tree-sitter
```

Or programmatically:
```python
from rag.ast.ast_builder import analyze_and_store_python_files

output_path = await analyze_and_store_python_files(
    root_dir="./rag",
    output_path="./ast_cache",
    parser_type="tree-sitter"
)
```

### 4. Build the Index
```
from rag.indexer.orchestrator import CodeGraphIndexer

indexer = CodeGraphIndexer(ast_cache_dir="./ast_cache", top_k=5)
await indexer.build()
```

### 5. Run a Query
```
from rag.query_processor import process_query

response = await process_query("Explain graph expansion")
print(response)
```

---

## Project Structure
```
lucusrag/
├── rag/               # Core RAG engine
├── docs/              # Paper + architecture
├── test/              # Unit + integration tests (85% coverage)
├── evaluation/        # Evaluation notebook
├── Dockerfile
├── docker-compose.yaml
└── README.md
```

---

## Testing & CI/CD
```
pytest --cov=rag --cov-report=html
```

Tools:
- pytest
- coverage
- mypy (strict)
- Ruff
- Black
- Bandit
- pip-audit
- pre-commit hooks

---

## Configuration

Create `.env`:
```
NEO4J_URL=bolt://localhost:7687
LLM_PROVIDER=anthropic
EMBEDDING_PROVIDER=voyage
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
VOYAGE_API_KEY=your_key
```

Adaptive-K settings:
```
probability_target=0.70
k_min=2
k_max=10
max_cost_per_query=0.01
```

---

## Documentation
- Architecture: `docs/ARCHITECTURE.md`
- Technical paper: `docs/LUCUSRAG_PAPER.pdf`
- Evaluation notebook & test queries data: `/evaluation/`

---

## Contributing
1. Fork the repository
2. Create a branch
3. Install pre-commit hooks
4. Add tests
5. Ensure tests pass
6. Open a PR

---

## License
MIT License.

---

## ❤️ Built for production-grade code understanding.
