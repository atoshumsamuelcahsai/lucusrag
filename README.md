# LucusRAG

A production-ready Retrieval-Augmented Generation (RAG) system for code search and understanding, combining semantic search, graph traversal, and hybrid retrieval strategies.

## About This Project

This project represents my journey from proof-of-concept to production-ready software, demonstrating advanced RAG techniques and modern software engineering practices.

**Origin Story:**
The initial concept emerged from identifying a real-world need at my previous company—building a centralized knowledge base combining code repositories, Confluence docs, and other documentation for customer support, product, and engineering teams. I developed the initial proof-of-concept independently on my own time to validate the technical approach.

**From POC to Production:**
This repository is a complete ground-up implementation showcasing my ability to architect production-ready systems. Key improvements include:

- ✅ **Clean Architecture:** Well-structured codebase with separation of concerns (ingestion, indexing, querying)
- ✅ **Comprehensive Testing:**  unit tests with pytest, including integration and edge cases
- ✅ **Production Infrastructure:** Docker containerization, docker-compose orchestration, and CI/CD with pre-commit hooks
- ✅ **Advanced Features:** AST-based code parsing, graph-based context expansion, hybrid retrieval strategies
- ✅ **Type Safety:** Full mypy type checking and static analysis
- ✅ **Modern DevOps:** Black, ruff, bandit, pip-audit for code quality and security

This project is actively used for my Lucus project (WiFi location algorithms) and serves as a foundation for building intelligent code analysis systems.

## Prerequisites

- Python 3.9+
- Conda (recommended) or virtualenv
- Docker and Docker Compose
- Neo4j (via Docker Compose)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd lucusrag
```

### 2. Create Conda Environment

```bash
conda create -n lucusrag python=3.11
conda activate lucusrag
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirments.txt

# Install development dependencies
pip install -r requirments-dev.txt
```

### 4. Install Pre-commit Hooks **IMPORTANT**

**This step is required for git hooks to work!** After installing dependencies, you must install the pre-commit hooks:

```bash
# Install pre-commit hooks (runs on git commit and push)
pre-commit install --hook-type pre-commit --hook-type pre-push
```

**Why this is needed:** Installing the `pre-commit` package doesn't automatically set up git hooks. You must run `pre-commit install` once per repository to configure git to run the hooks.

Verify installation:
```bash
ls -la .git/hooks/pre-commit .git/hooks/pre-push
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash for example
# Neo4j Configuration
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Vector Index Configuration
VECTOR_INDEX_NAME=lucus_code_embeddings
VECTOR_DIMENSION=1536
NODE_LABEL=LucusCodeElement
VECTOR_PROPERTY=embedding
SIMILARITY_METRIC=cosine

# LLM Configuration (single source of truth)
LLM_PROVIDER=anthropic  # Options: anthropic, openai
LLM_MODEL=claude-3-5-sonnet  # Model name (e.g., claude-3-5-sonnet, gpt-4, gpt-4-turbo)
LLM_TEMPERATURE=0  # Optional, default: 0
LLM_MAX_OUTPUT_TOKENS=768  # Maximum output tokens (default: 768)
LLM_CONTEXT_WINDOW=4200  # Optional, default: 4200

# Embedding Configuration
EMBEDDING_PROVIDER=voyage  # Options: voyage, voyage-large, voyage-lite, text-embedding-3-small

# API Keys (add your keys)
ANTHROPIC_API_KEY=your_key_here
VOYAGE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # Optional, required if using OpenAI LLM or embeddings
```

### 6. Start Neo4j with Docker Compose

```bash
docker-compose up -d
```

Verify Neo4j is running:
- Browser UI: http://localhost:7474
- Bolt endpoint: bolt://localhost:7687

