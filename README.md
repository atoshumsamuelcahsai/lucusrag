# LucusRAG

A production-ready Retrieval-Augmented Generation (RAG) system for code search and understanding, combining semantic search, graph traversal, and hybrid retrieval strategies.

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
# Install pre-commit hooks (runs on git commit)
pre-commit install

# Install pre-push hooks (runs on git push)
pre-commit install --hook-type pre-push
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

# LLM Configuration
LLM_PROVIDER=anthropic
EMBEDDING_PROVIDER=voyage

# API Keys (add your keys)
ANTHROPIC_API_KEY=your_key_here
VOYAGE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # Optional
```

### 6. Start Neo4j with Docker Compose

```bash
docker-compose up -d
```

Verify Neo4j is running:
- Browser UI: http://localhost:7474
- Bolt endpoint: bolt://localhost:7687

