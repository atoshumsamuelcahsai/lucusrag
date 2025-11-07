#!/bin/bash
# Complete Docker reset and rebuild for LucusRAG

set -e

echo "🧹 Cleaning up Docker environment..."

# Stop and remove everything
docker-compose down -v 2>/dev/null || true

# Remove old containers
docker rm -f lucusrag-app lucusrag-neo4j 2>/dev/null || true

# Remove old images
docker rmi lucusrag_app lucusrag_neo4j 2>/dev/null || true

# Prune dangling images and build cache
docker image prune -f
docker builder prune -f

echo ""
echo "🔨 Building fresh images..."
docker-compose build --no-cache

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "✅ Done! Services are starting up."
echo ""
echo "Next steps:"
echo "  - Watch logs: docker-compose logs -f"
echo "  - Test Neo4j: curl http://localhost:7474"
echo "  - Test API: curl http://localhost:8000/health"

