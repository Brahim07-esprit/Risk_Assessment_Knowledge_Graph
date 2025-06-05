#!/bin/bash
# Docker run script for Risk Assessment Knowledge Graph

echo "🎯 Risk Assessment Knowledge Graph - Docker Setup"
echo "================================================"

# Create data directory
mkdir -p data

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose down 2>/dev/null || docker-compose down 2>/dev/null

# Build and start services
echo "🔨 Building Docker images..."
docker compose build || docker-compose build

echo "🚀 Starting services..."
docker compose up -d || docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 15

# Check service health
echo "📊 Checking service status..."
docker compose ps || docker-compose ps

echo ""
echo "✅ Setup complete!"
echo ""
echo "🔗 Access Points:"
echo "  - Streamlit App: http://localhost:8501"
echo "  - Neo4j Browser: http://localhost:7474"
echo ""
echo "📝 Credentials:"
echo "  - Neo4j: neo4j / password"
echo ""
echo "📋 Useful commands:"
echo "  - View logs: docker compose logs -f"
echo "  - Stop services: docker compose down"
echo "  - View Neo4j data: docker exec -it risk-kg-neo4j cypher-shell" 