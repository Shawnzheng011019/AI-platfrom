#!/bin/bash

# AI Training Platform Validation Script

set -e

echo "🔍 Validating AI Training Platform configuration..."

# Check if required files exist
echo "📁 Checking required files..."

REQUIRED_FILES=(
    "docker-compose.yml"
    "backend/Dockerfile"
    "frontend/Dockerfile"
    "backend/requirements.txt"
    "frontend/package.json"
    "docker/prometheus/prometheus.yml"
    "docker/prometheus/alert_rules.yml"
    "docker/mongodb/init-mongo.js"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
done
echo "✅ All required files present"

# Check if scripts are executable
echo "🔧 Checking script permissions..."
chmod +x scripts/*.sh
echo "✅ Script permissions set"

# Validate Docker Compose configuration
echo "🐳 Validating Docker Compose configuration..."
if docker-compose config > /dev/null 2>&1; then
    echo "✅ Docker Compose configuration is valid"
else
    echo "❌ Docker Compose configuration has errors"
    docker-compose config
    exit 1
fi

# Check for port conflicts
echo "🔌 Checking for port conflicts..."
PORTS=(3000 3001 8000 9000 9001 9090 9100 27017 6379)
for port in "${PORTS[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "⚠️  Port $port is already in use"
    fi
done

# Check Docker and Docker Compose versions
echo "📋 Checking Docker versions..."
docker --version
docker-compose --version

# Check available disk space
echo "💾 Checking disk space..."
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "${AVAILABLE_SPACE%.*}" -lt 20 ]; then
    echo "⚠️  Warning: Less than 20GB disk space available"
else
    echo "✅ Sufficient disk space available"
fi

# Check available memory
echo "🧠 Checking memory..."
AVAILABLE_MEMORY=$(free -g | awk 'NR==2{printf "%.0f", $7}')
if [ "$AVAILABLE_MEMORY" -lt 8 ]; then
    echo "⚠️  Warning: Less than 8GB memory available"
else
    echo "✅ Sufficient memory available"
fi

echo ""
echo "🎉 Validation completed!"
echo ""
echo "To start the platform, run:"
echo "  ./scripts/setup.sh"
echo "  ./scripts/start.sh"
