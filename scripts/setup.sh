#!/bin/bash

# AI Training Platform Setup Script

set -e

echo "🚀 Setting up AI Training Platform..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads models datasets logs

# Set permissions
chmod 755 uploads models datasets logs

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Application Settings
APP_NAME=AI Training Platform
VERSION=1.0.0
DEBUG=false

# API Settings
API_V1_PREFIX=/api/v1
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))")
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
MONGODB_URL=mongodb://admin:password123@localhost:27017/ai_platform?authSource=admin
DATABASE_NAME=ai_platform

# Redis
REDIS_URL=redis://localhost:6379

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_SECURE=false

# Storage
UPLOAD_DIR=./uploads
MODELS_DIR=./models
DATASETS_DIR=./datasets

# Training
MAX_CONCURRENT_TRAININGS=2
TRAINING_TIMEOUT_HOURS=24

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8001
EOF
    echo "✅ .env file created"
fi

# Pull Docker images
echo "🐳 Pulling Docker images..."
docker-compose pull

# Build custom images
echo "🔨 Building custom images..."
docker-compose build

echo "✅ Setup completed successfully!"
echo ""
echo "To start the platform, run:"
echo "  ./scripts/start.sh"
echo ""
echo "To stop the platform, run:"
echo "  ./scripts/stop.sh"
