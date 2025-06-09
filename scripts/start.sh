#!/bin/bash

# AI Training Platform Start Script

set -e

echo "🚀 Starting AI Training Platform..."

# Check if setup has been run
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run ./scripts/setup.sh first."
    exit 1
fi

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if MongoDB is ready
echo "🔍 Checking MongoDB..."
until docker-compose exec -T mongodb mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
    echo "Waiting for MongoDB..."
    sleep 2
done
echo "✅ MongoDB is ready"

# Check if Redis is ready
echo "🔍 Checking Redis..."
until docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; do
    echo "Waiting for Redis..."
    sleep 2
done
echo "✅ Redis is ready"

# Check if MinIO is ready
echo "🔍 Checking MinIO..."
until curl -f http://localhost:9000/minio/health/live > /dev/null 2>&1; do
    echo "Waiting for MinIO..."
    sleep 2
done
echo "✅ MinIO is ready"

# Check if Backend is ready
echo "🔍 Checking Backend API..."
until curl -f http://localhost:8000/health > /dev/null 2>&1; do
    echo "Waiting for Backend API..."
    sleep 2
done
echo "✅ Backend API is ready"

# Check if Frontend is ready
echo "🔍 Checking Frontend..."
until curl -f http://localhost:3000 > /dev/null 2>&1; do
    echo "Waiting for Frontend..."
    sleep 2
done
echo "✅ Frontend is ready"

# Check if Celery worker is running
echo "🔍 Checking Celery Worker..."
sleep 5
if docker-compose ps celery-worker | grep -q "Up"; then
    echo "✅ Celery Worker is running"
else
    echo "⚠️  Celery Worker may not be running properly"
fi

echo ""
echo "🎉 AI Training Platform is now running!"
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🗄️  MinIO Console: http://localhost:9001"
echo ""
echo "Default MinIO credentials:"
echo "  Username: minioadmin"
echo "  Password: minioadmin123"
echo ""
echo "To view logs, run:"
echo "  docker-compose logs -f [service_name]"
echo ""
echo "To stop the platform, run:"
echo "  ./scripts/stop.sh"
