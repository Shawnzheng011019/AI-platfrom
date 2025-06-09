#!/bin/bash

# AI Training Platform Stop Script

set -e

echo "🛑 Stopping AI Training Platform..."

# Stop all services
docker-compose down

echo "✅ AI Training Platform stopped successfully!"
echo ""
echo "To start the platform again, run:"
echo "  ./scripts/start.sh"
echo ""
echo "To completely remove all data (including databases), run:"
echo "  docker-compose down -v"
