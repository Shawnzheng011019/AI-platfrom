#!/bin/bash

# Fix Docker network issues script
echo "Fixing Docker network connectivity issues..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Method 1: Configure Docker daemon with mirrors
echo "Configuring Docker daemon with registry mirrors..."

# Create Docker daemon config directory if it doesn't exist
sudo mkdir -p /etc/docker

# Backup existing daemon.json if it exists
if [ -f /etc/docker/daemon.json ]; then
    sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.backup
    echo "Backed up existing daemon.json"
fi

# Create new daemon.json with mirrors
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ],
  "dns": ["8.8.8.8", "8.8.4.4"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

echo "Docker daemon configuration updated."

# Method 2: Restart Docker service
echo "Restarting Docker service..."
if command -v systemctl > /dev/null; then
    sudo systemctl restart docker
elif command -v service > /dev/null; then
    sudo service docker restart
else
    echo "Please restart Docker manually"
fi

# Wait for Docker to restart
sleep 5

# Method 3: Test connectivity
echo "Testing Docker connectivity..."
if docker pull hello-world > /dev/null 2>&1; then
    echo "✅ Docker connectivity test passed!"
    docker rmi hello-world > /dev/null 2>&1
else
    echo "❌ Docker connectivity test failed. Trying alternative solutions..."
    
    # Method 4: Clear Docker cache
    echo "Clearing Docker cache..."
    docker system prune -f
    
    # Method 5: Try pulling with different registry
    echo "Trying to pull Python image from Alibaba Cloud registry..."
    if docker pull registry.cn-hangzhou.aliyuncs.com/library/python:3.11-slim; then
        echo "✅ Successfully pulled Python image from Alibaba Cloud registry!"
        # Tag it as the original name for compatibility
        docker tag registry.cn-hangzhou.aliyuncs.com/library/python:3.11-slim python:3.11-slim
    else
        echo "❌ Failed to pull from alternative registry as well."
        echo "Please check your internet connection and try again."
        exit 1
    fi
fi

echo "Docker network fix completed!"
echo "You can now try building your containers again."
