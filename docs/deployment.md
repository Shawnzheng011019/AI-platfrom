# AI Training Platform - Deployment Guide

## 🚀 Production Deployment

### Prerequisites

- **Docker** (version 20.0+)
- **Docker Compose** (version 2.0+)
- **NVIDIA Docker** (for GPU support)
- **Domain name** (for production)
- **SSL Certificate** (recommended)

### Environment Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai-platform
```

2. **Create production environment files**
```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

3. **Configure environment variables**

Edit `backend/.env`:
```bash
# Application Settings
APP_NAME=AI Training Platform
VERSION=1.0.0
DEBUG=false

# Security - IMPORTANT: Change these in production
SECRET_KEY=your-very-secure-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
MONGODB_URL=mongodb://admin:your-secure-password@mongodb:27017/ai_platform?authSource=admin
DATABASE_NAME=ai_platform

# Redis
REDIS_URL=redis://redis:6379

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=your-minio-access-key
MINIO_SECRET_KEY=your-minio-secret-key
MINIO_SECURE=false

# Training
MAX_CONCURRENT_TRAININGS=4
TRAINING_TIMEOUT_HOURS=48
```

Edit `frontend/.env`:
```bash
REACT_APP_API_URL=https://your-domain.com/api/v1
REACT_APP_APP_NAME=AI Training Platform
REACT_APP_VERSION=1.0.0
```

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  # MongoDB Database
  mongodb:
    image: mongo:7.0
    container_name: ai-platform-mongodb-prod
    restart: unless-stopped
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: ${MONGODB_PASSWORD}
      MONGO_INITDB_DATABASE: ai_platform
    volumes:
      - mongodb_data:/data/db
      - ./docker/mongodb/init-mongo.js:/docker-entrypoint-initdb.d/init-mongo.js:ro
    networks:
      - ai-platform-network

  # Redis Cache and Message Broker
  redis:
    image: redis:7.2-alpine
    container_name: ai-platform-redis-prod
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - ai-platform-network

  # MinIO Object Storage
  minio:
    image: minio/minio:latest
    container_name: ai-platform-minio-prod
    restart: unless-stopped
    environment:
      MINIO_ROOT_USER: ${MINIO_ACCESS_KEY}
      MINIO_ROOT_PASSWORD: ${MINIO_SECRET_KEY}
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    networks:
      - ai-platform-network

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    container_name: ai-platform-backend-prod
    restart: unless-stopped
    environment:
      - MONGODB_URL=mongodb://admin:${MONGODB_PASSWORD}@mongodb:27017/ai_platform?authSource=admin
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/models
      - ./datasets:/app/datasets
    depends_on:
      - mongodb
      - redis
      - minio
    networks:
      - ai-platform-network

  # Celery Worker
  celery-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    container_name: ai-platform-celery-worker-prod
    restart: unless-stopped
    command: celery -A app.core.celery worker --loglevel=info --concurrency=2
    environment:
      - MONGODB_URL=mongodb://admin:${MONGODB_PASSWORD}@mongodb:27017/ai_platform?authSource=admin
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/models
      - ./datasets:/app/datasets
    depends_on:
      - mongodb
      - redis
      - minio
    networks:
      - ai-platform-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    container_name: ai-platform-frontend-prod
    restart: unless-stopped
    depends_on:
      - backend
    networks:
      - ai-platform-network

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: ai-platform-nginx-prod
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend
    networks:
      - ai-platform-network

volumes:
  mongodb_data:
  redis_data:
  minio_data:

networks:
  ai-platform-network:
    driver: bridge
```

### SSL Configuration

1. **Obtain SSL certificates** (using Let's Encrypt):
```bash
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com
```

2. **Create Nginx configuration** (`nginx/nginx.conf`):
```nginx
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;

        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### Deployment Steps

1. **Set environment variables**:
```bash
export MONGODB_PASSWORD=your-secure-mongodb-password
export REDIS_PASSWORD=your-secure-redis-password
export MINIO_ACCESS_KEY=your-minio-access-key
export MINIO_SECRET_KEY=your-minio-secret-key
export SECRET_KEY=your-very-secure-secret-key
```

2. **Deploy the application**:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

3. **Verify deployment**:
```bash
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs
```

### Monitoring and Maintenance

1. **View logs**:
```bash
docker-compose -f docker-compose.prod.yml logs -f [service_name]
```

2. **Update application**:
```bash
git pull
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

3. **Backup data**:
```bash
# Backup MongoDB
docker exec ai-platform-mongodb-prod mongodump --out /backup

# Backup MinIO data
docker exec ai-platform-minio-prod mc mirror /data /backup/minio
```

### Security Considerations

1. **Change default passwords**
2. **Use strong SSL certificates**
3. **Configure firewall rules**
4. **Regular security updates**
5. **Monitor access logs**
6. **Implement rate limiting**

### Performance Optimization

1. **Resource allocation**:
   - Allocate sufficient RAM (minimum 8GB)
   - Use SSD storage for better I/O performance
   - Configure GPU resources for training

2. **Database optimization**:
   - Configure MongoDB indexes
   - Set appropriate connection pool sizes
   - Monitor query performance

3. **Caching**:
   - Configure Redis for optimal performance
   - Implement application-level caching

### Troubleshooting

Common issues and solutions:

1. **Service won't start**: Check logs and environment variables
2. **Database connection issues**: Verify MongoDB credentials and network
3. **Training jobs fail**: Check GPU availability and resource limits
4. **High memory usage**: Monitor and adjust container resource limits

For more detailed troubleshooting, see the [troubleshooting guide](troubleshooting.md).
