# AI Training Platform - Development Guide

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+ (for backend development)
- Node.js 16+ (for frontend development)
- Docker & Docker Compose (for services)
- Git

### Local Development Setup

#### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-platform
```

#### 2. Backend Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Start services (MongoDB, Redis, MinIO)
docker-compose up -d mongodb redis minio

# Run backend
python main.py
```

#### 3. Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm start
```

#### 4. Celery Worker (Optional for training)

```bash
cd backend

# Start Celery worker
celery -A app.core.celery worker --loglevel=info
```

## 🏗️ Architecture Overview

### Backend Architecture

```
backend/
├── app/
│   ├── api/              # API route handlers
│   ├── core/             # Core configuration and utilities
│   ├── models/           # Database models (Beanie/MongoDB)
│   ├── services/         # Business logic layer
│   ├── tasks/            # Celery background tasks
│   └── training/         # Training engine components
├── training-scripts/     # Pre-built training scripts
├── tests/               # Unit and integration tests
└── main.py             # FastAPI application entry point
```

### Frontend Architecture

```
frontend/src/
├── components/          # Reusable React components
├── pages/              # Page-level components
├── services/           # API integration and utilities
├── utils/              # Helper functions
└── App.tsx            # Main application component
```

### Key Technologies

**Backend:**
- **FastAPI**: Modern, fast web framework for building APIs
- **Beanie**: Async MongoDB ODM based on Pydantic
- **Celery**: Distributed task queue for background jobs
- **Redis**: Message broker and caching
- **PyTorch/TensorFlow**: Machine learning frameworks

**Frontend:**
- **React**: UI library with TypeScript
- **Ant Design**: Professional UI component library
- **Axios**: HTTP client for API calls
- **Recharts**: Charting library for data visualization

## 🔧 Development Workflow

### Adding New Features

#### 1. Backend API Development

**Step 1: Create Model**
```python
# app/models/new_feature.py
from beanie import Document
from pydantic import BaseModel
from datetime import datetime

class NewFeature(Document):
    name: str
    description: str
    created_at: datetime = datetime.utcnow()
    
    class Settings:
        collection = "new_features"

class NewFeatureCreate(BaseModel):
    name: str
    description: str

class NewFeatureResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
```

**Step 2: Create Service**
```python
# app/services/new_feature_service.py
from typing import List, Optional
from app.models.new_feature import NewFeature, NewFeatureCreate

class NewFeatureService:
    async def create_feature(self, feature_data: NewFeatureCreate) -> NewFeature:
        feature = NewFeature(**feature_data.dict())
        await feature.insert()
        return feature
    
    async def list_features(self) -> List[NewFeature]:
        return await NewFeature.find().to_list()
```

**Step 3: Create API Routes**
```python
# app/api/new_feature.py
from fastapi import APIRouter, Depends
from app.models.new_feature import NewFeatureCreate, NewFeatureResponse
from app.services.new_feature_service import NewFeatureService

router = APIRouter()

@router.post("/", response_model=NewFeatureResponse)
async def create_feature(feature_data: NewFeatureCreate):
    service = NewFeatureService()
    feature = await service.create_feature(feature_data)
    return NewFeatureResponse(
        id=str(feature.id),
        name=feature.name,
        description=feature.description,
        created_at=feature.created_at
    )
```

**Step 4: Register Routes**
```python
# main.py
from app.api import new_feature

app.include_router(
    new_feature.router, 
    prefix=f"{settings.api_v1_prefix}/new-feature", 
    tags=["new-feature"]
)
```

#### 2. Frontend Development

**Step 1: Create API Service**
```typescript
// src/services/newFeatureApi.ts
import api from './api';

export interface NewFeature {
  id: string;
  name: string;
  description: string;
  created_at: string;
}

export const newFeatureApi = {
  create: (data: { name: string; description: string }) =>
    api.post<NewFeature>('/new-feature/', data),
  
  list: () =>
    api.get<NewFeature[]>('/new-feature/'),
};
```

**Step 2: Create Component**
```typescript
// src/pages/NewFeature.tsx
import React, { useState, useEffect } from 'react';
import { Table, Button, Modal, Form, Input } from 'antd';
import { newFeatureApi, NewFeature } from '../services/newFeatureApi';

const NewFeaturePage: React.FC = () => {
  const [features, setFeatures] = useState<NewFeature[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchFeatures = async () => {
    setLoading(true);
    try {
      const response = await newFeatureApi.list();
      setFeatures(response.data);
    } catch (error) {
      console.error('Failed to fetch features:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchFeatures();
  }, []);

  return (
    <div>
      <Table
        dataSource={features}
        loading={loading}
        rowKey="id"
      />
    </div>
  );
};

export default NewFeaturePage;
```

### Adding Training Scripts

#### 1. Create Training Script

```python
# training-scripts/new_model_training.py
#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any

class NewModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def load_data(self):
        # Implement data loading logic
        pass
    
    def build_model(self):
        # Implement model building logic
        pass
    
    def train(self):
        # Implement training logic
        pass

def main():
    parser = argparse.ArgumentParser(description='Train new model')
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    trainer = NewModelTrainer(config)
    trainer.load_data()
    trainer.build_model()
    trainer.train()

if __name__ == "__main__":
    main()
```

#### 2. Register Training Script

```python
# app/services/training_service.py
def _get_training_script(self, model_type: str) -> str:
    script_mapping = {
        'llm': 'training-scripts/llm_fine_tuning.py',
        'new_model': 'training-scripts/new_model_training.py',  # Add new script
        # ... other scripts
    }
    # ... rest of the method
```

## 🧪 Testing

### Backend Testing

```bash
cd backend

# Run all tests
pytest

# Run specific test file
pytest tests/test_auth.py

# Run with coverage
pytest --cov=app tests/
```

### Frontend Testing

```bash
cd frontend

# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

### Integration Testing

```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/

# Cleanup
docker-compose -f docker-compose.test.yml down
```

## 📝 Code Style and Standards

### Backend Standards

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations for all functions
- **Docstrings**: Document all public methods
- **Error Handling**: Use appropriate exception handling

```python
# Good example
async def create_user(self, user_data: UserCreate) -> User:
    """
    Create a new user account.
    
    Args:
        user_data: User creation data
        
    Returns:
        Created user instance
        
    Raises:
        ValueError: If user already exists
    """
    existing_user = await User.find_one(User.email == user_data.email)
    if existing_user:
        raise ValueError("User already exists")
    
    user = User(**user_data.dict())
    await user.insert()
    return user
```

### Frontend Standards

- **TypeScript**: Use strict type checking
- **Component Structure**: Follow React best practices
- **Naming**: Use descriptive component and variable names
- **Error Handling**: Implement proper error boundaries

```typescript
// Good example
interface UserListProps {
  users: User[];
  onUserSelect: (user: User) => void;
  loading?: boolean;
}

const UserList: React.FC<UserListProps> = ({ 
  users, 
  onUserSelect, 
  loading = false 
}) => {
  return (
    <Table
      dataSource={users}
      loading={loading}
      onRow={(user) => ({
        onClick: () => onUserSelect(user),
      })}
    />
  );
};
```

## 🚀 Deployment

### Development Deployment

```bash
# Build and start all services
docker-compose up --build

# Start specific services
docker-compose up backend frontend
```

### Production Deployment

See [deployment.md](deployment.md) for detailed production deployment instructions.

## 🐛 Debugging

### Backend Debugging

1. **Enable Debug Mode**:
```python
# main.py
app = FastAPI(debug=True)
```

2. **Use Logging**:
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Processing request")
logger.error(f"Error occurred: {error}")
```

3. **Database Debugging**:
```python
# Enable MongoDB query logging
import motor.motor_asyncio
motor.motor_asyncio.AsyncIOMotorClient.get_io_loop = asyncio.get_event_loop
```

### Frontend Debugging

1. **React Developer Tools**: Install browser extension
2. **Console Logging**: Use console.log for debugging
3. **Network Tab**: Monitor API calls in browser dev tools

## 📚 Additional Resources

### Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [Ant Design Components](https://ant.design/components/)
- [MongoDB Documentation](https://docs.mongodb.com/)

### Best Practices

- **Security**: Always validate input data
- **Performance**: Use async/await for I/O operations
- **Scalability**: Design for horizontal scaling
- **Monitoring**: Implement comprehensive logging

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

For questions or support, please refer to the project documentation or contact the development team.
