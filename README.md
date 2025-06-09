# AI Model Training Platform

A comprehensive AI model training platform supporting various types of machine learning models including LLMs, diffusion models, NLP, and computer vision models.

## Features

### Core Functionality
- **Model Training Management**
  - Large Language Model Fine-tuning (LLM Fine-tuning)
  - Diffusion Model Training
  - NLP Model Training (Classification, NER, etc.)
  - Computer Vision Model Training (Image Classification, Object Detection, etc.)

- **Data Management**
  - Data upload and preprocessing
  - Dataset version management
  - Data annotation and validation

- **Training Pipeline**
  - Pre-configured training scripts
  - Training parameter configuration interface
  - Training progress monitoring
  - Training result visualization

- **Resource Management**
  - Local GPU resource monitoring
  - Training task queuing and scheduling
  - Resource usage statistics

- **Model Management**
  - Model version control
  - Model evaluation and comparison
  - Model export and deployment

## Technology Stack

### Backend
- **FastAPI**: High-performance async API framework
- **PyTorch/TensorFlow**: Core deep learning frameworks
- **Hugging Face Transformers**: LLM and NLP models
- **Diffusers**: Diffusion model support

### Frontend
- **React**: Responsive user interface
- **Ant Design**: UI component library

### Task Scheduling
- **Celery**: Distributed task queue
- **Redis**: Message broker and cache

### Storage
- **MongoDB**: Metadata storage
- **MinIO**: Model and dataset storage

### Monitoring & Logging
- **Prometheus + Grafana**: System monitoring
- **ELK Stack**: Log management

## Project Structure

```
ai-platform/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Core functionality
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic
│   │   └── training/       # Training engines
│   ├── requirements.txt
│   └── main.py
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   └── utils/          # Utilities
│   ├── package.json
│   └── public/
├── training-scripts/       # Pre-configured training scripts
├── docker/                 # Docker configurations
├── docs/                   # Documentation
└── scripts/               # Deployment and utility scripts
```

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Docker & Docker Compose
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-platform
```

2. Run the setup script:
```bash
./scripts/setup.sh
```

3. Start the platform:
```bash
./scripts/start.sh
```

4. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- MinIO Console: http://localhost:9001

### First Steps

1. **Register an account** at http://localhost:3000
2. **Upload a dataset** in the Datasets section
3. **Create a training job** in the Training section
4. **Monitor progress** and view results
5. **Manage your models** in the Models section

## Development

### Backend Development
The backend is built with FastAPI and follows a modular architecture:
- API routes are defined in `backend/app/api/`
- Business logic is in `backend/app/services/`
- Training engines are in `backend/app/training/`

### Frontend Development
The frontend is built with React and Ant Design:
- Components are in `frontend/src/components/`
- Pages are in `frontend/src/pages/`
- API integration is in `frontend/src/services/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
