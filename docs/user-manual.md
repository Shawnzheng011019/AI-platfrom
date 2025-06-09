# AI Training Platform - User Manual

## 📖 Complete User Guide

### Getting Started

#### 1. Account Registration and Login

**Registration:**
1. Navigate to the platform URL
2. Click on the "Register" tab
3. Fill in your details:
   - Username (unique identifier)
   - Email address
   - Full name (optional)
   - Password (minimum 6 characters)
4. Click "Register" to create your account

**Login:**
1. Enter your username and password
2. Click "Login" to access the platform

#### 2. Dashboard Overview

The dashboard provides an overview of your AI training activities:

- **Total Datasets**: Number of datasets you've uploaded
- **Active Trainings**: Currently running training jobs
- **Total Models**: Models you've created or have access to
- **GPU Usage**: Current GPU utilization
- **Training Activity Chart**: Visual representation of recent training activity
- **System Status**: Real-time system resource monitoring

### Managing Datasets

#### Uploading Datasets

1. **Navigate to Datasets**:
   - Click "Datasets" in the sidebar

2. **Upload New Dataset**:
   - Click "Upload Dataset" button
   - Fill in dataset information:
     - **Name**: Descriptive name for your dataset
     - **Description**: Optional description
     - **Dataset Type**: Choose from Text, Image, Audio, Video, or Tabular
     - **Format**: Select the file format (CSV, JSON, JSONL, etc.)
     - **Tags**: Add relevant tags for organization
     - **File**: Select your dataset file

3. **Dataset Processing**:
   - After upload, the system automatically processes your dataset
   - Processing includes metadata extraction and validation
   - Status will change from "Processing" to "Ready" when complete

#### Dataset Management

- **View Datasets**: Browse all your datasets in the table view
- **Filter**: Use filters to find specific datasets by type or status
- **Edit**: Update dataset information (name, description, tags)
- **Delete**: Remove datasets you no longer need
- **Download**: Access your original dataset files

#### Supported Dataset Formats

**Text Data:**
- CSV files with text and label columns
- JSON/JSONL files with structured text data
- Plain text files for language modeling

**Image Data:**
- ZIP archives containing organized image folders
- Individual image files (PNG, JPG, JPEG)
- Structured datasets with image paths and labels

**Tabular Data:**
- CSV files with numerical and categorical features
- Parquet files for large datasets
- Excel files (converted to CSV)

### Creating Training Jobs

#### 1. Starting a New Training Job

1. **Navigate to Training**:
   - Click "Training" in the sidebar

2. **Create New Job**:
   - Click "New Training Job" button
   - Fill in the training configuration:

#### 2. Training Configuration

**Basic Information:**
- **Job Name**: Descriptive name for your training job
- **Description**: Optional detailed description
- **Dataset**: Select from your available datasets
- **Priority**: Set job priority (Normal, High, Low)

**Model Configuration:**
- **Model Type**: Choose the type of model to train:
  - **Large Language Model (LLM)**: For text generation and understanding
  - **Diffusion Model**: For image generation
  - **NLP Classification**: For text classification tasks
  - **NLP NER**: For named entity recognition
  - **CV Classification**: For image classification
  - **Object Detection**: For detecting objects in images

- **Model Name**: Specify the model architecture
- **Base Model**: Pre-trained model to start from (optional)

**Training Parameters:**
- **Learning Rate**: Controls how fast the model learns (e.g., 2e-5)
- **Batch Size**: Number of samples processed together (e.g., 16)
- **Number of Epochs**: How many times to go through the entire dataset (e.g., 3)

#### 3. Monitoring Training Progress

**Training Status:**
- **Pending**: Job is queued and waiting to start
- **Running**: Training is currently in progress
- **Completed**: Training finished successfully
- **Failed**: Training encountered an error
- **Cancelled**: Training was manually stopped

**Progress Tracking:**
- Real-time progress bar showing completion percentage
- Current epoch and step information
- Live training logs and metrics
- Resource usage monitoring

#### 4. Managing Training Jobs

**Job Actions:**
- **Start**: Begin a pending training job
- **Stop**: Halt a running training job
- **View Details**: See comprehensive job information
- **Delete**: Remove completed or failed jobs

### Working with Models

#### 1. Model Overview

After successful training, models are automatically created and available in the Models section.

#### 2. Model Management

**Model Information:**
- **Name**: Model identifier
- **Type**: Model category (LLM, CV, NLP, etc.)
- **Framework**: Training framework used (PyTorch, TensorFlow)
- **Status**: Current model state
- **Accuracy**: Model performance metrics
- **Size**: Model file size
- **Creation Date**: When the model was created

**Model Actions:**
- **View**: See detailed model information and metrics
- **Download**: Download model files for local use
- **Deploy**: Deploy model for inference (if supported)
- **Delete**: Remove models you no longer need

#### 3. Model Deployment

1. **Deploy Model**:
   - Click the deploy button for a ready model
   - Model will be made available via API endpoint
   - Status changes to "Deployed"

2. **Using Deployed Models**:
   - Access via provided API endpoint
   - Send inference requests
   - Monitor usage and performance

#### 4. Model Evaluation

- **Performance Metrics**: View accuracy, precision, recall, F1-score
- **Comparison**: Compare multiple models side by side
- **Validation Results**: See how models perform on test data

### System Monitoring

#### Resource Usage

Monitor system resources to optimize training:

- **CPU Usage**: Current processor utilization
- **Memory Usage**: RAM consumption
- **GPU Usage**: Graphics card utilization
- **Storage Usage**: Disk space consumption

#### Training Resources

- **Active Training Jobs**: Currently running training processes
- **Resource Allocation**: How resources are distributed
- **Queue Status**: Pending jobs waiting for resources

### Best Practices

#### Dataset Preparation

1. **Data Quality**:
   - Ensure clean, well-formatted data
   - Remove duplicates and inconsistencies
   - Validate data types and formats

2. **Data Size**:
   - Larger datasets generally produce better models
   - Ensure balanced class distributions
   - Consider data augmentation for small datasets

3. **Data Organization**:
   - Use clear, descriptive filenames
   - Organize data in logical folder structures
   - Include metadata and documentation

#### Training Optimization

1. **Parameter Selection**:
   - Start with recommended default parameters
   - Adjust learning rate based on training progress
   - Increase batch size if you have sufficient memory

2. **Resource Management**:
   - Monitor GPU memory usage
   - Use mixed precision training for efficiency
   - Schedule long training jobs during off-peak hours

3. **Experiment Tracking**:
   - Use descriptive names for training jobs
   - Document parameter choices and results
   - Keep track of successful configurations

#### Model Management

1. **Version Control**:
   - Use clear naming conventions
   - Keep track of model versions
   - Document model improvements

2. **Performance Monitoring**:
   - Regularly evaluate model performance
   - Compare new models with baselines
   - Monitor for performance degradation

### Troubleshooting

#### Common Issues

1. **Training Job Fails**:
   - Check dataset format and quality
   - Verify sufficient system resources
   - Review training logs for error messages

2. **Slow Training**:
   - Reduce batch size if memory is limited
   - Check GPU utilization
   - Consider using mixed precision training

3. **Poor Model Performance**:
   - Increase training data size
   - Adjust learning rate and other hyperparameters
   - Try different model architectures

4. **Upload Issues**:
   - Check file format compatibility
   - Verify file size limits
   - Ensure stable internet connection

#### Getting Help

- **Documentation**: Refer to this user manual and API documentation
- **Logs**: Check training logs for detailed error information
- **Support**: Contact system administrators for technical issues
- **Community**: Share experiences with other users

### Advanced Features

#### Custom Training Scripts

For advanced users, the platform supports custom training scripts:

1. **Script Requirements**:
   - Must accept configuration via JSON file
   - Should output progress and metrics
   - Must save models in specified format

2. **Integration**:
   - Place scripts in the training-scripts directory
   - Update model type mappings
   - Test thoroughly before production use

#### API Access

The platform provides REST API access for programmatic interaction:

- **Authentication**: Use JWT tokens for API access
- **Endpoints**: Full CRUD operations for all resources
- **Documentation**: Available at `/docs` endpoint

#### Batch Operations

For processing multiple datasets or training jobs:

- **Bulk Upload**: Upload multiple datasets simultaneously
- **Batch Training**: Queue multiple training jobs
- **Automated Workflows**: Set up automated training pipelines

This completes the comprehensive user manual for the AI Training Platform. For additional support or advanced configurations, please refer to the technical documentation or contact your system administrator.
