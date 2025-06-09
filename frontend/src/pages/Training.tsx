import React, { useState, useEffect } from 'react';
import {
  Typography, Card, Row, Col, Button, Table, Tag, Progress,
  Modal, Form, Input, Select, Space, message, Popconfirm
} from 'antd';
import {
  PlusOutlined, PlayCircleOutlined, PauseCircleOutlined,
  StopOutlined, EyeOutlined, DeleteOutlined
} from '@ant-design/icons';
import api from '../services/api';

const { Title } = Typography;
const { Option } = Select;
const { TextArea } = Input;

interface TrainingJob {
  id: string;
  name: string;
  description?: string;
  config: {
    model_type: string;
    model_name: string;
    learning_rate: number;
    batch_size: number;
    num_epochs: number;
  };
  dataset_id: string;
  status: string;
  progress: number;
  current_epoch: number;
  current_step: number;
  owner_id: string;
  priority: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

interface Dataset {
  id: string;
  name: string;
  dataset_type: string;
}

const Training: React.FC = () => {
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [form] = Form.useForm();

  const fetchTrainingJobs = async () => {
    setLoading(true);
    try {
      const response = await api.get('/training-jobs/');
      setTrainingJobs(response.data);
    } catch (error) {
      message.error('Failed to fetch training jobs');
    }
    setLoading(false);
  };

  const fetchDatasets = async () => {
    try {
      const response = await api.get('/datasets/');
      setDatasets(response.data);
    } catch (error) {
      message.error('Failed to fetch datasets');
    }
  };

  useEffect(() => {
    fetchTrainingJobs();
    fetchDatasets();
  }, []);

  const handleCreateJob = async (values: any) => {
    try {
      const jobData = {
        name: values.name,
        description: values.description,
        config: {
          model_type: values.model_type,
          model_name: values.model_name,
          learning_rate: parseFloat(values.learning_rate),
          batch_size: parseInt(values.batch_size),
          num_epochs: parseInt(values.num_epochs),
          base_model: values.base_model,
        },
        dataset_id: values.dataset_id,
        priority: values.priority || 0,
      };

      await api.post('/training-jobs/', jobData);
      message.success('Training job created successfully');
      setModalVisible(false);
      form.resetFields();
      fetchTrainingJobs();
    } catch (error) {
      message.error('Failed to create training job');
    }
  };

  const handleStartJob = async (jobId: string) => {
    try {
      await api.post(`/training-jobs/${jobId}/start`);
      message.success('Training job started');
      fetchTrainingJobs();
    } catch (error) {
      message.error('Failed to start training job');
    }
  };

  const handleStopJob = async (jobId: string) => {
    try {
      await api.post(`/training-jobs/${jobId}/stop`);
      message.success('Training job stopped');
      fetchTrainingJobs();
    } catch (error) {
      message.error('Failed to stop training job');
    }
  };

  const handleDeleteJob = async (jobId: string) => {
    try {
      await api.delete(`/training-jobs/${jobId}`);
      message.success('Training job deleted');
      fetchTrainingJobs();
    } catch (error) {
      message.error('Failed to delete training job');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'blue';
      case 'completed': return 'green';
      case 'pending': return 'orange';
      case 'failed': return 'red';
      default: return 'default';
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: TrainingJob) => (
        <Space direction="vertical" size={0}>
          <strong>{text}</strong>
          {record.description && (
            <span style={{ color: '#666', fontSize: '12px' }}>
              {record.description}
            </span>
          )}
        </Space>
      ),
    },
    {
      title: 'Model Type',
      dataIndex: ['config', 'model_type'],
      key: 'model_type',
      render: (type: string) => <Tag color="blue">{type?.toUpperCase()}</Tag>,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>{status.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'Progress',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: TrainingJob) => (
        <Space direction="vertical" size={0}>
          <Progress percent={progress} size="small" />
          <span style={{ fontSize: '12px', color: '#666' }}>
            Epoch {record.current_epoch} / Step {record.current_step}
          </span>
        </Space>
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleDateString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: TrainingJob) => (
        <Space>
          <Button
            type="text"
            icon={<EyeOutlined />}
            onClick={() => {/* View job details */}}
          />
          {record.status === 'running' && (
            <Button
              type="text"
              icon={<StopOutlined />}
              onClick={() => handleStopJob(record.id)}
            />
          )}
          {record.status === 'pending' && (
            <Button
              type="text"
              icon={<PlayCircleOutlined />}
              onClick={() => handleStartJob(record.id)}
            />
          )}
          <Popconfirm
            title="Are you sure you want to delete this training job?"
            onConfirm={() => handleDeleteJob(record.id)}
            okText="Yes"
            cancelText="No"
          >
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
            />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Title level={2}>Training Jobs</Title>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setModalVisible(true)}
        >
          New Training Job
        </Button>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <div className="metric-card">
              <div className="metric-value">3</div>
              <div className="metric-label">Active Jobs</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div className="metric-card">
              <div className="metric-value">12</div>
              <div className="metric-label">Completed</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div className="metric-card">
              <div className="metric-value">75%</div>
              <div className="metric-label">GPU Usage</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div className="metric-card">
              <div className="metric-value">2.5h</div>
              <div className="metric-label">Avg Duration</div>
            </div>
          </Card>
        </Col>
      </Row>

      <Card>
        <Table
          columns={columns}
          dataSource={trainingJobs}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
          }}
        />
      </Card>

      <Modal
        title="Create Training Job"
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={800}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateJob}
        >
          <Form.Item
            name="name"
            label="Job Name"
            rules={[{ required: true, message: 'Please enter job name' }]}
          >
            <Input placeholder="Enter training job name" />
          </Form.Item>

          <Form.Item
            name="description"
            label="Description"
          >
            <TextArea rows={3} placeholder="Enter job description" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="model_type"
                label="Model Type"
                rules={[{ required: true, message: 'Please select model type' }]}
              >
                <Select placeholder="Select model type">
                  <Option value="llm">Large Language Model</Option>
                  <Option value="diffusion">Diffusion Model</Option>
                  <Option value="nlp_classification">NLP Classification</Option>
                  <Option value="nlp_ner">NLP Named Entity Recognition</Option>
                  <Option value="cv_classification">Computer Vision Classification</Option>
                  <Option value="cv_detection">Object Detection</Option>
                  <Option value="time_series">Time Series Forecasting</Option>
                  <Option value="recommendation">Recommendation System</Option>
                  <Option value="reinforcement_learning">Reinforcement Learning</Option>
                  <Option value="speech_recognition">Speech Recognition</Option>
                  <Option value="multimodal">Multimodal (Vision-Language)</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="dataset_id"
                label="Dataset"
                rules={[{ required: true, message: 'Please select dataset' }]}
              >
                <Select placeholder="Select dataset">
                  {datasets.map(dataset => (
                    <Option key={dataset.id} value={dataset.id}>
                      {dataset.name} ({dataset.dataset_type})
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="model_name"
                label="Model Name"
                rules={[{ required: true, message: 'Please enter model name' }]}
              >
                <Input placeholder="e.g., bert-base-uncased" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="base_model"
                label="Base Model"
              >
                <Input placeholder="e.g., bert-base-uncased" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="learning_rate"
                label="Learning Rate"
                initialValue="2e-5"
                rules={[{ required: true, message: 'Please enter learning rate' }]}
              >
                <Input placeholder="2e-5" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="batch_size"
                label="Batch Size"
                initialValue="16"
                rules={[{ required: true, message: 'Please enter batch size' }]}
              >
                <Input placeholder="16" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="num_epochs"
                label="Number of Epochs"
                initialValue="3"
                rules={[{ required: true, message: 'Please enter number of epochs' }]}
              >
                <Input placeholder="3" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="priority"
            label="Priority"
            initialValue={0}
          >
            <Select>
              <Option value={0}>Normal</Option>
              <Option value={1}>High</Option>
              <Option value={-1}>Low</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Create Training Job
              </Button>
              <Button onClick={() => {
                setModalVisible(false);
                form.resetFields();
              }}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default Training;
