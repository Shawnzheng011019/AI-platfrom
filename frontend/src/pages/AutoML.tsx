import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  Space,
  Tag,
  Progress,
  message,
  Descriptions,
  Tabs,
  Row,
  Col,
  Statistic
} from 'antd';
import {
  PlusOutlined,
  PlayCircleOutlined,
  StopOutlined,
  EyeOutlined,
  DeleteOutlined,
  RobotOutlined
} from '@ant-design/icons';
import api from '../services/api';

const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

interface AutoMLJob {
  id: string;
  name: string;
  description?: string;
  config: {
    model_type: string;
    optimization_type: string;
    optimization_metric: string;
    max_trials: number;
    max_time_minutes: number;
    dataset_id: string;
  };
  status: string;
  progress: number;
  trial_results: any[];
  best_parameters?: any;
  best_score?: number;
  total_trials_completed: number;
  owner_id: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

interface Dataset {
  id: string;
  name: string;
  dataset_type: string;
}

const AutoML: React.FC = () => {
  const [automlJobs, setAutoMLJobs] = useState<AutoMLJob[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedJob, setSelectedJob] = useState<AutoMLJob | null>(null);
  const [form] = Form.useForm();

  const fetchAutoMLJobs = async () => {
    setLoading(true);
    try {
      const response = await api.get('/automl/');
      setAutoMLJobs(response.data);
    } catch (error) {
      message.error('Failed to fetch AutoML jobs');
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
    fetchAutoMLJobs();
    fetchDatasets();
  }, []);

  const handleCreateJob = async (values: any) => {
    try {
      const jobData = {
        name: values.name,
        description: values.description,
        config: {
          name: values.name,
          model_type: values.model_type,
          dataset_id: values.dataset_id,
          optimization_type: values.optimization_type,
          optimization_metric: values.optimization_metric,
          optimization_direction: values.optimization_direction,
          max_trials: values.max_trials,
          max_time_minutes: values.max_time_minutes,
          early_stopping_patience: values.early_stopping_patience,
          max_concurrent_trials: values.max_concurrent_trials,
          use_gpu: values.use_gpu,
          cross_validation_folds: values.cross_validation_folds,
          test_split: values.test_split,
          random_seed: values.random_seed
        }
      };

      await api.post('/automl/', jobData);
      message.success('AutoML job created successfully');
      setModalVisible(false);
      form.resetFields();
      fetchAutoMLJobs();
    } catch (error) {
      message.error('Failed to create AutoML job');
    }
  };

  const handleStopJob = async (jobId: string) => {
    try {
      await api.post(`/automl/${jobId}/stop`);
      message.success('AutoML job stopped');
      fetchAutoMLJobs();
    } catch (error) {
      message.error('Failed to stop AutoML job');
    }
  };

  const handleDeleteJob = async (jobId: string) => {
    try {
      await api.delete(`/automl/${jobId}`);
      message.success('AutoML job deleted');
      fetchAutoMLJobs();
    } catch (error) {
      message.error('Failed to delete AutoML job');
    }
  };

  const handleViewDetails = (job: AutoMLJob) => {
    setSelectedJob(job);
    setDetailModalVisible(true);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'running': return 'processing';
      case 'failed': return 'error';
      case 'cancelled': return 'warning';
      default: return 'default';
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: AutoMLJob) => (
        <Space>
          <RobotOutlined />
          <span>{text}</span>
        </Space>
      ),
    },
    {
      title: 'Model Type',
      dataIndex: ['config', 'model_type'],
      key: 'model_type',
      render: (text: string) => <Tag>{text.toUpperCase()}</Tag>,
    },
    {
      title: 'Optimization',
      dataIndex: ['config', 'optimization_type'],
      key: 'optimization_type',
      render: (text: string) => <Tag color="blue">{text}</Tag>,
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
      render: (progress: number, record: AutoMLJob) => (
        <div style={{ width: 120 }}>
          <Progress 
            percent={Math.round(progress)} 
            size="small" 
            status={record.status === 'failed' ? 'exception' : 'active'}
          />
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.total_trials_completed}/{record.config.max_trials} trials
          </div>
        </div>
      ),
    },
    {
      title: 'Best Score',
      dataIndex: 'best_score',
      key: 'best_score',
      render: (score: number) => score ? score.toFixed(4) : '-',
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
      render: (_, record: AutoMLJob) => (
        <Space>
          <Button
            type="text"
            icon={<EyeOutlined />}
            onClick={() => handleViewDetails(record)}
          />
          {record.status === 'running' && (
            <Button
              type="text"
              icon={<StopOutlined />}
              onClick={() => handleStopJob(record.id)}
            />
          )}
          <Button
            type="text"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteJob(record.id)}
          />
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>AutoML Jobs</h2>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setModalVisible(true)}
        >
          Create AutoML Job
        </Button>
      </div>

      <Card>
        <Table
          columns={columns}
          dataSource={automlJobs}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
          }}
        />
      </Card>

      {/* Create AutoML Job Modal */}
      <Modal
        title="Create AutoML Job"
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
          initialValues={{
            optimization_type: 'hyperparameter',
            optimization_metric: 'accuracy',
            optimization_direction: 'maximize',
            max_trials: 20,
            max_time_minutes: 120,
            early_stopping_patience: 5,
            max_concurrent_trials: 2,
            use_gpu: true,
            cross_validation_folds: 3,
            test_split: 0.2,
            random_seed: 42
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="Job Name"
                rules={[{ required: true, message: 'Please enter job name' }]}
              >
                <Input placeholder="Enter AutoML job name" />
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
                  <Option value="cv_classification">Computer Vision Classification</Option>
                  <Option value="nlp_classification">NLP Classification</Option>
                  <Option value="time_series">Time Series Forecasting</Option>
                  <Option value="recommendation">Recommendation System</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="optimization_type"
                label="Optimization Type"
                rules={[{ required: true, message: 'Please select optimization type' }]}
              >
                <Select>
                  <Option value="hyperparameter">Hyperparameter Optimization</Option>
                  <Option value="architecture">Architecture Search</Option>
                  <Option value="feature_engineering">Feature Engineering</Option>
                  <Option value="full_pipeline">Full Pipeline</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="optimization_metric"
                label="Optimization Metric"
              >
                <Select>
                  <Option value="accuracy">Accuracy</Option>
                  <Option value="f1_score">F1 Score</Option>
                  <Option value="precision">Precision</Option>
                  <Option value="recall">Recall</Option>
                  <Option value="loss">Loss</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="max_trials"
                label="Max Trials"
              >
                <InputNumber min={1} max={100} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="max_time_minutes"
                label="Max Time (minutes)"
              >
                <InputNumber min={10} max={1440} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="use_gpu"
                label="Use GPU"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="max_concurrent_trials"
                label="Max Concurrent Trials"
              >
                <InputNumber min={1} max={5} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Create AutoML Job
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

      {/* Job Details Modal */}
      <Modal
        title={`AutoML Job: ${selectedJob?.name}`}
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={1000}
      >
        {selectedJob && (
          <Tabs defaultActiveKey="overview">
            <TabPane tab="Overview" key="overview">
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic title="Status" value={selectedJob.status.toUpperCase()} />
                </Col>
                <Col span={8}>
                  <Statistic title="Progress" value={selectedJob.progress} suffix="%" />
                </Col>
                <Col span={8}>
                  <Statistic title="Best Score" value={selectedJob.best_score?.toFixed(4) || 'N/A'} />
                </Col>
              </Row>
              
              <Descriptions title="Configuration" bordered style={{ marginTop: 16 }}>
                <Descriptions.Item label="Model Type">{selectedJob.config.model_type}</Descriptions.Item>
                <Descriptions.Item label="Optimization Type">{selectedJob.config.optimization_type}</Descriptions.Item>
                <Descriptions.Item label="Optimization Metric">{selectedJob.config.optimization_metric}</Descriptions.Item>
                <Descriptions.Item label="Max Trials">{selectedJob.config.max_trials}</Descriptions.Item>
                <Descriptions.Item label="Max Time">{selectedJob.config.max_time_minutes} minutes</Descriptions.Item>
                <Descriptions.Item label="Completed Trials">{selectedJob.total_trials_completed}</Descriptions.Item>
              </Descriptions>
            </TabPane>
            
            <TabPane tab="Trial Results" key="trials">
              <Table
                dataSource={selectedJob.trial_results}
                columns={[
                  { title: 'Trial ID', dataIndex: 'trial_id', key: 'trial_id' },
                  { title: 'Score', dataIndex: 'score', key: 'score', render: (score: number) => score.toFixed(4) },
                  { title: 'Parameters', dataIndex: 'parameters', key: 'parameters', render: (params: any) => JSON.stringify(params) },
                ]}
                pagination={{ pageSize: 5 }}
                size="small"
              />
            </TabPane>
            
            <TabPane tab="Best Parameters" key="parameters">
              {selectedJob.best_parameters ? (
                <pre style={{ background: '#f5f5f5', padding: '16px', borderRadius: '4px' }}>
                  {JSON.stringify(selectedJob.best_parameters, null, 2)}
                </pre>
              ) : (
                <div>No best parameters available yet.</div>
              )}
            </TabPane>
          </Tabs>
        )}
      </Modal>
    </div>
  );
};

export default AutoML;
