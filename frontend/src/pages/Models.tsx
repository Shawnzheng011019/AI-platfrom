import React from 'react';
import { Typography, Card, Row, Col, Button, Table, Tag, Space } from 'antd';
import { PlusOutlined, DownloadOutlined, DeploymentUnitOutlined, EyeOutlined } from '@ant-design/icons';

const { Title } = Typography;

interface Model {
  id: string;
  name: string;
  type: string;
  framework: string;
  status: string;
  accuracy: number;
  size: string;
  created_at: string;
}

const Models: React.FC = () => {
  const mockModels: Model[] = [
    {
      id: '1',
      name: 'BERT-base-uncased-finetuned',
      type: 'NLP',
      framework: 'PyTorch',
      status: 'ready',
      accuracy: 94.2,
      size: '440 MB',
      created_at: '2024-01-15T10:30:00Z'
    },
    {
      id: '2',
      name: 'ResNet50-ImageNet',
      type: 'CV',
      framework: 'PyTorch',
      status: 'deployed',
      accuracy: 92.8,
      size: '98 MB',
      created_at: '2024-01-14T15:20:00Z'
    },
    {
      id: '3',
      name: 'GPT-2-small-finetuned',
      type: 'LLM',
      framework: 'PyTorch',
      status: 'training',
      accuracy: 0,
      size: '548 MB',
      created_at: '2024-01-15T12:00:00Z'
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'green';
      case 'deployed': return 'blue';
      case 'training': return 'orange';
      case 'error': return 'red';
      default: return 'default';
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Model) => (
        <Space direction="vertical" size={0}>
          <strong>{text}</strong>
          <span style={{ color: '#666', fontSize: '12px' }}>
            {record.framework} • {record.size}
          </span>
        </Space>
      ),
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Tag color="blue">{type}</Tag>,
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
      title: 'Accuracy',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy: number) => 
        accuracy > 0 ? `${accuracy}%` : '-',
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
      render: (_, record: Model) => (
        <Space>
          <Button type="text" icon={<EyeOutlined />} />
          <Button type="text" icon={<DownloadOutlined />} />
          {record.status === 'ready' && (
            <Button type="text" icon={<DeploymentUnitOutlined />} />
          )}
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Title level={2}>Models</Title>
        <Button type="primary" icon={<PlusOutlined />}>
          Import Model
        </Button>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <div className="metric-card">
              <div className="metric-value">8</div>
              <div className="metric-label">Total Models</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div className="metric-card">
              <div className="metric-value">3</div>
              <div className="metric-label">Deployed</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div className="metric-card">
              <div className="metric-value">92.5%</div>
              <div className="metric-label">Avg Accuracy</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div className="metric-card">
              <div className="metric-value">2.1 GB</div>
              <div className="metric-label">Total Size</div>
            </div>
          </Card>
        </Col>
      </Row>

      <Card>
        <Table
          columns={columns}
          dataSource={mockModels}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
          }}
        />
      </Card>
    </div>
  );
};

export default Models;
