import React, { useState, useEffect } from 'react';
import {
  Typography,
  Button,
  Table,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  Upload,
  message,
  Popconfirm,
} from 'antd';
import {
  PlusOutlined,
  UploadOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import { ColumnsType } from 'antd/es/table';
import api from '../services/api';

const { Title } = Typography;
const { Option } = Select;
const { TextArea } = Input;

interface Dataset {
  id: string;
  name: string;
  description?: string;
  dataset_type: string;
  format: string;
  status: string;
  file_size: number;
  tags: string[];
  owner_id: string;
  is_public: boolean;
  version: string;
  created_at: string;
  updated_at: string;
}

const Datasets: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [form] = Form.useForm();

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const response = await api.get('/datasets/');
      setDatasets(response.data);
    } catch (error) {
      message.error('Failed to fetch datasets');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleUpload = async (values: any) => {
    const formData = new FormData();
    formData.append('file', values.file.file);
    formData.append('name', values.name);
    formData.append('description', values.description || '');
    formData.append('dataset_type', values.dataset_type);
    formData.append('format', values.format);
    formData.append('tags', values.tags?.join(',') || '');
    formData.append('is_public', values.is_public || false);

    try {
      await api.post('/datasets/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      message.success('Dataset uploaded successfully');
      setModalVisible(false);
      form.resetFields();
      fetchDatasets();
    } catch (error) {
      message.error('Failed to upload dataset');
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.delete(`/datasets/${id}`);
      message.success('Dataset deleted successfully');
      fetchDatasets();
    } catch (error) {
      message.error('Failed to delete dataset');
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'green';
      case 'processing': return 'blue';
      case 'uploading': return 'orange';
      case 'error': return 'red';
      default: return 'default';
    }
  };

  const columns: ColumnsType<Dataset> = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
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
      title: 'Type',
      dataIndex: 'dataset_type',
      key: 'dataset_type',
      render: (type) => <Tag color="blue">{type.toUpperCase()}</Tag>,
    },
    {
      title: 'Format',
      dataIndex: 'format',
      key: 'format',
      render: (format) => <Tag>{format.toUpperCase()}</Tag>,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={getStatusColor(status)}>{status.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'Size',
      dataIndex: 'file_size',
      key: 'file_size',
      render: (size) => formatFileSize(size),
    },
    {
      title: 'Tags',
      dataIndex: 'tags',
      key: 'tags',
      render: (tags) => (
        <>
          {tags.map((tag: string) => (
            <Tag key={tag} color="geekblue">{tag}</Tag>
          ))}
        </>
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => new Date(date).toLocaleDateString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            type="text"
            icon={<EyeOutlined />}
            onClick={() => {/* View dataset details */}}
          />
          <Button
            type="text"
            icon={<EditOutlined />}
            onClick={() => {/* Edit dataset */}}
          />
          <Popconfirm
            title="Are you sure you want to delete this dataset?"
            onConfirm={() => handleDelete(record.id)}
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
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>Datasets</Title>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setModalVisible(true)}
        >
          Upload Dataset
        </Button>
      </div>

      <Table
        columns={columns}
        dataSource={datasets}
        rowKey="id"
        loading={loading}
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showQuickJumper: true,
        }}
      />

      <Modal
        title="Upload Dataset"
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleUpload}
        >
          <Form.Item
            name="name"
            label="Dataset Name"
            rules={[{ required: true, message: 'Please enter dataset name' }]}
          >
            <Input placeholder="Enter dataset name" />
          </Form.Item>

          <Form.Item
            name="description"
            label="Description"
          >
            <TextArea rows={3} placeholder="Enter dataset description" />
          </Form.Item>

          <Form.Item
            name="dataset_type"
            label="Dataset Type"
            rules={[{ required: true, message: 'Please select dataset type' }]}
          >
            <Select placeholder="Select dataset type">
              <Option value="text">Text</Option>
              <Option value="image">Image</Option>
              <Option value="audio">Audio</Option>
              <Option value="video">Video</Option>
              <Option value="tabular">Tabular</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="format"
            label="Format"
            rules={[{ required: true, message: 'Please select format' }]}
          >
            <Select placeholder="Select format">
              <Option value="csv">CSV</Option>
              <Option value="json">JSON</Option>
              <Option value="jsonl">JSONL</Option>
              <Option value="parquet">Parquet</Option>
              <Option value="images">Images</Option>
              <Option value="audio_files">Audio Files</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="tags"
            label="Tags"
          >
            <Select
              mode="tags"
              placeholder="Add tags"
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Form.Item
            name="file"
            label="File"
            rules={[{ required: true, message: 'Please select a file' }]}
          >
            <Upload
              beforeUpload={() => false}
              maxCount={1}
            >
              <Button icon={<UploadOutlined />}>Select File</Button>
            </Upload>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Upload
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
    </Space>
  );
};

export default Datasets;
