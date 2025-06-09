import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Typography, Space, Progress } from 'antd';
import {
  DatabaseOutlined,
  ExperimentOutlined,
  RobotOutlined,
  CloudServerOutlined,
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const { Title } = Typography;

interface DashboardStats {
  totalDatasets: number;
  activeTrainings: number;
  totalModels: number;
  gpuUsage: number;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats>({
    totalDatasets: 0,
    activeTrainings: 0,
    totalModels: 0,
    gpuUsage: 0,
  });

  const [trainingData] = useState([
    { name: 'Mon', trainings: 4 },
    { name: 'Tue', trainings: 3 },
    { name: 'Wed', trainings: 6 },
    { name: 'Thu', trainings: 8 },
    { name: 'Fri', trainings: 5 },
    { name: 'Sat', trainings: 2 },
    { name: 'Sun', trainings: 1 },
  ]);

  useEffect(() => {
    // Simulate loading stats
    setStats({
      totalDatasets: 12,
      activeTrainings: 3,
      totalModels: 8,
      gpuUsage: 75,
    });
  }, []);

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Title level={2}>Dashboard</Title>

      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Datasets"
              value={stats.totalDatasets}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Active Trainings"
              value={stats.activeTrainings}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Models"
              value={stats.totalModels}
              prefix={<RobotOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="GPU Usage"
              value={stats.gpuUsage}
              suffix="%"
              prefix={<CloudServerOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
            <Progress
              percent={stats.gpuUsage}
              showInfo={false}
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
              style={{ marginTop: 8 }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card title="Training Activity (Last 7 Days)">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="trainings"
                  stroke="#1890ff"
                  strokeWidth={2}
                  dot={{ fill: '#1890ff' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="System Status">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>CPU Usage</span>
                  <span>45%</span>
                </div>
                <Progress percent={45} showInfo={false} />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Memory Usage</span>
                  <span>68%</span>
                </div>
                <Progress percent={68} showInfo={false} />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Storage Usage</span>
                  <span>32%</span>
                </div>
                <Progress percent={32} showInfo={false} />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>GPU Memory</span>
                  <span>75%</span>
                </div>
                <Progress percent={75} showInfo={false} />
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </Space>
  );
};

export default Dashboard;
