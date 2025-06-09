# AI Training Platform - Quick Start Guide

## 🚀 快速开始

### 前置要求

确保您的系统已安装以下软件：

- **Docker** (版本 20.0+)
- **Docker Compose** (版本 2.0+)
- **Git**
- **至少 8GB RAM**
- **至少 20GB 可用磁盘空间**

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd ai-platform
```

2. **运行安装脚本**
```bash
./scripts/setup.sh
```

3. **启动平台**
```bash
./scripts/start.sh
```

4. **访问应用**
- 前端界面: http://localhost:3000
- API文档: http://localhost:8000/docs
- MinIO控制台: http://localhost:9001

### 首次使用

1. **注册账户**
   - 访问 http://localhost:3000
   - 点击"Register"标签
   - 填写用户信息并注册

2. **上传数据集**
   - 登录后进入"Datasets"页面
   - 点击"Upload Dataset"
   - 选择数据文件并填写相关信息

3. **创建训练任务**
   - 进入"Training"页面
   - 点击"New Training Job"
   - 配置训练参数并启动

4. **查看模型**
   - 训练完成后在"Models"页面查看结果
   - 可以下载或部署模型

## 🔧 配置说明

### 环境变量

主要配置文件位于 `.env`，包含以下重要设置：

```bash
# 数据库配置
MONGODB_URL=mongodb://admin:password123@localhost:27017/ai_platform?authSource=admin

# Redis配置
REDIS_URL=redis://localhost:6379

# MinIO配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# 安全配置
SECRET_KEY=your-secret-key-here
```

### 存储目录

- `./uploads/` - 上传文件临时存储
- `./datasets/` - 数据集存储
- `./models/` - 训练模型存储
- `./logs/` - 应用日志

## 📊 支持的模型类型

### 1. 大语言模型 (LLM)
- **支持框架**: Hugging Face Transformers
- **基础模型**: GPT-2, BERT, RoBERTa, T5等
- **训练类型**: 微调 (Fine-tuning)
- **数据格式**: JSON, JSONL

### 2. 扩散模型 (Diffusion Models)
- **支持框架**: Diffusers
- **基础模型**: Stable Diffusion, DDPM等
- **训练类型**: 文本到图像生成
- **数据格式**: 图像+文本对

### 3. NLP模型
- **任务类型**: 文本分类, 命名实体识别, 情感分析
- **支持框架**: PyTorch, TensorFlow
- **数据格式**: CSV, JSON

### 4. 计算机视觉模型
- **任务类型**: 图像分类, 目标检测, 语义分割
- **支持框架**: PyTorch, TensorFlow
- **数据格式**: 图像文件夹结构

## 🛠️ 常用命令

### 服务管理
```bash
# 启动所有服务
./scripts/start.sh

# 停止所有服务
./scripts/stop.sh

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f [service_name]
```

### 数据库管理
```bash
# 连接MongoDB
docker-compose exec mongodb mongosh -u admin -p password123

# 连接Redis
docker-compose exec redis redis-cli
```

### 开发模式
```bash
# 后端开发
cd backend
pip install -r requirements.txt
python main.py

# 前端开发
cd frontend
npm install
npm start
```

## 🔍 故障排除

### 常见问题

1. **端口冲突**
   - 检查端口 3000, 8000, 27017, 6379, 9000 是否被占用
   - 修改 docker-compose.yml 中的端口映射

2. **内存不足**
   - 确保至少有 8GB 可用内存
   - 减少并发训练任务数量

3. **磁盘空间不足**
   - 清理不需要的Docker镜像: `docker system prune`
   - 删除旧的训练数据和模型

4. **GPU支持**
   - 安装 NVIDIA Docker runtime
   - 修改 docker-compose.yml 添加GPU支持

### 日志查看
```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs backend
docker-compose logs frontend
docker-compose logs mongodb
```

## 📚 更多资源

- [API文档](http://localhost:8000/docs) - 完整的API接口文档
- [用户手册](./docs/user-manual.md) - 详细使用说明
- [开发指南](./docs/development.md) - 开发和扩展指南
- [部署指南](./docs/deployment.md) - 生产环境部署

## 🤝 获取帮助

如果遇到问题，请：

1. 查看日志文件
2. 检查系统资源使用情况
3. 参考故障排除部分
4. 提交Issue到项目仓库

---

**祝您使用愉快！** 🎉
