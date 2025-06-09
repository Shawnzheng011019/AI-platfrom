// MongoDB initialization script
db = db.getSiblingDB('ai_platform');

// Create collections
db.createCollection('users');
db.createCollection('datasets');
db.createCollection('training_jobs');
db.createCollection('models');

// Create indexes
db.users.createIndex({ "username": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true });

db.datasets.createIndex({ "name": 1 });
db.datasets.createIndex({ "owner_id": 1 });
db.datasets.createIndex({ "dataset_type": 1 });
db.datasets.createIndex({ "status": 1 });
db.datasets.createIndex({ "created_at": 1 });

db.training_jobs.createIndex({ "name": 1 });
db.training_jobs.createIndex({ "owner_id": 1 });
db.training_jobs.createIndex({ "status": 1 });
db.training_jobs.createIndex({ "created_at": 1 });
db.training_jobs.createIndex({ "priority": 1 });

db.models.createIndex({ "name": 1 });
db.models.createIndex({ "owner_id": 1 });
db.models.createIndex({ "model_type": 1 });
db.models.createIndex({ "status": 1 });
db.models.createIndex({ "created_at": 1 });

print('Database initialized successfully');
