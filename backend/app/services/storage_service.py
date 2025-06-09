import os
from minio import Minio
from minio.error import S3Error
from typing import Optional
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self):
        self.client = None
        self.bucket_name = "ai-platform"
        self._initialize_client()

    def _initialize_client(self):
        """Initialize MinIO client"""
        try:
            self.client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure
            )
            
            # Create bucket if it doesn't exist
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize MinIO client: {e}")
            self.client = None

    async def upload_file(self, file_path: str, object_name: str) -> bool:
        """Upload a file to MinIO storage"""
        if not self.client:
            logger.warning("MinIO client not available, using local storage")
            return True
        
        try:
            self.client.fput_object(self.bucket_name, object_name, file_path)
            logger.info(f"Uploaded {file_path} as {object_name}")
            return True
        except S3Error as e:
            logger.error(f"Failed to upload file: {e}")
            return False

    async def download_file(self, object_name: str, file_path: str) -> bool:
        """Download a file from MinIO storage"""
        if not self.client:
            logger.warning("MinIO client not available")
            return False
        
        try:
            self.client.fget_object(self.bucket_name, object_name, file_path)
            logger.info(f"Downloaded {object_name} to {file_path}")
            return True
        except S3Error as e:
            logger.error(f"Failed to download file: {e}")
            return False

    async def delete_file(self, object_name: str) -> bool:
        """Delete a file from MinIO storage"""
        if not self.client:
            logger.warning("MinIO client not available")
            return True
        
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Deleted {object_name}")
            return True
        except S3Error as e:
            logger.error(f"Failed to delete file: {e}")
            return False

    async def get_file_url(self, object_name: str, expires: int = 3600) -> Optional[str]:
        """Get a presigned URL for file access"""
        if not self.client:
            return None
        
        try:
            url = self.client.presigned_get_object(self.bucket_name, object_name, expires=expires)
            return url
        except S3Error as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

    def is_available(self) -> bool:
        """Check if storage service is available"""
        return self.client is not None
