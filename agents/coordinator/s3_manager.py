import boto3
import os
import logging
from typing import Any, Optional
from botocore.exceptions import ClientError
from botocore.config import Config

logger = logging.getLogger(__name__)

class S3Manager:
    """S3-compatible storage manager (optional - system works without it)"""
    
    s3_client: Optional[Any] = None
    endpoint_url: str
    bucket_name: str
    enabled: bool = False

    def __init__(self):
        self.endpoint_url = os.getenv("S3_ENDPOINT", "http://seaweedfs:8333")
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID", "any")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "any")
        self.bucket_name = os.getenv("S3_BUCKET", "deepfake-evidence")
        self.region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        boto_config = Config(
            connect_timeout=5,
            read_timeout=10,
            retries={'max_attempts': 2}
        )
        
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region_name,
                config=boto_config
            )
            self._ensure_bucket_exists()
            self.enabled = True
            logger.info(f"✅ S3Manager initialized: {self.endpoint_url}/{self.bucket_name}")
        except Exception as e:
            logger.warning(f"⚠️ S3 Storage unavailable (non-critical): {e}")
            logger.warning("⚠️ System will continue without S3 uploads")
            self.s3_client = None
            self.enabled = False

    def _ensure_bucket_exists(self) -> None:
        """Verifică existența bucket-ului"""
        if not self.s3_client:
            return

        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ Bucket '{self.bucket_name}' verified")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code in ('404', 'NoSuchBucket'):
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"📦 Bucket '{self.bucket_name}' created")
                except Exception as create_err:
                    logger.error(f"❌ Could not create bucket: {create_err}")
                    raise
            else:
                logger.warning(f"⚠️ S3 Bucket check failed: {error_code}")

    def upload_file(self, file_content: bytes, object_name: str, content_type: Optional[str] = None) -> str:
        """Upload file to S3 (returns empty string if S3 unavailable)"""
        if not self.enabled or not self.s3_client:
            logger.warning(f"⚠️ S3 unavailable, skipping upload of {object_name}")
            return ""

        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=file_content,
                **extra_args
            )
            
            url = f"{self.endpoint_url.rstrip('/')}/{self.bucket_name}/{object_name}"
            logger.info(f"📤 Uploaded: {url}")
            return url
            
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            return ""

    def get_download_url(self, object_name: str) -> str:
        """Returnează URL-ul direct"""
        if not self.enabled:
            return ""
        return f"{self.endpoint_url.rstrip('/')}/{self.bucket_name}/{object_name}"