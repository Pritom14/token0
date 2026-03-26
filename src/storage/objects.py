"""Object storage — S3/MinIO in full mode, local filesystem in lite mode."""

import io
import os

from src.config import settings

_client = None
_lite_dir: str | None = None


def _get_lite_dir() -> str:
    global _lite_dir
    if _lite_dir is None:
        _lite_dir = os.path.join(os.path.dirname(settings.sqlite_path), ".token0_images")
        os.makedirs(_lite_dir, exist_ok=True)
    return _lite_dir


def get_s3_client():
    global _client
    if settings.is_lite:
        return None  # not used in lite mode

    if _client is None:
        import boto3
        from botocore.config import Config as BotoConfig

        _client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )
        try:
            _client.head_bucket(Bucket=settings.s3_bucket)
        except _client.exceptions.ClientError:
            _client.create_bucket(Bucket=settings.s3_bucket)
    return _client


def upload_image(key: str, data: bytes, content_type: str = "image/jpeg") -> str:
    if settings.is_lite:
        path = os.path.join(_get_lite_dir(), key)
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "wb") as f:
            f.write(data)
        return f"file://{path}"

    client = get_s3_client()
    client.upload_fileobj(
        io.BytesIO(data),
        settings.s3_bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"{settings.s3_endpoint}/{settings.s3_bucket}/{key}"


def download_image(key: str) -> bytes:
    if settings.is_lite:
        path = os.path.join(_get_lite_dir(), key)
        with open(path, "rb") as f:
            return f.read()

    client = get_s3_client()
    buf = io.BytesIO()
    client.download_fileobj(settings.s3_bucket, key, buf)
    buf.seek(0)
    return buf.read()
