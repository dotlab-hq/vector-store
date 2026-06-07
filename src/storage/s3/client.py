import asyncio
from dataclasses import dataclass

import boto3
import httpx
from botocore.exceptions import ClientError

from src.config import settings
from src.observability.logging import get_logger

logger = get_logger()

# Custom S3-compatible providers (e.g. behind Cloudflare) may reject
# the chunked transfer encoding that boto3 sends by default.
# Using a presigned-URL PUT with httpx avoids this by sending a
# plain Content-Length based request instead.
_USE_PRESIGNED_UPLOAD = True


@dataclass
class S3Object:
    key: str
    size: int
    content_type: str = ""


class S3Client:
    def __init__(self) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint or None,
            aws_access_key_id=settings.s3_access_key or None,
            aws_secret_access_key=settings.s3_secret_key or None,
            region_name=settings.s3_region,
        )
        self._bucket = settings.s3_bucket

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous boto3 call in a thread executor to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> S3Object:
        """Upload raw bytes to S3 via presigned URL PUT.

        Boto3's default chunked transfer encoding is incompatible with
        some S3-compatible providers.  Generating a presigned PUT URL and
        uploading with a standard HTTP client avoids this.
        """
        if _USE_PRESIGNED_UPLOAD:
            await self._presigned_put_upload(key, data, content_type)
        else:
            await self._run_sync(
                self._client.put_object,
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        logger.info("s3_upload", key=key, size=len(data))
        return S3Object(key=key, size=len(data), content_type=content_type)

    async def _presigned_put_upload(self, key: str, data: bytes, content_type: str) -> None:
        """Generate a presigned PUT URL and upload data via httpx."""
        presigned_url = await self._run_sync(
            self._client.generate_presigned_url,
            "put_object",
            Params={
                "Bucket": self._bucket,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=settings.s3_presign_expiry,
            HttpMethod="PUT",
        )
        async with httpx.AsyncClient() as client:
            response = await client.put(
                presigned_url,
                content=data,
                headers={"Content-Type": content_type},
                timeout=30.0,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"S3 presigned PUT failed: HTTP {response.status_code} "
                    f"{response.text[:500]}"
                )

    async def download(self, key: str) -> bytes:
        """Download raw bytes from S3."""
        try:
            response = await self._run_sync(
                self._client.get_object,
                Bucket=self._bucket,
                Key=key,
            )
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"S3 key not found: {key}")
            raise

    async def delete(self, key: str) -> None:
        """Delete an object from S3."""
        await self._run_sync(
            self._client.delete_object,
            Bucket=self._bucket,
            Key=key,
        )
        logger.info("s3_delete", key=key)

    async def exists(self, key: str) -> bool:
        """Check if an object exists in S3."""
        try:
            await self._run_sync(
                self._client.head_object,
                Bucket=self._bucket,
                Key=key,
            )
            return True
        except ClientError:
            return False

    async def list_objects(self, prefix: str = "", max_keys: int = 1000) -> list[S3Object]:
        """List objects under a prefix.

        Some S3-compatible providers (e.g. the one behind
        storage.wpsadi.dev) return NoSuchKey for empty / unknown
        prefixes instead of an empty Contents list.  Treat that as
        a successful empty result so callers don't crash.
        """
        try:
            response = await self._run_sync(
                self._client.list_objects_v2,
                Bucket=self._bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "NoSuchBucket"):
                return []
            raise
        objects: list[S3Object] = []
        for obj in response.get("Contents", []):
            objects.append(
                S3Object(
                    key=obj["Key"],
                    size=obj["Size"],
                )
            )
        return objects

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for downloading."""
        return await self._run_sync(
            self._client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )
