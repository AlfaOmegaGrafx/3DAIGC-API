"""
Redis-based Job Queue for Multi-Worker Deployment

This allows multiple uvicorn workers to share the same job queue
without conflicts.
"""

import asyncio
import json
import logging
from datetime import timedelta
from typing import Any, Dict, Optional

from core.utils.job_time import (
    enrich_job_timestamps,
    ensure_job_tz,
    format_job_timestamp,
    job_now,
    parse_job_timestamp,
)

import redis.asyncio as aioredis

from .job_queue import JobRequest, JobStatus

logger = logging.getLogger(__name__)


def _status_to_str(status: JobStatus) -> str:
    """Convert JobStatus enum to string for JSON serialization"""
    if hasattr(status, 'value'):
        return status.value
    return str(status)


class RedisJobQueue:
    """Redis-backed job queue for multi-worker deployments"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        queue_prefix: str = "3daigc",
        max_job_age_hours: int = 24,
    ):
        self.redis_url = redis_url
        self.queue_prefix = queue_prefix
        self.max_job_age_hours = max_job_age_hours
        
        # Redis keys
        self.pending_queue_key = f"{queue_prefix}:queue:pending"
        self.processing_set_key = f"{queue_prefix}:queue:processing"
        self.jobs_hash_key = f"{queue_prefix}:jobs"
        self.results_hash_key = f"{queue_prefix}:results"
        
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info(f"Connected to Redis at {self.redis_url}")

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()

    async def recover_orphaned_jobs(self):
        """
        Find any jobs stuck in the 'processing' set on startup and requeue them.
        This handles cases where the scheduler crashed while jobs were running.
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        orphaned_job_ids = await self.redis.smembers(self.processing_set_key)
        if not orphaned_job_ids:
            logger.info("No orphaned jobs found in 'processing' set.")
            return

        logger.warning(f"Found {len(orphaned_job_ids)} orphaned jobs. Requeuing them...")
        for job_id in orphaned_job_ids:
            logger.info(f"Requeuing orphaned job {job_id}")
            await self.requeue_job(job_id)

        logger.info("Finished requeuing all orphaned jobs.")

    async def enqueue(self, job_request: JobRequest) -> str:
        """Add a job to the queue"""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        job_id = job_request.job_id
        
        # Store job data (convert enum to string for JSON serialization)
        job_data = {
            "job_id": job_id,
            "feature": job_request.feature,
            "inputs": json.dumps(job_request.inputs),
            "model_preference": job_request.model_preference or "",
            "priority": job_request.priority,
            "status": _status_to_str(JobStatus.QUEUED),
            "created_at": format_job_timestamp(job_request.created_at),
            "metadata": json.dumps(job_request.metadata),
            "user_id": job_request.user_id or "",  # Store user_id for job isolation
        }
        
        await self.redis.hset(self.jobs_hash_key, job_id, json.dumps(job_data))
        
        # Add to pending queue (sorted by priority and timestamp)
        score = job_request.priority * 1e10 + ensure_job_tz(job_request.created_at).timestamp()
        await self.redis.zadd(self.pending_queue_key, {job_id: score})
        
        logger.info(f"Enqueued job {job_id} to Redis")
        return job_id

    async def dequeue(self) -> Optional[JobRequest]:
        """Get next job from queue (FIFO with priority)"""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        # Use ZPOPMIN to get lowest score (highest priority, earliest timestamp)
        result = await self.redis.zpopmin(self.pending_queue_key, count=1)
        
        if not result:
            return None
        
        job_id = result[0][0]
        
        # Get job data
        job_data_str = await self.redis.hget(self.jobs_hash_key, job_id)
        if not job_data_str:
            logger.warning(f"Job {job_id} not found in jobs hash")
            return None
        
        job_data = json.loads(job_data_str)
        
        # Mark as processing
        await self.redis.sadd(self.processing_set_key, job_id)
        job_data["status"] = _status_to_str(JobStatus.PROCESSING)
        await self.redis.hset(self.jobs_hash_key, job_id, json.dumps(job_data))
        
        # Reconstruct JobRequest (job_id is auto-generated, so we set it after)
        job_request = JobRequest(
            feature=job_data["feature"],
            inputs=json.loads(job_data["inputs"]),
            model_preference=job_data.get("model_preference") or None,
            priority=job_data.get("priority", 0),
            metadata=json.loads(job_data.get("metadata", "{}")),
            user_id=job_data.get("user_id") or None,  # Restore user_id
        )
        # Restore original job_id and created_at
        job_request.job_id = job_id
        job_request.created_at = parse_job_timestamp(job_data["created_at"]) or job_now()
        
        return job_request

    async def requeue_job(self, job_id: str):
        """Put a job back at the front of the queue"""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        # Remove from processing set
        await self.redis.srem(self.processing_set_key, job_id)
        
        # Get job data to update status
        job_data_str = await self.redis.hget(self.jobs_hash_key, job_id)
        if job_data_str:
            job_data = json.loads(job_data_str)
            job_data["status"] = _status_to_str(JobStatus.QUEUED)
            job_data["created_at"] = format_job_timestamp(job_now())
            job_data.pop("error", None)
            job_data.pop("failed_at", None)
            await self.redis.hset(self.jobs_hash_key, job_id, json.dumps(job_data))
            
            # Re-add to queue with same priority but earlier timestamp for FIFO
            score = job_data.get("priority", 0) * 1e10 + job_now().timestamp() - 1000
            await self.redis.zadd(self.pending_queue_key, {job_id: score})

    async def complete_job(self, job_id: str, result: Any):
        """Mark job as completed"""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        # Remove from processing
        await self.redis.srem(self.processing_set_key, job_id)
        
        # Update job status
        job_data_str = await self.redis.hget(self.jobs_hash_key, job_id)
        if job_data_str:
            job_data = json.loads(job_data_str)
            job_data["status"] = _status_to_str(JobStatus.COMPLETED)
            job_data["completed_at"] = format_job_timestamp(job_now())
            await self.redis.hset(self.jobs_hash_key, job_id, json.dumps(job_data))
        
        # Store result
        await self.redis.hset(self.results_hash_key, job_id, json.dumps(result))
        
        # Set expiration for result (24 hours)
        await self.redis.expire(f"{self.results_hash_key}:{job_id}", 86400)

    async def fail_job(self, job_id: str, error: str):
        """Mark job as failed"""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        # Remove from processing
        await self.redis.srem(self.processing_set_key, job_id)
        
        # Update job status
        job_data_str = await self.redis.hget(self.jobs_hash_key, job_id)
        if job_data_str:
            job_data = json.loads(job_data_str)
            job_data["status"] = _status_to_str(JobStatus.FAILED)
            job_data["error"] = error
            job_data["failed_at"] = format_job_timestamp(job_now())
            await self.redis.hset(self.jobs_hash_key, job_id, json.dumps(job_data))

    async def fail_processing_jobs(self, error: str) -> int:
        """Mark every in-flight job as failed (e.g. scheduler shutdown)."""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        processing_job_ids = await self.redis.smembers(self.processing_set_key)
        if not processing_job_ids:
            return 0

        failed_count = 0
        for job_id in processing_job_ids:
            await self.fail_job(job_id, error)
            failed_count += 1
            logger.warning(f"Marked in-flight job {job_id} as failed: {error}")

        return failed_count

    async def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status and result"""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        job_data_str = await self.redis.hget(self.jobs_hash_key, job_id)
        if not job_data_str:
            return None
        
        job_data = json.loads(job_data_str)
        
        # Parse nested JSON fields
        if "inputs" in job_data and isinstance(job_data["inputs"], str):
            job_data["inputs"] = json.loads(job_data["inputs"])
        if "metadata" in job_data and isinstance(job_data["metadata"], str):
            job_data["metadata"] = json.loads(job_data["metadata"])
        
        # Get result if completed
        if job_data["status"] == _status_to_str(JobStatus.COMPLETED):
            result_str = await self.redis.hget(self.results_hash_key, job_id)
            if result_str:
                job_data["result"] = json.loads(result_str)
        
        return enrich_job_timestamps(job_data)

    async def get_queue_status(self) -> Dict:
        """Get queue statistics"""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        pending_count = await self.redis.zcard(self.pending_queue_key)
        processing_count = await self.redis.scard(self.processing_set_key)
        total_jobs = await self.redis.hlen(self.jobs_hash_key)
        
        return {
            "pending": pending_count,
            "processing": processing_count,
            "total_jobs": total_jobs,
        }

    async def cleanup_old_jobs(self):
        """Clean up old completed/failed jobs"""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        cutoff_time = job_now() - timedelta(hours=self.max_job_age_hours)
        
        # Get all jobs
        all_jobs = await self.redis.hgetall(self.jobs_hash_key)
        
        deleted_count = 0
        for job_id, job_data_str in all_jobs.items():
            job_data = json.loads(job_data_str)
            
            # Check if job is old and completed/failed
            if job_data["status"] in [_status_to_str(JobStatus.COMPLETED), _status_to_str(JobStatus.FAILED)]:
                completed_at = job_data.get("completed_at") or job_data.get("failed_at")
                if completed_at:
                    job_time = ensure_job_tz(parse_job_timestamp(completed_at) or job_now())
                    if job_time < cutoff_time:
                        await self.redis.hdel(self.jobs_hash_key, job_id)
                        await self.redis.hdel(self.results_hash_key, job_id)
                        deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old jobs from Redis")

    async def mark_job_started(self, job_id: str, model_id: str):
        """Mark job as started with model info"""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        job_data_str = await self.redis.hget(self.jobs_hash_key, job_id)
        if job_data_str:
            job_data = json.loads(job_data_str)
            job_data["status"] = _status_to_str(JobStatus.PROCESSING)
            job_data["model_id"] = model_id
            job_data["started_at"] = format_job_timestamp(job_now())
            await self.redis.hset(self.jobs_hash_key, job_id, json.dumps(job_data))

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's still pending"""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        # Try to remove from pending queue
        removed = await self.redis.zrem(self.pending_queue_key, job_id)
        
        if removed:
            # Update status
            job_data_str = await self.redis.hget(self.jobs_hash_key, job_id)
            if job_data_str:
                job_data = json.loads(job_data_str)
                job_data["status"] = _status_to_str(JobStatus.FAILED)
                job_data["error"] = "Cancelled by user"
                await self.redis.hset(self.jobs_hash_key, job_id, json.dumps(job_data))
            return True
        
        return False

    # Compatibility methods for original JobQueue interface
    async def start_persistence(self):
        """Compatibility method - Redis persistence is always active"""
        logger.debug("Redis persistence is always active, no action needed")
        pass

    async def stop_persistence(self):
        """Compatibility method - Redis persistence is always active"""
        logger.debug("Redis persistence is always active, no action needed")
        pass

    async def cleanup_expired_jobs(self):
        """Alias for cleanup_old_jobs for compatibility"""
        await self.cleanup_old_jobs()
    
    def _job_request_from_redis(self, job_id: str, job_data: dict) -> JobRequest:
        """Rebuild a JobRequest with full status/error fields from Redis JSON."""
        inputs = job_data.get("inputs", {})
        if isinstance(inputs, str):
            inputs = json.loads(inputs)
        metadata = job_data.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}

        payload = {
            "job_id": job_id,
            "feature": job_data["feature"],
            "inputs": inputs,
            "model_preference": job_data.get("model_preference"),
            "priority": job_data.get("priority", 0),
            "timeout_seconds": job_data.get("timeout_seconds", 3600),
            "metadata": metadata,
            "user_id": job_data.get("user_id"),
            "status": job_data.get("status", _status_to_str(JobStatus.QUEUED)),
            "progress": job_data.get("progress", 0.0),
            "assigned_model": job_data.get("model_id") or job_data.get("assigned_model"),
            "created_at": job_data["created_at"],
            "started_at": job_data.get("started_at"),
            "completed_at": job_data.get("completed_at") or job_data.get("failed_at"),
            "error": job_data.get("error"),
            "retry_count": job_data.get("retry_count", 0),
            "max_retries": job_data.get("max_retries", 4),
            "last_retry_at": job_data.get("last_retry_at"),
            "retry_reason": job_data.get("retry_reason"),
        }
        return JobRequest.from_dict(payload)

    async def get_jobs_by_status(self, status: JobStatus) -> list:
        """Get all jobs with specified status (compatibility method)"""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        target_status = _status_to_str(status)
        matching_jobs = []
        
        # Get all jobs from Redis
        all_jobs = await self.redis.hgetall(self.jobs_hash_key)
        
        for job_id, job_data_str in all_jobs.items():
            try:
                job_data = json.loads(job_data_str)
                if job_data.get("status") == target_status:
                    matching_jobs.append(self._job_request_from_redis(job_id, job_data))
            except Exception as e:
                logger.error(f"Error parsing job {job_id}: {e}")
                continue
        
        return matching_jobs
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job from Redis (compatibility method)"""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        try:
            # Remove from pending queue
            await self.redis.zrem(self.pending_queue_key, job_id)
            
            # Remove from processing set
            await self.redis.srem(self.processing_set_key, job_id)
            
            # Delete job data
            deleted_count = await self.redis.hdel(self.jobs_hash_key, job_id)
            
            # Delete result if exists
            await self.redis.hdel(self.results_hash_key, job_id)
            
            if deleted_count > 0:
                logger.info(f"Deleted job {job_id} from Redis")
                return True
            else:
                logger.warning(f"Job {job_id} not found in Redis")
                return False
        except Exception as e:
            logger.error(f"Error deleting job {job_id}: {e}")
            return False

