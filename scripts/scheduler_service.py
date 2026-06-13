"""
Standalone GPU Scheduler Service

This service runs the GPU scheduler and job processor independently from FastAPI.
This allows multiple FastAPI uvicorn workers to share the same scheduler.

Architecture:
    - FastAPI workers: Handle HTTP requests, submit jobs to Redis queue
    - Scheduler service (this file): Processes jobs from Redis queue using GPU workers

Deployment:
    1. Start Redis:
        docker run -d -p 6379:6379 redis:latest
    
    2. Start this scheduler service (single instance):
        python scripts/scheduler_service.py
    
    3. Start FastAPI with multiple workers:
        uvicorn api.main:app --host 0.0.0.0 --port 7842 --workers 4

Usage:
    python scripts/scheduler_service.py [--redis-url REDIS_URL] [--log-level INFO]
"""

import argparse
import asyncio
import logging
import signal
import sys
sys.path.append(".")

from core.config import get_settings, setup_logging
from core.scheduler import GPUMonitor
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler
from core.scheduler.redis_job_queue import RedisJobQueue

logger = logging.getLogger(__name__)

# Global scheduler instance for graceful shutdown
scheduler_instance = None
redis_queue_instance = None


shutdown_requested = False


async def run_scheduler_service(redis_url: str):
    """Run the scheduler service"""
    global scheduler_instance, redis_queue_instance, shutdown_requested

    try:
        from core.utils.gpu_env import apply_local_gpu_env

        apply_local_gpu_env()
        # Load settings
        settings = get_settings()
        setup_logging(settings.logging)

        logger.info("=" * 60)
        logger.info("Starting Standalone GPU Scheduler Service")
        logger.info("=" * 60)
        logger.info(f"Redis URL: {redis_url}")
        logger.info(f"Environment: {settings.environment}")

        # Initialize GPU monitor
        gpu_monitor = GPUMonitor(memory_buffer=1024)
        gpu_status = gpu_monitor.get_gpu_status()
        logger.info(f"Detected {len(gpu_status)} GPU(s):")
        for gpu in gpu_status:
            logger.info(
                f"  - GPU {gpu['id']}: {gpu['name']} "
                f"({gpu['memory_free']}/{gpu['memory_total']} MB free)"
            )

        # Initialize Redis job queue
        logger.info("Connecting to Redis job queue...")
        redis_queue_instance = RedisJobQueue(
            redis_url=redis_url,
            queue_prefix="3daigc",
            max_job_age_hours=24,
        )
        await redis_queue_instance.connect()
        logger.info("✓ Connected to Redis")

        # Recover any orphaned jobs from a previous crash
        logger.info("Checking for orphaned jobs from previous session...")
        await redis_queue_instance.recover_orphaned_jobs()

        # Create scheduler with job processing enabled
        logger.info("Initializing GPU scheduler...")
        scheduler_instance = MultiprocessModelScheduler(
            gpu_monitor=gpu_monitor,
            job_queue=redis_queue_instance,
            enable_processing=True,  # Enable job processing in this service
        )

        # Register models based on environment
        from core.scheduler.model_factory import get_model_configs_from_settings

        model_configs = get_model_configs_from_settings(settings.models)
        logger.info(f"Registering {len(model_configs)} models...")
        for model_id, config in model_configs.items():
            scheduler_instance.register_model(config)
            logger.info(
                f"  ✓ {model_id} ({config['feature_type']}) - "
                f"VRAM: {config.get('vram_requirement', 'N/A')}MB"
            )

        # Start scheduler
        logger.info("Starting scheduler...")
        await scheduler_instance.start()

        logger.info("=" * 60)
        logger.info("✓ Scheduler Service is READY")
        logger.info("=" * 60)
        logger.info("The scheduler is now processing jobs from the Redis queue.")
        logger.info("You can start FastAPI workers with: uvicorn api.main:app --workers N")
        logger.info("Press Ctrl+C to stop the service")
        logger.info("=" * 60)

        # Keep running until interrupted
        while not shutdown_requested:
            await asyncio.sleep(1)

            # Periodically log queue status
            if int(asyncio.get_event_loop().time()) % 30 == 0:
                queue_status = await redis_queue_instance.get_queue_status()
                logger.info(
                    f"Queue status: {queue_status['pending']} pending, "
                    f"{queue_status['processing']} processing"
                )

    except KeyboardInterrupt:
        logger.info("\nReceived shutdown signal (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal error in scheduler service: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Graceful shutdown
        logger.info("Shutting down scheduler service...")

        if scheduler_instance:
            try:
                await scheduler_instance.stop()
                logger.info("✓ Scheduler stopped")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")

        if redis_queue_instance:
            try:
                await redis_queue_instance.disconnect()
                logger.info("✓ Redis disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting Redis: {e}")

        logger.info("Scheduler service shutdown complete")


def signal_handler(signum, frame):
    """Request graceful shutdown; the main loop exits and ``finally`` drains workers."""
    global shutdown_requested
    logger.info(f"\nReceived signal {signum}")
    shutdown_requested = True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GPU Scheduler Service")
    parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://localhost:6379",
        help="Redis connection URL (default: redis://localhost:6379)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the service
    try:
        asyncio.run(run_scheduler_service(args.redis_url))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

