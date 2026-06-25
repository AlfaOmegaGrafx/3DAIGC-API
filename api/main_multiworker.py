"""
FastAPI application main entry point - Multi-Worker Configuration

This version is designed for multi-worker deployment with a separate scheduler service.

Architecture:
    - Multiple FastAPI workers handle HTTP requests
    - Jobs are submitted to Redis queue
    - Separate scheduler service (scheduler_service.py) processes jobs
    
Deployment:
    1. Start Redis: docker run -d -p 6379:6379 redis:latest
    2. Start scheduler: python scripts/scheduler_service.py
    3. Start API: uvicorn api.main_multiworker:app --workers 4

Key Differences from main.py:
    - No scheduler creation (connects to external scheduler via Redis)
    - Redis-based job queue for cross-worker communication
    - Can safely run with multiple uvicorn workers
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings, setup_logging #, create_directories
from core.file_store import FileStore
from core.scheduler.redis_job_queue import RedisJobQueue
from core.utils.exceptions import BaseAPIException

from .routers import (
    auto_rigging,
    file_upload,
    image_tools,
    mesh_editing,
    mesh_generation,
    mesh_retopology,
    mesh_segmentation,
    mesh_uv_unwrapping,
    system,
    users,
)

logger = logging.getLogger(__name__)

# Global variables for shared resources
redis_job_queue = None
auth_service = None
file_store = None


# Configure CORS
def configure_cors(app: FastAPI, settings):
    """Configure CORS middleware.

    Wildcard origins cannot be combined with credentialed CORS (browser spec).
    This API uses Bearer tokens in Authorization, not cookies, so credentials
    are unnecessary and ``["*"]`` works for browser clients (e.g. Vite dev).
    """
    origins = settings.security.cors_origins
    allow_credentials = origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


# Configure security middleware
def configure_security(app: FastAPI, settings):
    """Configure security middleware"""
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"],
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global redis_job_queue, auth_service, file_store

    # Startup
    logger.info("Starting 3D Generative Models Backend (Multi-Worker Mode)...")

    try:
        # Load configuration
        settings = get_settings()

        # Setup logging
        setup_logging(settings.logging)

        # Create necessary directories
        # create_directories(settings.storage)

        # Connect to Redis job queue (shared with scheduler service)
        redis_url = getattr(settings, "redis_url", "redis://localhost:6379")
        logger.info(f"Connecting to Redis at {redis_url}")

        redis_job_queue = RedisJobQueue(
            redis_url=redis_url,
            queue_prefix="3daigc",
            max_job_age_hours=24,
        )
        await redis_job_queue.connect()

        # Store job queue in app state for dependency injection
        app.state.job_queue = redis_job_queue

        # Initialize file store for cross-worker file metadata sharing
        from redis.asyncio import Redis as AsyncRedis
        
        logger.info("Initializing Redis-based file store...")
        file_store_redis = AsyncRedis.from_url(redis_url, decode_responses=False)
        file_store = FileStore(
            redis_client=file_store_redis,
            key_prefix="3daigc",
            default_ttl_seconds=86400,  # 24 hours
        )
        app.state.file_store = file_store
        logger.info("✓ File store initialized")

        # Initialize authentication service (conditionally based on settings)
        if settings.user_auth_enabled:
            from redis.asyncio import Redis
            from core.auth import AuthService, UserStorage
            
            logger.info("Initializing authentication service...")
            redis_client = Redis.from_url(redis_url, decode_responses=True)
            user_storage = UserStorage(redis_client, key_prefix="3daigc")
            auth_service = AuthService(user_storage)
            
            # Store auth service in app state for dependency injection
            app.state.auth_service = auth_service
            
            logger.info("✓ Authentication service initialized")
        else:
            app.state.auth_service = None
            logger.info("⚠ User authentication is DISABLED - running in simple mode")

        logger.info("=" * 60)
        logger.info("✓ FastAPI worker startup completed successfully")
        logger.info("=" * 60)
        logger.info("This worker submits jobs to Redis queue")
        logger.info("Scheduler service processes jobs independently")
        logger.info(f"Debug mode: {'ENABLED' if settings.debug else 'DISABLED'}")
        if settings.user_auth_enabled:
            logger.info("User authentication: ENABLED (Redis-based)")
            logger.info("  - Users can only see their own jobs")
            logger.info("  - Admins can see all jobs")
        else:
            logger.info("User authentication: DISABLED (Simple mode)")
            logger.info("  - All clients can see all jobs")
        logger.info("=" * 60)

        yield

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

    finally:
        # Shutdown
        logger.info("Shutting down 3D Generative Models Backend...")

        # Cleanup resources
        if redis_job_queue:
            await redis_job_queue.disconnect()
        
        if file_store and hasattr(file_store, 'redis'):
            await file_store.redis.aclose()
        
        if auth_service and hasattr(auth_service.storage, 'redis'):
            await auth_service.storage.redis.close()

        logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="3D Generative Models API",
    description="Scalable 3D AI model inference server with VRAM-aware scheduling (Multi-Worker Mode)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS and security middleware
settings = get_settings()
configure_cors(app, settings)
configure_security(app, settings)


# Add middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()

    logger.info(f"Request: {request.method} {request.url}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} - "
        f"{request.method} {request.url} - "
        f"Time: {process_time:.3f}s"
    )

    return response


# Exception handlers
@app.exception_handler(BaseAPIException)
async def base_api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions"""
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.error_code or "API_ERROR",
            "message": exc.message,
            "detail": str(exc),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors"""
    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_VALUE",
            "message": "Invalid input value",
            "detail": str(exc),
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "NOT_FOUND",
            "message": "Resource not found",
            "detail": str(exc.detail)
            if hasattr(exc, "detail")
            else "The requested resource was not found",
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An internal server error occurred",
            "detail": "Please try again later or contact support",
        },
    )


# Include routers
app.include_router(system.router, prefix="/api/v1/system", tags=["System"])

app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])

app.include_router(file_upload.router, prefix="/api/v1", tags=["File Upload"])

app.include_router(image_tools.router, prefix="/api/v1", tags=["Image Tools"])

app.include_router(mesh_generation.router, prefix="/api/v1", tags=["Mesh Generation"])

app.include_router(mesh_editing.router, prefix="/api/v1", tags=["Mesh Editing"])

app.include_router(auto_rigging.router, prefix="/api/v1", tags=["Auto Rigging"])

app.include_router(
    mesh_segmentation.router, prefix="/api/v1", tags=["Mesh Segmentation"]
)

app.include_router(mesh_retopology.router, prefix="/api/v1", tags=["Mesh Retopology"])

app.include_router(
    mesh_uv_unwrapping.router, prefix="/api/v1", tags=["Mesh UV Unwrapping"]
)

from api.routers import splat_generation
from api.routers import world_generation
from api.routers import spatial_fabric

app.include_router(
    splat_generation.router, prefix="/api/v1", tags=["Splat Generation"]
)

app.include_router(
    world_generation.router, prefix="/api/v1", tags=["World Generation"]
)

app.include_router(
    spatial_fabric.router, prefix="/api/v1", tags=["Spatial Fabric"]
)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time(), "version": "1.0.0"}


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "3D Generative Models API",
        "version": "1.0.0",
        "description": "Scalable 3D AI model inference server (Multi-Worker Mode)",
        "docs_url": "/docs",
        "health_url": "/health",
        "deployment_mode": "multi_worker",
        "note": "Jobs are processed by external scheduler service",
    }

