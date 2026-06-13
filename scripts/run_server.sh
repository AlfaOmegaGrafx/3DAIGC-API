#!/bin/bash

# Multi-Worker Deployment Script for 3D Generative Models Backend
# This script starts both the scheduler service and multiple FastAPI workers
#
# Usage:
#   ./scripts/run_multiworker.sh [OPTIONS]
#
# Options:
#   --user-auth-enabled     Enable user authentication (default: false)
#   --debug                 Enable debug mode (default: false)
#   --help                  Show this help message

set -e

# Parse command line arguments
USER_AUTH_ENABLED="false"
DEBUG_MODE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --user-auth-enabled)
            USER_AUTH_ENABLED="true"
            shift
            ;;
        --debug)
            DEBUG_MODE="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --user-auth-enabled     Enable user authentication (default: false)"
            echo "  --debug                 Enable debug mode (default: false)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  P3D_REDIS_URL          Redis connection URL (default: redis://localhost:6379)"
            echo "  P3D_HOST               API host address (default: 0.0.0.0)"
            echo "  P3D_PORT               API port (default: 7842)"
            echo "  P3D_WORKERS            Number of API workers (default: 4)"
            echo "  P3D_LOG_LEVEL          Logging level (default: info)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting 3D Generative Models Backend (Multi-Worker Mode)..."
echo ""

# Check if configuration files exist
if [ ! -f "config/system.yaml" ]; then
    echo "❌ Configuration file config/system.yaml not found"
    echo "Please run ./scripts/setup.sh to create configuration files"
    exit 1
fi

if [ ! -f "config/models.yaml" ]; then
    echo "❌ Configuration file config/models.yaml not found"
    echo "Please run ./scripts/setup.sh to create configuration files"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# CUDA 12.x + spconv/cumm arch list (default /usr/local/cuda may point at CUDA 13).
# shellcheck source=scripts/env_local_gpu.sh
source "$(dirname "$0")/env_local_gpu.sh"

# Essential configuration parameters
export P3D_USER_AUTH_ENABLED="$USER_AUTH_ENABLED"
export P3D_DEBUG="$DEBUG_MODE"

# Configuration
REDIS_URL=${P3D_REDIS_URL:-"redis://localhost:6379"}
API_HOST=${P3D_HOST:-"0.0.0.0"}
API_PORT=${P3D_PORT:-7842}
API_WORKERS=${P3D_WORKERS:-4}
LOG_LEVEL=${P3D_LOG_LEVEL:-"info"}

echo "📋 Configuration:"
echo "   Redis URL: $REDIS_URL"
echo "   API Host: $API_HOST"
echo "   API Port: $API_PORT"
echo "   API Workers: $API_WORKERS"
echo "   Log Level: $LOG_LEVEL"
echo "   User Auth: $USER_AUTH_ENABLED"
echo "   Debug Mode: $DEBUG_MODE"
echo ""

# Check if Redis is running
echo "🔍 Checking Redis connection..."
if command -v redis-cli &> /dev/null; then
    if ! redis-cli -u "$REDIS_URL" ping > /dev/null 2>&1; then
        echo "❌ Cannot connect to Redis at $REDIS_URL"
        echo ""
        echo "Please start Redis first:"
        echo "   docker run -d -p 6379:6379 redis:latest"
        echo "   # or"
        echo "   redis-server --daemonize yes"
        exit 1
    fi
    echo "✅ Redis is running"
else
    echo "⚠️  redis-cli not found, skipping Redis check"
fi
echo ""

# Create PID directory for tracking processes
PID_DIR="./run"
mkdir -p "$PID_DIR"

SCHEDULER_PID_FILE="$PID_DIR/scheduler.pid"
API_PID_FILE="$PID_DIR/api.pid"

# Function to cleanup processes on exit
wait_for_jobs_to_finish() {
    local redis_url="${P3D_REDIS_URL:-redis://localhost:6379}"
    local max_wait="${P3D_SHUTDOWN_DRAIN_SEC:-7200}"
    local poll="${P3D_SHUTDOWN_POLL_SEC:-5}"

    if [[ "${P3D_DRAIN_JOBS_ON_SHUTDOWN:-1}" != "1" ]]; then
        return 0
    fi
    if ! command -v redis-cli &> /dev/null; then
        echo "   (redis-cli unavailable; skipping job drain before shutdown)"
        return 0
    fi

    local processing
    processing="$(redis-cli -u "$redis_url" SCARD 3daigc:queue:processing 2>/dev/null || echo 0)"
    if [[ "${processing:-0}" -eq 0 ]]; then
        return 0
    fi

    echo "   Waiting for ${processing} in-flight GPU job(s) to finish (max ${max_wait}s)..."
    echo "   Tip: run under nohup/systemd so closing the terminal does not stop long jobs."
    local elapsed=0
    while [[ "$elapsed" -lt "$max_wait" ]]; do
        processing="$(redis-cli -u "$redis_url" SCARD 3daigc:queue:processing 2>/dev/null || echo 0)"
        if [[ "${processing:-0}" -eq 0 ]]; then
            echo "   All in-flight jobs finished."
            return 0
        fi
        if (( elapsed % 60 == 0 && elapsed > 0 )); then
            echo "   Still waiting... ${processing} job(s) processing (${elapsed}s elapsed)"
        fi
        sleep "$poll"
        elapsed=$((elapsed + poll))
    done
    echo "   Drain timeout (${max_wait}s) — stopping scheduler anyway."
}

cleanup() {
    echo ""
    echo "🛑 Shutting down services..."

    wait_for_jobs_to_finish
    
    # Stop API workers
    if [ -f "$API_PID_FILE" ]; then
        API_PID=$(cat "$API_PID_FILE")
        if ps -p "$API_PID" > /dev/null 2>&1; then
            echo "   Stopping API workers (PID: $API_PID)..."
            kill "$API_PID" 2>/dev/null || true
            # Wait for graceful shutdown
            sleep 2
            # Force kill if still running
            if ps -p "$API_PID" > /dev/null 2>&1; then
                kill -9 "$API_PID" 2>/dev/null || true
            fi
        fi
        rm -f "$API_PID_FILE"
    fi
    
    # Stop scheduler service
    if [ -f "$SCHEDULER_PID_FILE" ]; then
        SCHEDULER_PID=$(cat "$SCHEDULER_PID_FILE")
        if ps -p "$SCHEDULER_PID" > /dev/null 2>&1; then
            echo "   Stopping scheduler service (PID: $SCHEDULER_PID)..."
            kill "$SCHEDULER_PID" 2>/dev/null || true
            # Wait for graceful shutdown
            sleep 3
            # Force kill if still running
            if ps -p "$SCHEDULER_PID" > /dev/null 2>&1; then
                kill -9 "$SCHEDULER_PID" 2>/dev/null || true
            fi
        fi
        rm -f "$SCHEDULER_PID_FILE"
    fi
    
    echo "✅ Services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start scheduler service
echo "🔧 Starting scheduler service..."
./venv/bin/python scripts/scheduler_service.py --redis-url "$REDIS_URL" --log-level "$LOG_LEVEL" > logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo $SCHEDULER_PID > "$SCHEDULER_PID_FILE"
echo "   Scheduler service started (PID: $SCHEDULER_PID)"
echo "   Logs: logs/scheduler.log"

# Wait for scheduler to initialize
echo "   Waiting for scheduler to initialize..."
sleep 5

# Check if scheduler is still running
if ! ps -p "$SCHEDULER_PID" > /dev/null 2>&1; then
    echo "❌ Scheduler service failed to start"
    echo "   Check logs/scheduler.log for details"
    cleanup
    exit 1
fi
echo "✅ Scheduler service ready"
echo ""

# Start FastAPI with multiple workers
echo "🌐 Starting FastAPI with $API_WORKERS workers..."
uvicorn api.main_multiworker:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --workers "$API_WORKERS" \
    --log-level "$LOG_LEVEL" \
    > logs/api.log 2>&1 &
API_PID=$!
echo $API_PID > "$API_PID_FILE"
echo "   API workers started (PID: $API_PID)"
echo "   Logs: logs/api.log"
echo ""

# Wait for API to initialize
echo "   Waiting for API to initialize..."
sleep 3

# Check if API is still running
if ! ps -p "$API_PID" > /dev/null 2>&1; then
    echo "❌ API workers failed to start"
    echo "   Check logs/api.log for details"
    cleanup
    exit 1
fi
echo "✅ API workers ready"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "✅ Multi-Worker Deployment Started Successfully!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📊 Service Status:"
echo "   Scheduler Service: Running (PID: $SCHEDULER_PID)"
echo "   API Workers:       Running (PID: $API_PID, $API_WORKERS workers)"
echo ""
echo "🔗 Endpoints:"
echo "   API:     http://$API_HOST:$API_PORT"
echo "   Docs:    http://$API_HOST:$API_PORT/docs"
echo "   Health:  http://$API_HOST:$API_PORT/health"
echo ""
echo "📝 Logs:"
echo "   Scheduler: tail -f logs/scheduler.log"
echo "   API:       tail -f logs/api.log"
echo ""
echo "🛑 To stop services: Press Ctrl+C or run: kill $API_PID $SCHEDULER_PID"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Monitor processes and keep script running
echo "📊 Monitoring services... (Press Ctrl+C to stop)"
echo ""

while true; do
    # Check if scheduler is still running
    if ! ps -p "$SCHEDULER_PID" > /dev/null 2>&1; then
        echo "❌ Scheduler service has stopped unexpectedly!"
        echo "   Check logs/scheduler.log for details"
        cleanup
        exit 1
    fi
    
    # Check if API is still running
    if ! ps -p "$API_PID" > /dev/null 2>&1; then
        echo "❌ API workers have stopped unexpectedly!"
        echo "   Check logs/api.log for details"
        cleanup
        exit 1
    fi
    
    # Sleep and check again
    sleep 5
done

