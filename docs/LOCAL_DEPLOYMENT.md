# Local GPU Deployment (DGX / bare metal)

Run the API + job scheduler on a local machine with NVIDIA GPU(s). Used with **OpenNexus Character Studio** on a separate dev PC (`VITE_API_ENDPOINT` → DGX `:7842`).

## Prerequisites

- Linux (Ubuntu 20.04+ tested on DGX Spark)
- Python 3.10+ in `./venv` (see `scripts/install.sh`)
- CUDA 12.x (see `scripts/env_local_gpu.sh`)
- **Redis** on port 6379 (job queue for multi-worker API)
- Blender on `PATH` for template VRM rig (`BLENDER_BIN`, default `/usr/bin/blender`)

## Quick start

```bash
cd /path/to/3DAIGC-API

# 1) Redis (Docker)
docker start 3daigc-redis
# First time: docker compose up -d redis

# 2) Activate venv (if not already)
source venv/bin/activate
source scripts/env_local_gpu.sh

# 3) Start scheduler + API (background)
bash scripts/start_services_detached.sh
```

Verify:

```bash
curl -s http://127.0.0.1:7842/docs -o /dev/null -w '%{http_code}\n'   # expect 200
curl -s http://127.0.0.1:7842/api/v1/system/features | head
```

Logs:

```bash
tail -f logs/api.log logs/scheduler.log
```

## Foreground mode

```bash
bash scripts/run_server.sh
```

Stops gracefully and drains in-flight GPU jobs when `P3D_DRAIN_JOBS_ON_SHUTDOWN=1` (default).

## Stop services

```bash
kill $(cat run/api.pid) 2>/dev/null
kill $(cat run/scheduler.pid) 2>/dev/null
rm -f run/api.pid run/scheduler.pid
# Optional: docker stop 3daigc-redis
```

## Model weights

Commercial-safe defaults (see [MODEL_LICENSES.md](MODEL_LICENSES.md)):

```bash
./scripts/download_models.sh -m trellis,trellis2,hunyuan21,unirig,triposplat
./scripts/verify_all_enabled_models.sh
```

## Configuration

| File | Purpose |
|------|---------|
| `config/system.yaml` | Host, port, Redis URL, auth, upload TTL |
| `config/models.yaml` | Enabled models per feature route |
| `config/logging.yaml` | Log levels |

Environment overrides (common):

| Variable | Default | Description |
|----------|---------|-------------|
| `P3D_HOST` | `0.0.0.0` | API bind address |
| `P3D_PORT` | `7842` | API port |
| `P3D_REDIS_URL` | `redis://localhost:6379` | Job queue |
| `P3D_WORKERS` | `4` | Uvicorn worker processes |
| `P3D_USER_AUTH_ENABLED` | `false` | Enable Bearer token auth |

**Never commit** `.env`, tokens, or `venv/` — see root `.gitignore`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Error 111 connecting to localhost:6379` | Start Redis: `docker start 3daigc-redis` |
| Jobs stuck / empty history after Redis outage | Restart Redis + API; stale queue data may be lost — re-submit jobs |
| Scheduler not running | Check `run/scheduler.pid`, `logs/scheduler.log` |
| Template rig fails | Verify `BLENDER_BIN`, `assets/example_autorig/template.vrm` |
| VRAM OOM | Disable unused models in `config/models.yaml` |

## Character Studio integration

On the dev PC (Surface):

```env
DEV_API_PROXY_TARGET=http://10.0.0.158:7842
VITE_API_ENDPOINT=/__dev_dgx_proxy
```

Vite proxies `https://<PC>:3000/__dev_dgx_proxy/...` → DGX API.
