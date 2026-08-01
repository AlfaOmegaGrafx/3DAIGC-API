# Project — stable facts beyond CLAUDE.md. Read before feature work or when confused. Cap: 80 lines.

## Architecture (≤10 lines — how the pieces talk; MAP.md owns the directory list)
- Clients (OpenNexus3DStudio) POST jobs → FastAPI routers (`api/routers/`) → Redis job queue.
- Separate **scheduler** process (`scripts/scheduler_service.py`) dequeues work, loads model adapters, writes results under `outputs/`.
- Multi-worker uvicorn (`api/main_multiworker.py`) serves polling/download; GPU work stays in scheduler workers.
- Model enablement + VRAM hints live in `config/models.yaml`; adapters in `adapters/` wrap thirdparty trees.
- Optional **MSF Map Service** (`:8443`) is a sibling process on DGX (not this repo) for fabric hosting; spatial fabric settings sync via `scripts/sync-spatial-fabric-env.sh`.
- Optional **LingBot-Map** is not a separate server — environment-scan jobs run inside this API after `scripts/install_lingbot_map.sh`.

## Constraints & non-goals (hard requirements; things deliberately unsupported)
- Prefer self-hosted / local GPU (DGX Spark aarch64); cloud SaaS inference is not the product.
- Non-commercial model routes stay disabled unless explicitly enabled in `models.yaml`.
- Do not `git push` from agent sessions unless the user asks.
- Do not put Space-Time Host / Sneeze in this repo’s reboot script — they are native GUI apps; API+MSF are the long-running services.
- Happy-path-only adapters are incomplete: empty uploads, missing weights, and OOM must fail with actionable errors.

## Glossary (domain terms with exact meanings — misreading one produces wrong code)
- **Job** — async unit of work tracked in Redis; clients poll `/api/v1/system/jobs/{id}`.
- **Adapter** — Python wrapper that loads a thirdparty model and implements one feature contract.
- **World package** — `world.manifest.json` + assets for OpenNexus splat/mesh worlds.
- **Environment scan** — walk video / frames → LingBot-Map recon → world package (+ optional metric scale).
- **Template rig** — Blender path applying `assets/example_autorig/template.vrm` skeleton to AIGC meshes.
- **MSF** — Metaverse Scene Format fabric hosted by MSF_Map_Svc (separate repo), not LingBot.

## Landmines (cross-cutting gotchas, ≤15; area-specific ones belong in .agent/areas/)
<!-- format: symptom → actual cause → what to do instead -->
- TRELLIS / spconv JIT fails → missing `source scripts/env_local_gpu.sh` (ninja / `CUMM_CUDA_ARCH_LIST`) → always source before starting services.
- GB10 UMA free VRAM looks tiny via `cudaMemGetInfo` → unified memory pool; don’t trust free alone → see DGX hardware rules / live probe.
- `/system/status/` may return JSON `nan` serialization error → known fragile health payload → use `/docs` or feature endpoints to confirm up.
- LingBot jobs fail with install hint → weights/package not present → `bash scripts/install_lingbot_map.sh` then restart services.
- Client orientation / VRM vs template-rig GLB confusion → different loaders on OpenNexus → see `docs/API_AVATAR_RIG_CONTRACT.md`.
- Restarting API without Redis → jobs break → `bash scripts/ensure_redis.sh` (or `start-dgx-after-reboot.sh`).
