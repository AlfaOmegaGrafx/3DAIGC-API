# Kimodo text-to-motion — operations (DGX)

**Status (2026-06-25):** Working. Verified job `544b726e-d335-4a30-8de9-5a94f98d651b` completed (~60s warm).

## Agent behavior (mandatory)

- **Run everything on DGX yourself** — prefetch, restart, test jobs, log grep. **Never** tell the user to run scripts unless blocked (credentials, physical device, approval gate).
- On Kimodo failure: check `logs/scheduler.log`, job history API, Llama HF cache shards **before** asking the user.

## Architecture

| Piece | Location |
|-------|----------|
| API endpoint | `POST /api/v1/motion-generation/text-to-motion` |
| Adapter | `adapters/kimodo_adapter.py` |
| Model config | `config/models.yaml` → `kimodo_text_to_motion` |
| Frontend | OpenNexus3DStudio animation bar (`KimodoMotionPromptBar.jsx`) |

Kimodo loads **Llama-3-8B-Instruct** locally via LLM2Vec when the text-encoder sidecar is down (`Connection refused` → fallback is normal).

## Required env (`.env` + worker inherit)

- `TEXT_ENCODER_DEVICE=cpu` — keeps Llama off GPU (~16GB); Kimodo diffusion uses GPU.
- `HF_TOKEN` or `~/.cache/huggingface/token` — gated Llama access.

## Failure modes & fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `403 gated repo` | No HF access to Llama | Approve access; ensure token in `.env` or HF cache |
| `Timed out waiting for worker model load` (15 min) | Default worker load timeout **900s** while downloading Llama (~16GB) | `worker_load_timeout_sec: 3600` in `models.yaml`; scheduler reads per-model timeout |
| `Worker process exited before model load completed` | **Incomplete Llama shards** in HF cache | Run `bash scripts/prefetch_kimodo_deps.sh`; verify all 4 `model-0000*-of-00004.safetensors` |
| First job slow | Cold worker load (Llama CPU + Kimodo GPU) | Expected; subsequent jobs ~1 min with warm worker |

## Prefetch / verify

```bash
bash scripts/prefetch_kimodo_deps.sh
ls ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/*/model*.safetensors
# Must list 00001, 00002, 00003, 00004
```

## Scheduler config (do not revert)

- `config/models.yaml`: `worker_load_timeout_sec: 3600` on `kimodo_text_to_motion`
- `core/config.py`: `ModelConfig.worker_load_timeout_sec`
- `core/scheduler/multiprocess_scheduler.py`: uses per-model timeout (not hardcoded 900)
- `scripts/prefetch_kimodo_deps.sh`: Llama + Kimodo-SOMA-RP-v1.1 cache

## Restart after backend changes

```bash
bash scripts/restart_services.sh
curl -sf http://localhost:7842/api/v1/system/models | python3 -m json.tool
```

## Test job (agent runs this)

```bash
curl -sf -X POST http://localhost:7842/api/v1/motion-generation/text-to-motion \
  -H 'Content-Type: application/json' \
  -d '{"text_prompt":"walking forward slowly","duration":3,"output_format":"studio_motion"}'
```

## License

Ship **Kimodo-SOMA-RP-v1.1** only. **Kimodo-SMPLX-RP-v1** is BLOCKED. See `docs/MODEL_LICENSES.md`.
