# Arc2Avatar track

[Arc2Avatar](https://github.com/dimgerogiannis/Arc2Avatar) / [paper](https://arc2avatar.github.io/) — photoreal **FLAME + 3DGS** heads from one image (SDS train).

## Fit

| Layer | Arc2Avatar | Template VRM pipeline |
|-------|------------|------------------------|
| Body | No | TRELLIS + template / template_wrap |
| Head | High (3DGS) | AIGC mesh / wrap morphs |
| Export | `.ply` splat | `.vrm` / rigged GLB |
| Renderer | Spark.js | three-vrm |

**Composite (Studio Body+Cloth):** neck-open body VRM via `template_wrap` + Arc2Avatar head `.ply` parented to the Head bone (`attachHeadSplatToBody`). The selfie is **not** the body image.

## Status (API)

- Adapter: `adapters/arc2avatar_adapter.py` (`Arc2AvatarHeadAdapter`)
- Helper: `utils/arc2avatar_pipeline_helper.py`
- Routes: `GET /api/v1/arc2avatar/status`, `POST /api/v1/arc2avatar/image-to-head`
- Feature / model id: `arc2avatar_head`
- Repo: `thirdparty/Arc2Avatar` (+ CUDA submodules)
- `integrated: true` requires weights + Python that imports `torch`+CUDA + `diff_gaussian_rasterization`

## Install on DGX (GB10 / CUDA 12.8)

Upstream README pins CUDA 11.8 — **do not use on Spark**. Use:

```bash
bash /home/sifr/3DAIGC-API/scripts/install_arc2avatar_env.sh
# writes ARC2AVATAR_PYTHON + ARC2AVATAR_ROOT into .env
P3D_SKIP_PREFLIGHT=1 bash /home/sifr/3DAIGC-API/scripts/restart_services.sh
```

Weights: `python download_models.py` inside `thirdparty/Arc2Avatar` (already done when `weights_ready`).

License: FLAME + Arc2Face — record acceptance in `docs/MODEL_LICENSES.md`.

## Smoke

```bash
curl -s http://127.0.0.1:7842/api/v1/arc2avatar/status | jq .
# When integrated=true:
# POST /api/v1/arc2avatar/image-to-head  with image_file_id
```

SDS default **7000** iterations — expect long runtime. Smoke with `model_parameters.iterations` (e.g. 200) only after train loop accepts overrides.
