# Model & weight license audit (OpenNexus3DStudio / 3DAIGC-API)

**Hard prerequisite:** Any model integrated into OpenNexus3DStudio or shipped to paying users must be **cleared for commercial use** under its license. Research-only or personal-use weights are **not allowed** in production builds.

This document is the source of truth. Re-check upstream licenses when upgrading weights or swapping mirrors (e.g. `fishwowater/*` vs `tencent/*`).

**Legend**

| Status | Meaning |
|--------|---------|
| **OK** | Commercial use allowed (subject to standard license conditions). |
| **CONDITIONAL** | Commercial use allowed only if you meet listed constraints (territory, MAU, attribution, gated acceptance, etc.). |
| **BLOCKED** | Non-commercial / personal-use only — do not enable in commercial product. |
| **UNKNOWN** | No clear license on weight repo — treat as BLOCKED until verified. |

---

## Summary table (API routes)

| Model ID | Weights / repo | License | Commercial | Notes |
|----------|----------------|---------|--------------|-------|
| `trellis_image_to_textured_mesh` | `microsoft/TRELLIS-image-large` | MIT | **OK** | Cond: `facebook/dinov2-giant` (Apache-2.0) |
| `trellis_text_to_textured_mesh` | `microsoft/TRELLIS-text-xlarge` | MIT | **OK** | Same DINOv2 dependency |
| `trellis_image_mesh_painting` | TRELLIS image-large | MIT | **OK** | |
| `trellis_text_mesh_painting` | TRELLIS text-xlarge | MIT | **OK** | |
| `trellis2_image_to_textured_mesh` | `microsoft/TRELLIS.2-4B` | MIT | **CONDITIONAL** | DINOv3 + BiRefNet (see below) |
| `trellis2_image_mesh_painting` | TRELLIS.2-4B | MIT | **CONDITIONAL** | Same aux models |
| `hunyuan3dv21_image_to_raw_mesh` | `tencent/Hunyuan3D-2.1` or `fishwowater/Hunyuan3D-2.1` | Tencent Community | **CONDITIONAL** | Not EU/UK/KR; 1M+ MAU → Tencent approval |
| `hunyuan3dv21_image_to_textured_mesh` | Hunyuan3D-2.1 + paint | Tencent Community | **CONDITIONAL** | Same; uses DINOv2 + RealESRGAN |
| `hunyuan3dv21_image_mesh_painting` | Hunyuan3D-2.1 paint | Tencent Community | **CONDITIONAL** | Same |
| `ultrashape_image_to_raw_mesh` | `infinith/UltraShape` + Hunyuan coarse | Apache-2.0 (weights) + Tencent (Hunyuan) | **CONDITIONAL** | UltraShape weights OK; inherits Hunyuan rules |
| `partpacker_image_to_raw_mesh` | `nvidia/PartPacker` | NVIDIA Source (§3.3) | **BLOCKED** | **Non-commercial research only** |
| `partfield_mesh_segmentation` | `pretrained/PartField/model_objaverse.pt` | NVIDIA License (§3.3) | **BLOCKED** | **Non-commercial research/education only** |
| `p3sam_mesh_segmentation` | P3-SAM / Hunyuan3D-Part | Tencent 3D-Part Community | **CONDITIONAL** | Same territory / MAU rules as Hunyuan |
| `unirig_auto_rig` | `VAST-AI/UniRig` | MIT | **OK** | Requires Blender at runtime (`sudo apt install blender` on aarch64; see `scripts/install_blender_unirig.sh` and `utils/blender_runtime.py`) |
| `skintokens_auto_rig` | `VAST-AI/SkinTokens` | MIT | **OK** | TokenRig unified rigging; outputs GLB. Upstream recommends >=14GB VRAM. |
| `triposplat_image_to_splat` | `VAST-AI/TripoSplat` | MIT | **OK** | Image → `.ply`/`.splat`; render with [Spark.js](https://sparkjs.dev/) / `@sparkjsdev/spark`. Uses BiRefNet rembg weights (MIT). |
| `fastmesh_v1k_retopology` | `WopperSet/FastMesh-V1K` | S-Lab (code) / HF: none | **BLOCKED** | Upstream `thirdparty/FastMesh/LICENSE`: **non-commercial only** |
| `fastmesh_v4k_retopology` | `WopperSet/FastMesh-V4K` | Same | **BLOCKED** | Same |
| `partuv_uv_unwrapping` | PartField `model_objaverse.ckpt` | NVIDIA (via PartField) | **BLOCKED** | Same weights as PartField |
| `voxhammer_text_mesh_editing` | VoxHammer + TRELLIS | MIT | **OK** | TRELLIS weights MIT |
| `voxhammer_image_mesh_editing` | VoxHammer + TRELLIS | MIT | **OK** | |
| `kimodo_text_to_motion` | `Kimodo-SOMA-RP-v1.1` (+ `nv-tlabs/kimodo` code) | NVIDIA Open Model + Apache-2.0 | **CONDITIONAL** | **Ship SOMA-RP only.** Do not enable SMPL-X/G1 variants (see Kimodo section). |

---

## Auxiliary / dependency weights (not always separate API routes)

| Asset | Source | License | Commercial | Action |
|-------|--------|---------|--------------|--------|
| DINOv2 ViT-g | `facebook/dinov2-giant` | Apache-2.0 | **OK** | Used by TRELLIS v1, Hunyuan |
| DINOv3 ViT-L | `facebook/dinov3-vitl16-pretrain-lvd1689m` | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) | **CONDITIONAL** | Commercial allowed; export/military restrictions; HF gated access |
| Background removal | `ZhengPeng7/BiRefNet` | MIT | **OK** | **Use this** for TRELLIS.2 rembg |
| Background removal | `briaai/RMBG-2.0` | Bria personal / non-commercial | **BLOCKED** | **Never use** in OpenNexus3DStudio (Microsoft default in upstream `pipeline.json`) |
| RealESRGAN | `RealESRGAN_x4plus.pth` | BSD-3-Clause (typical) | **OK** | Hunyuan texture upscaler; confirm release terms |
| TRELLIS.2 rembg | See BiRefNet above | MIT | **OK** | Config patched in `pretrained/TRELLIS.2/TRELLIS.2-4B/*.json` |

---

## BLOCKED — do not use commercially

### `briaai/RMBG-2.0`
- Personal / non-commercial use only.
- TRELLIS.2 upstream defaults to this; **3DAIGC-API overrides to `ZhengPeng7/BiRefNet` (MIT).**

### NVIDIA PartField (`pretrained/PartField/model_objaverse.pt`)
- License §3.3: *"only may be used or intended for use **non-commercially**"* (research and educational purposes).
- Blocks: `partfield_mesh_segmentation`, `partuv_uv_unwrapping` (same checkpoint family).

### NVIDIA PartPacker (`nvidia/PartPacker` / `flow.pt`)
- NVIDIA Source Code License §3.3: non-commercial **research** only.

### FastMesh (`WopperSet/FastMesh-V1K`, `WopperSet/FastMesh-V4K`, `thirdparty/FastMesh`)
- `thirdparty/FastMesh/LICENSE` (S-Lab): redistribution and use for **non-commercial purpose** only.
- HF weight repos declare **no license** — treat as non-commercial until a permissive license is published.

---

## CONDITIONAL — commercial with constraints

### Tencent Hunyuan3D-2.1 (`tencent/Hunyuan3D-2.1`, `fishwowater/Hunyuan3D-2.1`)
- **Tencent Hunyuan 3D 2.1 Community License** (`thirdparty/Hunyuan3D-2.1/LICENSE`).
- Commercial use is contemplated, but:
  - **Territory:** worldwide **except EU, UK, South Korea** (no use outside Territory).
  - **Scale:** if products exceed **1M MAU**, must obtain **written approval** from Tencent (`hunyuan3d@tencent.com`).
  - **Hosted services:** allowed; must pass through license terms to users.
  - **Outputs:** you own outputs; cannot use outputs to train unrelated third-party models.
- Legal review recommended for your jurisdictions and SaaS model.

### Tencent Hunyuan3D-Part / P3-SAM
- Same structure as Hunyuan 3D-Part Community License (`thirdparty/Hunyuan3DPart/LICENSE`).
- Same territory and 1M MAU rules.

### Meta DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`)
- **DINOv3 License** — not “personal only”; commercial use permitted with restrictions (export controls, no military/weapons end-use, redistribution under same license).
- Requires accepting terms on Hugging Face or [Meta download portal](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
- GitHub repo weights are the **same** checkpoints via a different download path (not a different license).

### UltraShape (`infinith/UltraShape`)
- Weights on HF tagged **Apache-2.0** (commercial-friendly).
- Pipeline still depends on **Hunyuan3D-2.1** for coarse mesh → inherits Tencent **CONDITIONAL** rules.

### NVIDIA Kimodo text-to-motion (`kimodo_text_to_motion`)

**What we ship:** `Kimodo-SOMA-RP-v1.1` via `adapters/kimodo_adapter.py` (SOMA skeleton → `studio_motion.json` for uploaded VRM).

| Asset | Source | License | Commercial | Action |
|-------|--------|---------|--------------|--------|
| Kimodo Python package | [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) | Apache-2.0 | **OK** | Code in `thirdparty/kimodo` |
| **Kimodo-SOMA-RP-v1.1** (default) | [HF Kimodo collection](https://huggingface.co/collections/nvidia/kimodo-v1) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | **CONDITIONAL** | **Allowed for commercial product use** under NVIDIA Open Model terms (accept HF/NVIDIA terms; follow redistribution/output rules). |
| Kimodo-SOMA/G1 **SEED** | Same collection | NVIDIA Open Model + **BONES-SEED** dataset license | **CONDITIONAL** | Benchmark/seed weights — not used in API adapter; review BONES-SEED if enabling. |
| **Kimodo-SMPLX-RP-v1** | Same collection | **NVIDIA R&D / internal scientific R&D model license** | **BLOCKED** | Non-commercial R&D only — **do not expose** in OpenNexus3DStudio or `models.yaml`. |
| Kimodo-G1-RP | Same collection | NVIDIA Open Model | **CONDITIONAL** | Robot (Unitree G1) skeleton — not wired; different retarget path than VRM. |
| Bones Rigplay 700h (training) | Proprietary (NVIDIA) | Proprietary | N/A | Training data for RP models; inference outputs are user-generated NPZ/JSON — counsel on redistribution of bundled motion files. |
| SOMA-X skeleton lib | [NVlabs/SOMA-X](https://github.com/NVlabs/SOMA-X) | Verify upstream `LICENSE` | **CONDITIONAL** | Runtime dep for SOMA joint naming / optional `[soma]` extra. |
| Output **NPZ** / **studio_motion.json** | Generated at inference | Governed by model license + your ToS | **OK** (typical) | User prompts → motion artifacts; do not redistribute pretrained weights. |

**Audit conclusion:** **No conflict** if the API stays on **Kimodo-SOMA-RP-v1.1** only. **SMPL-X NPZ weights are BLOCKED** for commercial OpenNexus3DStudio — same class as NVIDIA PartField/PartPacker R&D licenses. G1/SOMA NPZ checkpoint families other than SOMA-RP-v1.1 need separate legal review before enabling.

---

## OK for commercial use (typical OpenNexus3DStudio deployment)

| Component | License |
|-----------|---------|
| TRELLIS / TRELLIS-text (`microsoft/*`) | MIT |
| TRELLIS.2 core (`microsoft/TRELLIS.2-4B`) | MIT |
| VoxHammer (code) | MIT |
| UniRig (`VAST-AI/UniRig`) | MIT |
| BiRefNet rembg (`ZhengPeng7/BiRefNet`) | MIT |
| TripoSplat (`VAST-AI/TripoSplat`) | MIT |
| DINOv2 (`facebook/dinov2-giant`) | Apache-2.0 |

---

## Enforcement in this repo

1. **`config/models.yaml`** — routes with **BLOCKED** licenses have `enabled: false` and a `license_note`.
2. **`scripts/download_models.sh`** — by default skips BLOCKED models unless `ALLOW_NON_COMMERCIAL_MODELS=1`.
3. **New models** — add a row here and set commercial status **before** merging; default to BLOCKED until verified.

### Environment variables

```bash
# Default: only download / verify commercial-safe model groups
export COMMERCIAL_USE=1

# Opt-in for research-only weights (NOT for OpenNexus3DStudio production)
export ALLOW_NON_COMMERCIAL_MODELS=1
./scripts/download_models.sh -m partfield,partpacker
```

---

## Commercial alternatives (replace BLOCKED routes)

Use this when shipping OpenNexus3DStudio. **Integration effort** is approximate (S/M/L).

### Mesh segmentation (replaces PartField)

| Option | License | Fit | Effort | Notes |
|--------|---------|-----|--------|-------|
| **P3-SAM** (`p3sam_mesh_segmentation`) | Tencent 3D-Part — **CONDITIONAL** | Best drop-in | **S** | Already in 3DAIGC-API; verified on GB10. Enable in `models.yaml` if Territory + MAU rules OK. |
| **HoloPart** ([VAST-AI/HoloPart](https://huggingface.co/VAST-AI/HoloPart)) | MIT (code); verify HF weight card | Part decomposition / amodal segments | **L** | Different from PartField (completes occluded parts). Good for editable part meshes. |
| **Geometric segmentation** (trimesh / pymeshlab / connectivity) | Permissive deps | Coarse part splits | **S** | No ML license risk; quality below learned methods. |
| PartField | NVIDIA non-commercial | — | — | **Do not ship** |

### Mesh retopology (replaces FastMesh)

| Option | License | Fit | Effort | Notes |
|--------|---------|-----|--------|-------|
| **Instant Meshes** ([wjakob/instant-meshes](https://github.com/wjakob/instant-meshes)) | BSD-3-Clause | Quad-dominant artist-style retopo | **M** | Used commercially (e.g. Modo). Not GPT vertex budgets like FastMesh v1k/v4k. |
| **QuadriFlow** ([hjwdzh/QuadriFlow](https://github.com/hjwdzh/QuadriFlow)) | MIT (use `BUILD_FREE_LICENSE=ON` for deps) | Quad remesh | **M** | Blender “Quad” remesh; loses mesh data layers. |
| **meshoptimizer** | MIT | Triangle decimation / simplification | **S** | Good for poly reduction, not semantic quad flow. |
| **TRELLIS.2 `remesh`** (cumesh path) | MIT (TRELLIS.2 stack) | Cleanup / remesh in generation | **S** | Already on textured mesh route; not a standalone retopo API. |
| FastMesh | S-Lab non-commercial | — | — | **Do not ship** |

### UV unwrapping (replaces PartUV)

| Option | License | Fit | Effort | Notes |
|--------|---------|-----|--------|-------|
| **xatlas** ([jpcy/xatlas](https://github.com/jpcy/xatlas)) | MIT | Atlas / lightmap UVs | **S** | Already used via `cumesh` / TRELLIS.2 tooling. Part-blind. |
| **P3-SAM parts + xatlas per part** | Tencent + MIT | Part-aware UV without PartField | **M** | Segment with P3-SAM, unwrap each part with xatlas, pack. |
| **HoloPart parts + xatlas** | MIT + MIT | Part-aware, complete parts | **L** | After HoloPart integration. |
| Blender unwrap (`bpy`) | GPL (Blender) | Full-featured UV | **L** | Blocked on aarch64 today (no `bpy` wheel). |
| PartUV + PartField weights | NVIDIA non-commercial | — | — | **Do not ship** |

### Image → raw mesh (replaces PartPacker)

| Option | License | Notes |
|--------|---------|-------|
| TRELLIS / TRELLIS.2 / Hunyuan3D-2.1 | MIT / CONDITIONAL | Already in product; prefer these over PartPacker. |
| PartPacker | NVIDIA non-commercial | **Do not ship** |

### Recommended OpenNexus3DStudio path (minimal new work)

1. **Segmentation:** **`p3sam_mesh_segmentation`** — enabled in `config/models.yaml` (Tencent **CONDITIONAL**).
2. **Retopo:** **`instant_meshes_retopology`** — enabled; build binary via `./scripts/install_instant_meshes.sh` or set `INSTANT_MESHES_BIN`.
3. **UV:** **`xatlas_uv_unwrapping`** — enabled (MIT `xatlas` pip package); optional v2 with P3-SAM part splits.

| Model ID | License | Config |
|----------|---------|--------|
| `p3sam_mesh_segmentation` | Tencent 3D-Part (CONDITIONAL) | `enabled: true` |
| `xatlas_uv_unwrapping` | MIT (xatlas) | `enabled: true` |
| `instant_meshes_retopology` | BSD-3-Clause | `enabled: true` (binary required) |

---

## Checklist before adding a new model

- [ ] Read the **weight** license (not just the GitHub code license).
- [ ] Confirm **commercial** use explicitly (not merely “open source”).
- [ ] List all **auxiliary** HF repos (encoders, rembg, VAE, etc.).
- [ ] Document territory, MAU, or export restrictions.
- [ ] Add row to this file and wire `download_models.sh` + `models.yaml`.
- [ ] Never add `briaai/RMBG-2.0` or community **mirrors** of gated weights without legal review.

---

*Last audited: 2026-05-29. Not legal advice — have counsel review before commercial launch.*
