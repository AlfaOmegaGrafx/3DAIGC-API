# Avatar pipeline (3DAIGC-API)

End-to-end path: **photo → textured mesh → template VRM rig → (optional) VRM file / splat preview**.

## Assets

| File | Purpose |
|------|---------|
| `assets/example_autorig/template.vrm` | Master humanoid VRM (124+ morph targets, ARKit/Vive presets) |
| `assets/example_autorig/skeleton/template.fbx` | Cached skeleton extract (optional) |
| `assets/example_autorig/regression/template.json` | Expected counts for CI |

Legacy alias: template id `sifr2` → same files.

## API endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/mesh-generation/image-to-textured-mesh` | TRELLIS / Hunyuan mesh from image |
| `POST /api/v1/auto-rigging/generate-rig` | Auto-rig; use `rig_mode: "template"`, `humanoid_template_id: "template"` |
| `GET /api/v1/auto-rigging/humanoid-templates/{id}/manifest` | Template metadata for frontend VRM export |
| `POST /api/v1/splat-generation/image-to-splat` | TripoSplat → `.ply` / `.splat` (Spark.js preview) |

### Template rig request

```json
{
  "mesh_file_id": "<uploaded glb>",
  "rig_mode": "template",
  "humanoid_template_id": "template",
  "output_format": "glb",
  "model_preference": "unirig_auto_rig"
}
```

## Template rig alignment (fixed 2026-06)

`scripts/blender/apply_humanoid_template_rig.py`:

1. Uniform scale from template reference mesh height → target mesh height (**Blender Z-up** after glTF import)  
2. Optional yaw around vertical (**Blender Z**, default 0°) — do not rotate around Y; that was the prior bug  
3. **Feet alignment** — foot bones' lowest Z → target mesh min Z (not bbox center)  
4. X/Y centering on the ground plane  
5. `ARMATURE_ENVELOPE` parenting → skinned GLB  

**Prior bug (2026-06):** alignment used glTF Y as “up” inside Blender. Blender imports glTF as **Z-up**, so height/feet/yaw were applied on the wrong axes → inverted skeleton and mesh in the viewport.

See [API avatar rig contract](../../OpenNexus3DStudio/CharacterStudio/docs/API_AVATAR_RIG_CONTRACT.md) for export validation (`[API-Contract]` gate).


## Blend shapes & Arc2Avatar

| Approach | Blend shapes on avatar? | Status |
|----------|-------------------------|--------|
| Template rig (bones-only) | No — skeleton only on AIGC mesh | **Implemented** |
| Mesh wrap (CC Wrap / MeshMonk analog) | Yes — after non-rigid transfer | **Roadmap** (`apply_humanoid_template_wrap.py` stub) |
| [Arc2Avatar](https://github.com/dimgerogiannis/Arc2Avatar) | Yes — on **3D Gaussian head** (FLAME), not VRM body | **Stub** (`adapters/arc2avatar_adapter.py`, `docs/ARC2AVATAR_TRACK.md`) |
| Head/body stitch (P3-SAM) | Yes — keep template head mesh | **Planned** |

Project direction: **avatars must support blend shapes** for XR face tracking. Short term: template metadata in VRM export + wrap R&D. Optional: Arc2Avatar splat head + rigged body composite.

## VRM export from rigged GLB

Happens in **Character Studio** (browser download), not on the API:

1. Pipeline completes → rigged GLB loads in viewport  
2. User checks **Download VRM after pipeline** (or Save → VRM)  
3. `VRMExporter` writes `.vrm` with template manifest metadata merged  

The rigged GLB is not automatically a VRM until export runs — export **downloads** a `.vrm` file to the user's machine.

## Tests

```bash
./venv/bin/python -m pytest tests/test_humanoid_template.py tests/test_template_rig_alignment.py -q
./scripts/verify_humanoid_template.sh
```

## Download models

```bash
./scripts/download_models.sh trellis2 unirig triposplat
```
