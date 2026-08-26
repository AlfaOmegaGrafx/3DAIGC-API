# Avatar pipeline (3DAIGC-API)

End-to-end path: **photo → textured mesh → ICT humanoid template → (optional) VRM file / splat preview**.

## Assets

| File | Purpose |
|------|---------|
| Operator-local `template_ict.vrm` | Default Body+Cloth morph head (`humanoid_template_id=ict`). Not shipped in the public tree — set `HUMANOID_TEMPLATE_VRM` or place at `assets/example_autorig/template_ict.vrm`. |
| `assets/example_autorig/appearance_base.vrm` | Appearance clothing fit base (slots) |

Deprecated request ids `template` / `sifr2` normalize to `ict`.

## API endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/mesh-generation/image-to-textured-mesh` | TRELLIS / Hunyuan mesh from image |
| `POST /api/v1/auto-rigging/generate-rig` | Auto-rig; use `rig_mode: "template"`, `humanoid_template_id: "ict"` |
| `GET /api/v1/auto-rigging/humanoid-templates/{id}/manifest` | Template metadata for frontend VRM export |
| `POST /api/v1/splat-generation/image-to-splat` | TripoSplat → `.ply` / `.splat` (Spark.js preview) |
| `POST /api/v1/world-generation/image-to-world` | World package: splat environment + optional mesh props |
| `GET /api/v1/system/jobs/{id}/download?asset=manifest` | World manifest after `image_to_world` job completes |

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

See [API avatar rig contract](API_AVATAR_RIG_CONTRACT.md) for export validation (`[API-Contract]` gate).


## Blend shapes & Arc2Avatar

| Approach | Blend shapes on avatar? | Status |
|----------|-------------------------|--------|
| Template rig (bones-only) | No — skeleton only on AIGC mesh | **Implemented** |
| Head stitch (`rig_mode: template_wrap`) | Yes — keep ICT morph head + AIGC body | **Phase 5 MVP** ([MESH_WRAP_ROADMAP.md](MESH_WRAP_ROADMAP.md)) |
| MeshMonk dense wrap | Yes — AIGC face likeness onto morph topo | **Deferred** (Phase 4, after stitch) |
| Shrinkwrap shape-key transfer | Uncertain on AIGC topo | **Optional R&D** (Phase 3 demoted) |
| Creature / SkinTokens face | Bone retarget (`jaw`/`chin`/`eye`) | **Client** `creatureFaceRetarget.js` — not MeshMonk |
| [Arc2Avatar](https://github.com/dimgerogiannis/Arc2Avatar) | Yes — on **3D Gaussian head** (FLAME), not VRM body | **Stub** (`adapters/arc2avatar_adapter.py`, `docs/ARC2AVATAR_TRACK.md`) |

Project direction: **avatars must support blend shapes** for XR face tracking. Ship Phase 5 head stitch first; MeshMonk likeness later. Creatures never use wrap.

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
