# Mesh wrap roadmap (blend shapes on AIGC topology)

## Problem

`apply_humanoid_template_rig.py` is **bones-only**: it parents the AIGC mesh to the template armature but does **not** copy `template.vrm` morph targets onto the generated face (different vertex count).

Without wrap, **VRM facial expressions do not deform the generated face** even if expression names exist in an exported VRM file.

## Target

CC Wrap / MeshMonk-style **non-rigid registration**:

1. Deform template mesh to match AIGC surface  
2. Transfer shape keys (ARKit / Vive presets)  
3. Export single skinned mesh + morph targets → VRM  

## Phases

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Bones-only template rig + alignment fix | Done |
| 1 | VRM export merges template **metadata** + skeleton | Done (Character Studio) |
| 2 | `rig_mode: template_wrap` stub → same as rig + doc | Stub |
| 3 | Blender shrinkwrap + shape-key transfer (head ROI) | R&D |
| 4 | External MeshMonk / wrap binary integration | R&D |
| 5 | P3-SAM head/body stitch (keep template head mesh) | Planned |

## Files

- `scripts/blender/apply_humanoid_template_rig.py` — production bones path  
- `scripts/blender/apply_humanoid_template_wrap.py` — wrap stub  
- `core/utils/humanoid_template.py` — template registry  

## API flag (future)

```json
{
  "rig_mode": "template_wrap",
  "humanoid_template_id": "template"
}
```

Until phase 3+, behaves as `template` with stderr note.
