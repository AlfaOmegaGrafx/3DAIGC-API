# Arc2Avatar track (optional)

[Arc2Avatar](https://github.com/dimgerogiannis/Arc2Avatar) / [paper](https://arc2avatar.github.io/) generates **photoreal 3D Gaussian heads** from one image with **FLAME blendshape** expression control.

## Fit in OpenNexus stack

| Layer | Arc2Avatar | Template VRM pipeline |
|-------|------------|------------------------|
| Body | No | TRELLIS mesh + template rig |
| Head detail | High (3DGS) | AIGC mesh texture |
| Blend shapes | FLAME on splats | template.vrm presets (after wrap) |
| Export | `.ply` / splat folder | `.vrm` download |
| Renderer | Spark.js | three-vrm |

**Composite vision:** rigged body (GLB/VRM) + Arc2Avatar head splat parented to head bone, driven by same expression weights where mapped.

## Status

- Stub: `adapters/arc2avatar_adapter.py`  
- Not in `models.yaml` / scheduler until FLAME + Arc2Face license review  
- Training is SDS-heavy (minutes per subject), not real-time API inference  

## Integration checklist (future)

1. Clone `thirdparty/Arc2Avatar` + submodules  
2. `download_models.py` weights on DGX  
3. Adapter: image → head splat `.ply`  
4. OpenNexus3DStudio: load splat on `head` bone; expression map FLAME → VRM presets  
5. Document license in `docs/MODEL_LICENSES.md`
