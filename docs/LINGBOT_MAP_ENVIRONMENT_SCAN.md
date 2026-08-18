# Galaxy XR walk → digital twin (LingBot-Map)

Physical-replica metaverse anchoring: capture with **outward-facing** cameras while walking, reconstruct a world package, and apply **1:1 meter** scale.

## Status

| Piece | Status |
|-------|--------|
| Default image-to-world (TripoSplat) | Unchanged |
| `POST /api/v1/world-generation/environment-scan` | Added |
| `POST /api/v1/file-upload/video` | Added |
| Metric calibration (`metric_scale.py`) | Added |
| LingBot-Map weights / runtime | **Optional** — `bash scripts/install_lingbot_map.sh` |
| Phase A: point cloud → isotropic Gaussian (Spark) | **Shipped** — `refine_to_3dgs` / `scripts/refine_env_scan_to_3dgs.py` |
| Phase B: gsplat train on exported COLMAP | **Shipped** — `train_3dgs` / `POST /train-3dgs` / `scripts/train_env_scan_3dgs.py` |
| Env mesh bake → OMB GLB | **Shipped** — `bake_env_mesh` / `POST /bake-env-mesh` / `scripts/bake_env_mesh.py` |

## Install (DGX)

```bash
cd /home/sifr/3DAIGC-API
bash scripts/install_lingbot_map.sh
# restart API workers after install
```

Refs: [LingBot-Map](https://github.com/Robbyant/lingbot-map), [paper](https://arxiv.org/html/2604.14141v2), [weights](https://huggingface.co/robbyant/lingbot-map), [project](https://technology.robbyant.com/lingbot-map).

## Capture (Galaxy XR)

1. Walk the space with **world-facing** cameras (steady pace, revisit corners).
2. Record / export a video (or ≥3 ordered frames).
3. Upload: `POST /api/v1/file-upload/video` → `video_file_id`.
4. Measure one real length (door width, wall span) in **meters**.

## 1:1 scale (required for physical replica)

Monocular recon is unitless. Pass `metric_calibration` so the twin matches reality:

```json
{
  "video_file_id": "…",
  "world_name": "Studio walk",
  "model_preference": "lingbot_map_environment_scan",
  "metric_calibration": {
    "mode": "reference_length",
    "true_meters": 0.9,
    "recon_length": 0.45
  }
}
```

| Mode | Fields | When |
|------|--------|------|
| `reference_length` | `true_meters`, `recon_length` | Best: measure a door/wall in recon UI later, or after a dry run |
| `two_points` | `true_meters`, `point_a`, `point_b` | Two 3D points spanning a known length |
| `player_height` | `recon_height`, optional `player_height_meters` (default 1.6) | Vertical span ≈ person height |
| `auto_bbox` | `true_meters` only | Uses point-cloud bbox diagonal as `recon_length` (approximate — prefer door/wall) |

Applied as `environment.transform.scale` (uniform or horizontal-only `[sx,1,sz]`) in `world.manifest.json`. Metadata records `coordinate_units: "meters"` and `one_to_one: true`.

### Orientation processing (required)

After LingBot reconstructs, every env-scan applies this order:

1. **Gravity → +Y** via **floor RANSAC** (`prefer_floor=True` — product default). Do **not** trust windowed LingBot camera-up alone (`camera_extrinsics+…` tilted Office rooms).
2. **Ceiling/floor check** — if the densest slab is on top, **Y-flip**
3. **Seat on Y=0**
4. **X-mirror** — OpenCV/LingBot left↔right vs Three.js
5. Optional **horizontal metric scale** for door width (`axis: horizontal`) so height stays correct

Broken camera averages that point near world **-Y** are flipped before align (Galaxy + windowed poses).

Mis-aligned worlds (wrong pitch / identity metric): `repair_world_gravity_alignment(world_dir, metric_calibration=…)` inverts the stored gravity, re-applies floor RANSAC, refreshes PLYs/cameras, and re-runs Phase A.

## API

```http
POST /api/v1/world-generation/environment-scan
Content-Type: application/json

{
  "video_file_id": "<id>",
  "world_name": "Living room",
  "metric_calibration": {
    "mode": "reference_length",
    "true_meters": 2.4,
    "recon_length": 1.2
  },
  "refine_to_3dgs": true
}
```

Poll job → `world_manifest_url` / `world_base_url` like image-to-world.

- Default: colored **point cloud** (`renderer: points`).
- `refine_to_3dgs: true`: Phase A isotropic **Gaussian splat** (`renderer: spark`) + optional `gs_dataset/` for Phase B.

## OpenNexus

Task type **Environment scan** (optional). Default Image to World stays TripoSplat.
Enable **Refine to 3DGS (Phase A)** when submitting if you want Spark Gaussians immediately.

## 3DGS refinement (continue here)

Point cloud first is intentional (fast, GB10-safe). Once the cloud looks right (gravity, left/right, metric door), convert:

### Phase A — isotropic Gaussians from points (no train)

Converts `environment.ply` (XYZRGB) → Spark PLY (`f_dc_*`, `opacity`, `scale_*`, `rot_*`). Keeps backup as `environment.points.ply`. Updates manifest to `gaussian_splat` / `spark`.

**On a new scan:**

```json
{ "refine_to_3dgs": true, "...": "rest of environment-scan body" }
```

**On an existing world package (Office / previous jobs):**

```bash
cd /home/sifr/3DAIGC-API
# use the API venv (has numpy / torch)
source .venv/bin/activate   # or: source venv/bin/activate
python scripts/refine_env_scan_to_3dgs.py \
  outputs/worlds/<JOB_ID>
```

If the scan was run **after** camera export landed, the script also writes:

| Path | Purpose |
|------|---------|
| `environment.points.ply` | XYZRGB backup |
| `environment.ply` | Spark Gaussians |
| `cameras_aligned.npz` | Gravity-aligned c2w + K |
| `gs_dataset/images/` + `gs_dataset/sparse/0/*.txt` | COLMAP TXT for Phase B |

Older jobs (pre-camera-export) still get Phase A Gaussians; re-run env-scan (or keep `_work/lingbot_out`) to get poses for Phase B.

Code: `core/utils/lingbot_3dgs_refine.py`.

### Phase B — photometric gsplat train (shipped)

Requires Phase A `gs_dataset/` (images + poses). Trains with the installed
[`gsplat`](https://github.com/nerfstudio-project/gsplat) package (no nerfstudio
install required). Preserves gravity-aligned coordinates (no scene renormalization).

**Pose source (important):** LingBot Studio X-mirror makes `det(R_c2w)=-1`.
COLMAP `images.txt` quaternions cannot represent that and silently corrupt poses
(muddy brown blob). Phase B loads matrix poses from `gs_dataset/poses_c2w.npy`
or `cameras_aligned.npz`. Export always writes `poses_c2w.npy`.

**Defaults:** init from Phase A Gaussians, densify/clone **off** (LingBot poses
+ densify previously exploded the AABB), opacity reset disabled, Spark DC-only
export + AABB prune. Pass `--enable-densify` only for true SfM COLMAP.

**On a new scan** (long job — prefer separate train for big walks):

```json
{ "refine_to_3dgs": true, "train_3dgs": true, "train_3dgs_steps": 7000 }
```

**On an existing world** (Office / previous Phase A):

```bash
cd /home/sifr/3DAIGC-API
source venv/bin/activate
python scripts/train_env_scan_3dgs.py outputs/worlds/<JOB_ID> --max-steps 7000
# smoke: --max-steps 200 --max-images 24
```

Or API:

```http
POST /api/v1/world-generation/train-3dgs
{ "world_id": "<JOB_ID>", "max_steps": 7000, "data_factor": 4 }
```

Writes:

| Path | Purpose |
|------|---------|
| `gs_train/point_cloud.ply` | Trained splat |
| `environment.ply` | Replaced with trained PLY (Spark) |
| `environment.phaseA.ply` | Phase A backup |
| `gs_dataset/poses_c2w.npy` | Authoritative c2w (4×4) |
| manifest `gaussian_phase` | `B_gsplat_trained` |

Default: 7000 steps, `data_factor=4` (downsample images). Metric `transform.scale` unchanged.

**Do not** full-batch `images.to(cuda)` for LingBot on GB10; Phase B uses gsplat’s per-view dataloader.

### Sharper / more video-faithful Gaussians (practical)

Capture (biggest lever):

1. **Slower walk, more overlap** — pause at corners; avoid motion blur and rolling shutter.
2. **More frames** — `max_frames: 600`, `frame_stride: 1` (pipeline cap ~600). Old Office jobs used ~365.
3. **Keep camera height steady**; cover walls/ceiling from multiple angles (holes = missing views).
4. **Correct door metric** — `axis: horizontal`, measured `true_meters` / `recon_length` (Office: 0.762 / 0.47). Wrong scale makes Spark look “soft” after parent transform.

Train (Phase B on a good Phase A world):

| Goal | Settings |
|------|----------|
| Normal | `--max-steps 7000`, `data_factor=2` |
| Sharper | `--max-steps 10000`, `data_factor=2` |
| Avoid | densify on LingBot poses; `>15000` steps (pose overfit); `data_factor=4` if you want max fidelity |

```bash
python scripts/train_env_scan_3dgs.py outputs/worlds/<JOB_ID> --max-steps 10000 --data-factor 2
```

Do **not** re-enable densify or metric SVD bake. Photometric color-only refine is optional after poses are solid; PC recolor from `environment.points.ply` is safer when poses are soft.

Same video can be reprocessed after the gravity lock — new scan picks up `prefer_floor=True` automatically.

### Env mesh bake (OMB / RP1)

Spark splat environments are **viewport-only**. Scene Assembler needs a **GLB**. After Phase A (`gs_dataset/` present):

```http
POST /api/v1/world-generation/bake-env-mesh
{ "world_id": "<JOB_ID>", "target_face_count": 100000, "max_views": 48 }
```

Or on a new scan: `"bake_env_mesh": true` (implies Phase A). CLI:

```bash
python scripts/bake_env_mesh.py <JOB_ID> --target-faces 100000
```

Writes `environment_mesh.glb`, `collider.glb`, and sets manifest `environment.mesh_url` / `collider_url`.

**Stack (commercial):** gsplat depth render (Apache) + NumPy TSDF + scikit-image marching cubes (BSD) + trimesh decimate (MIT). Open3D is not used on DGX aarch64.

**Image-to-world (TripoSplat):** no multi-view cameras → bake returns 400. RP1 for I2W = TRELLIS `props/*.glb` only. Future multi-view I2W (HY-World) can call the same `/bake-env-mesh`.

## Limitations

- LingBot must be installed or jobs fail with an install hint (other models unaffected).
- Default output is a **colored point cloud**; denser trained Gaussians are Phase B.
- Absolute scale needs a measured real-world length — headset passthrough alone is not enough.
- Phase A Gaussians are isotropic (same radius per point) — good for Spark load / XR preview, not final quality.
- Phase B sharpens appearance via photometric gsplat train; quality still depends on pose accuracy and view coverage.
