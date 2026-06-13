"""Generic single-model verification harness.

Usage:
  python scripts/verify_model.py <module> <Class> '<json_inputs>'

Example:
  python scripts/verify_model.py adapters.partpacker_adapter \
      PartPackerImageToRawMeshAdapter '{"image_path": "assets/example_image/203.png"}'
"""
import sys, os, json, time, importlib
sys.path.insert(0, os.getcwd())

from core.utils.gpu_env import apply_local_gpu_env

apply_local_gpu_env()

mod_name, cls_name, inputs_json = sys.argv[1], sys.argv[2], sys.argv[3]
inputs = json.loads(inputs_json)

mod = importlib.import_module(mod_name)
cls = getattr(mod, cls_name)
a = cls()
print(f"[verify] loading {cls_name} ...", flush=True)
t0 = time.time()
a.load(gpu_id=0)
print(f"[verify] loaded in {time.time()-t0:.1f}s; processing {inputs}", flush=True)
t1 = time.time()
out = a.process(inputs)
print(f"[verify] done in {time.time()-t1:.1f}s", flush=True)
for k in ("output_mesh_path", "output_path", "segmentation_info_path", "success"):
    if k in out:
        print(f"  {k}: {out[k]}")
op = out.get("output_mesh_path") or out.get("output_path")
if op:
    print("  output exists:", os.path.exists(op), "size:", (os.path.getsize(op) if os.path.exists(op) else 0))
print("VERIFY_OK")
