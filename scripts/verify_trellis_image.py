import sys, os, time
sys.path.insert(0, os.getcwd())
from adapters.trellis_adapter import TrellisImageToTexturedMeshAdapter

img = "assets/example_image/typical_creature_robot_dinosour.png"
a = TrellisImageToTexturedMeshAdapter()
print("[verify] loading TRELLIS image-large...", flush=True)
t0 = time.time()
a.load(gpu_id=0)
print(f"[verify] loaded in {time.time()-t0:.1f}s; running generation on {img}", flush=True)
t1 = time.time()
out = a.process({"image_path": img, "seed": 1, "output_format": "glb", "simplify": 0.95})
print(f"[verify] generation done in {time.time()-t1:.1f}s", flush=True)
op = out.get("output_mesh_path")
print("  output_mesh_path:", op)
print("  output exists:", bool(op) and os.path.exists(op), "size:", (os.path.getsize(op) if op and os.path.exists(op) else 0))
print("TRELLIS_IMAGE_VERIFY_OK")
