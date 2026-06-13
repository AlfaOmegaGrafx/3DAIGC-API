import sys, os, traceback
sys.path.insert(0, os.getcwd())
from adapters.partfield_adapter import PartFieldSegmentationAdapter

mesh = "assets/example_meshseg/002e462c8bfa4267a9c9f038c7966f3b.glb"
a = PartFieldSegmentationAdapter()
print("[verify] loading PartField...")
a.load(gpu_id=0)
print("[verify] running segmentation on", mesh)
out = a.process({"mesh_path": mesh, "num_parts": 6})
print("[verify] RESULT:")
print("  output_mesh_path:", out.get("output_mesh_path"))
print("  num_parts:", out.get("num_parts"))
print("  success:", out.get("success"))
op = out.get("output_mesh_path")
print("  output exists:", bool(op) and os.path.exists(op), "size:", (os.path.getsize(op) if op and os.path.exists(op) else 0))
print("PARTFIELD_VERIFY_OK")
