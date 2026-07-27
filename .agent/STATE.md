# State — rewritten in full at session END. Cap: 40 lines.

Session: 14
Focus: Lock prefer_floor gravity + Phase B train API
Active: prefer_floor=True default; repair_world_gravity_alignment; Office jobs repaired
Next: user hard-reload 12a8bc1b / 1a7b74fd; optional Phase B 7k
Blocked: none for pose; sharpness still pose-coverage limited

## Watch-outs
- Never gravity_align without prefer_floor=True on LingBot demo path
- camera_extrinsics+… alone → tilted rooms (Office Train/Train2)
- Metric SVD bake → black spikes — bake OFF; densify OFF on LingBot
- poses_c2w.npy only; sh_degree=0 for Spark DC; orientationMode none
- Keep door metric on parent transform [sx,1,sz]

## Recently shipped
- prefer_floor default + repair helper; Phase B /train-3dgs (2026-07-27)
- Scale sanitize + disable metric bake (2026-07-27)
