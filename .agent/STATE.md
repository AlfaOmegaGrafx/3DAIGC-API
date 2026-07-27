# State — rewritten in full at session END. Cap: 40 lines.

Session: 15
Focus: Office Train2 8672869c repaired; gravity lock already on origin
Active: all Office worlds floor_ransac; no camera_extrinsics left
Next: Phase B 7–10k on best world for sharpness; denser rescan optional
Blocked: none for pose

## Watch-outs
- Newest-named job ≠ last-repaired ID (Office Train2=8672869c vs Office 3DGS Train2=1a7b74fd)
- prefer_floor=True locked in 14c5872; densify OFF; no metric SVD bake
- data_factor=2 sharper than 4; poses_c2w.npy only; orientationMode none

## Recently shipped
- Repaired 8672869c Office Train2 → matches 65360950 (2026-07-27)
- prefer_floor default + repair helper + Phase B /train-3dgs (14c5872)
