# Decisions — binding choices. Append-only. Format: `YYYY-MM-DD · <decision> · why: <reason>`
<!-- Cap 80 lines. Older entries: move to journal or archive note; keep active constraints here. -->

2026-08-14 · Entire `memory-bank/` is gitignored moat (untracked); ops docs sync via scp only · why: align with OpenNexus; stop ops/strategy leakage in public git
2026-08-14 · Vendored GNM pulls from https://github.com/AlfaOmegaGrafx/GNM (origin); google/GNM is upstream compare-only · why: user fork is canonical for Spacetime/API vendor updates
2026-07-26 · LingBot env-scan orientation order is floor/camera→+Y, densest-slab Y-flip, seat, X-mirror, horizontal door scale · why: Office walk twin must match physical left/right and door width without stretching height; inverted camera-up from windowed poses must not win
2026-07-26 · LingBot Phase A Gaussians load via Spark with orientationMode none · why: XYZRGB parser on 68-byte vertices scatters points; TripoSplat X-flip undoes gravity align
2026-06-26 · Agents execute DGX ops directly · why: user is often on Surface; DGX is the runtime host
2026-06-26 · API contract changes require OpenNexus3DStudio client updates · why: shared job/result shapes
2026-06-26 · Kimodo uses long worker_load_timeout + TEXT_ENCODER_DEVICE=cpu on Spark · why: load/VRAM stability on aarch64
2026-06-26 · DGX↔Surface sync is scp scripts, not agent git push · why: keep publish intentional
2026-07-22 · LingBot-Map is optional in-process model, not a daemon · why: same :7842 job path as other adapters; reboot script stays API+MSF
2026-07-22 · Adopt RepoResident harness in `.agent/` · why: structured maps/workflows for large multi-model repo; MindLink kept in `.brain/` for personal memory
2026-07-23 · CLAUDE.md is RepoResident operating manual; MindLink full ritual archived to `.agent/areas/mindlink.md` · why: avoid double conflicting session protocols
