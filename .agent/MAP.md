# Map — where things live. First stop when locating code; grep comes after, wholesale reading never.
<!-- One line per module: `path — what it is; entry: <file>`. Update on any structure change.
     Cap 120 lines: when over, collapse a subtree into .agent/areas/<x>.md and keep one line here
     pointing at it. `(?)` marks unverified bootstrap guesses — verify on first visit, then remove. -->

.agent/ — RepoResident harness: STATE, MAP, PROJECT, DECISIONS, workflows, journal
.brain/ — MindLink persistent memory (optional; see .agent/areas/mindlink.md)
adapters/ — per-model inference adapters (TRELLIS, Kimodo, LingBot, UniRig, …); entry: each `*_adapter.py`
api/ — FastAPI app + routers; entry: `api/main_multiworker.py`, `api/routers/`
assets/ — example meshes/VRM templates (e.g. `example_autorig/`)
config/ — `models.yaml` feature/model registry + runtime config
core/ — scheduler, utils, pipelines (e.g. `core/utils/lingbot_map_pipeline.py`, `metric_scale.py`)
docs/ — operator + API docs (`LOCAL_DEPLOYMENT.md`, `LINGBOT_MAP_ENVIRONMENT_SCAN.md`, avatar contracts)
mcp/ — MCP server + XR voice helper scripts
memory-bank/ — internal ops notes (some paths gitignored)
pretrained/ — local model weights roots `(?)`
scripts/ — DGX ops: start/stop, redis, lingbot install, blender, env_local_gpu
tests/ — pytest suite
thirdparty/ — vendored model trees (lingbot-map, TRELLIS, …)
uploads/ · outputs/ · logs/ · run/ — runtime artifacts (not source of truth)
venv/ — Python virtualenv (do not commit)
docker-compose.yml · Dockerfile — container deploy path
