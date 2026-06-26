# Project Memory

> Permanent facts about this project. Your AI reads this at the start of every session.
> Managed by MindLink — https://github.com/404-not-found/mindlink

---

<!--
  MEMORY.md is a form, not a notebook. Fill in each section — don't free-write.
  Keep Core tight: the agent reads it every session. Extended sections are
  read on demand when the task touches that area.

  Total Core target: under 50 lines. If it grows beyond that, consolidate.
  Merge related entries, remove redundant ones. A bloated memory is as
  useless as no memory.
-->

## Core  <!-- READ EVERY SESSION — keep under 50 lines total -->

### What this project is
**3DAIGC-API** — Python FastAPI backend for 3D AIGC jobs (image/text-to-3D, rigging, segmentation, Kimodo text-to-motion, XR voice, spatial fabric / RP1 publish). <!-- added: 2026-06-26 -->
Serves OpenNexus3DStudio on Surface via LAN; runs on DGX Spark.

### Stack
Python 3.10+, FastAPI, Redis job queue, model adapters in `adapters/`, config in `config/models.yaml`
Entry: `main.py` | Dev: `scripts/run_server.sh` (:7842) | XR stack: `mcp/scripts/run_xr_ai_3daigc_stack.sh`

### Top decisions
- Agents execute all ops on DGX — never delegate scripts to user <!-- added: 2026-06-26 -->
- API contract changes require matching OpenNexus3DStudio client updates <!-- added: 2026-06-26 -->
- Kimodo: `worker_load_timeout_sec=3600`, `TEXT_ENCODER_DEVICE=cpu` on Spark <!-- added: 2026-06-26 -->
- MSF `MSF_PUBLIC_BASE_URL` / browser: `https://10.0.0.32:8453` (Surface proxy) <!-- added: 2026-06-26 -->
- DGX ↔ Surface: scp sync only; no agent git push unless user asks <!-- added: 2026-06-26 -->

### Current focus
Kimodo text-to-motion operational; spatial fabric MSF on :8443; MindLink installed Jun 26 2026.
DGX IP `10.0.0.158`, Surface `10.0.0.32`.

---

## Architecture  <!-- Read when the task involves project structure -->

| Service | Port | Script |
|---------|------|--------|
| 3DAIGC-API | 7842 | `scripts/run_server.sh` |
| MSF map svc | 8443 | `scripts/run-msf-map-svc.sh` |
| XR AI hub | 8088 | `mcp/scripts/run_xr_ai_3daigc_stack.sh` |

Active dirs: `api/`, `adapters/`, `core/`, `config/`, `scripts/`, `mcp/`
MCP server for Cursor: `mcp/` via `uv run 3daigc-mcp`
Docs: `docs/api_documentation.md` | Memory-bank: `memory-bank/`

---

## Decisions  <!-- Read when making a choice, or when unsure why something is the way it is -->

| Decision | What was decided | Why |
|---|---|---|
| MSF public URL | `https://10.0.0.32:8453` in `.env` | Tailscale hostname unreachable without funnel |
| Kimodo cold start | 3600s worker load timeout | First run downloads Llama-3-8B shards |
| MindLink | `.brain/` git-tracked | Team memory alongside SessionMem + memory-bank |
| Solid Skills | `.agents/skills/solid` + `skills-lock.json` | SOLID/clean-code skill; project rules override strict TDD <!-- added: 2026-06-26 --> |

---

## Conventions  <!-- Read when writing code -->

- Code quality: `.cursor/rules/solid-skills.mdc` + `.agents/skills/solid/` — SOLID for adapters/routers; minimize scope + meaningful tests over mandatory TDD <!-- added: 2026-06-26 -->
- New model: `config/models.yaml` + adapter + verify `/api/v1/system/models` + sync OpenNexus `aiModelsCatalog.js`
- Restart API after config/adapter changes; Redis required
- `scripts/sync-spatial-fabric-env.sh` honors `MSF_BROWSER_PUBLIC_URL`

---

## User Profile  <!-- READ EVERY SESSION — personal facts about the user -->

3DAIGC / OpenNexus developer; DGX Spark primary backend host. Agents execute ops directly.

---

## Important Context  <!-- Read when something feels off or context is missing -->

- Before declaring Kimodo broken: verify all 4 Llama safetensor shards in HF cache
- Never commit `.env` (contains secrets); agent has no GitHub push access
- XR remote log: `/home/sifr/logs/xr-remote-log.txt` (needs hub `?remoteLog=1`)
- Surface git: no GitHub SSH key — pull via HTTPS over `ssh Surface-PC-Tailscale`; DGX has GitHub SSH for push <!-- added: 2026-06-26 -->
