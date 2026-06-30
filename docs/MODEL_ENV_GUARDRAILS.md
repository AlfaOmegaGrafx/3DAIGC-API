# Model environment guardrails (3DAIGC-API)

Prevent silent breakage when pip, HuggingFace hub models, or adapter code drift — for **every** enabled model, not only TRELLIS.2.

## Defense layers

| Layer | Script / file | When |
|-------|----------------|------|
| Pip pins (HF) | `scripts/constraints-hf.txt` | Every main-venv install |
| Pip pins (runtime) | `scripts/constraints-models-runtime.txt` | diffusers, peft, accelerate, omegaconf |
| Kimodo isolated pins | `scripts/constraints-kimodo.txt` | `.venv-kimodo` only |
| Guarded pip | `scripts/pip_main_venv.sh` | **Use instead of raw `pip install`** |
| Post-pip gate | `scripts/post_pip_guard.sh` | After any manual pip into `venv/` |
| Main venv drift | `scripts/check_venv_drift.sh` | Install end, API startup |
| Kimodo drift | `scripts/check_kimodo_venv_drift.sh` | `setup_kimodo.sh`, weekly health |
| HF conditioning | `scripts/verify_hf_conditioning.py` | TRELLIS.2 BiRefNet + DINOv3 (runtime mock + optional GPU) |
| API restart | `scripts/restart_services.sh` | Runs HF quick verify unless `P3D_SKIP_PREFLIGHT=1` |
| Registry validate | `config/verify_profiles.yaml` + `verify_registry.py --validate` | CI, install, preflight |
| Adapter import | `verify_registry.py --tier quick --all-enabled` | API preflight |
| Full inference | `verify_all_enabled_models.sh` | Release / weekly (all 25 enabled) |
| Weekly smoke | `scripts/weekly_model_health.sh` | Cron / manual |

## Adding or enabling a model (required checklist)

1. Add adapter to `core/scheduler/model_factory.py` `ADAPTER_REGISTRY`.
2. Add entry to `config/models.yaml` with `enabled: true`.
3. **Add verify profile** to `config/verify_profiles.yaml` (module, class, inputs, timeout).
4. Run `python scripts/verify_registry.py --validate` — must pass before merge.
5. Pin any new critical pip deps in `constraints-hf.txt` or `constraints-models-runtime.txt`.
6. If model uses **separate venv** (like Kimodo): add `constraints-<stack>.txt` + drift script.
7. Run canary: `verify_registry.py --tier infer --model <id>` before production.

**Preflight will fail** if an enabled model has no verify profile.

## Daily commands (DGX)

```bash
# After ANY pip install into venv/
bash scripts/pip_main_venv.sh install <package>   # preferred
# or:
bash scripts/post_pip_guard.sh

# API startup (automatic in run_server.sh)
./venv/bin/python scripts/verify_env_compat.py

# Weekly health (quick — no heavy GPU infer)
bash scripts/weekly_model_health.sh

# Full GPU matrix (~hours)
bash scripts/verify_all_enabled_models.sh

# TRELLIS.2 fast canary after HF changes
bash scripts/verify_trellis2_canary.sh
P3D_HF_VERIFY_GPU=1 ./venv/bin/python scripts/verify_hf_conditioning.py --gpu
```

## Constraint files

```bash
./venv/bin/pip install \
  -c scripts/constraints-hf.txt \
  -c scripts/constraints-models-runtime.txt \
  -r requirements.txt
```

## API startup

`scripts/run_server.sh` runs drift + full preflight unless:

```bash
P3D_SKIP_PREFLIGHT=1 ./scripts/run_server.sh
```

## Stack isolation

| Stack | Venv | transformers |
|-------|------|----------------|
| Main API (TRELLIS, Hunyuan, Krea, …) | `venv/` | `4.57.3` (pinned) |
| Kimodo | `.venv-kimodo` | `5.1.0` (pinned) |

**Krea 2:** `setup_krea2.sh` installs a **pinned diffusers git ref** (`DIFFUSERS_KREA_GIT_REF`) because `Krea2Pipeline` is not in PyPI wheels yet. Do not `pip install -U transformers` outside constraints.

## Agent rules

See `.cursor/rules/model-env-guardrails.mdc`.
