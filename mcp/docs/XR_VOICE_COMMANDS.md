# Galaxy XR Voice Commands — 3daigc-vlm-example

Reference for the **OpenNexus3DStudio + 3DAIGC-API** voice agent on Galaxy XR.

**Bookmark:** `https://10.0.0.32:8443/?remoteLog=1` (Surface proxy → DGX hub)

---

## Before you speak

1. **Connect** → **Start Camera** (passthrough) → **Start Mic**
2. Start every new request with **`"Agent"`** or **`"Hey agent"`**
3. Follow-ups within **5 seconds** can skip the wake phrase
4. Jobs run on **DGX Spark**; results appear in **OpenNexus3DStudio** via **Sync DGX** (see below)

---

## Wake phrase & control

| Say | Effect |
|-----|--------|
| `Hey agent, …` | Wake the agent (required for first utterance) |
| `Agent, …` | Same |
| `Stop` / `Be quiet` / `Shut up` | Interrupt the agent (no wake phrase needed) |

---

## Camera → 3D mesh (image-to-textured-mesh)

Point the camera at the object, then say:

| Example phrase |
|----------------|
| **Hey agent, make a 3D model of this** |
| Agent, create a 3D mesh of what I see |
| Turn this into a 3D model |
| Generate a three-dimensional model |
| Build a mesh from the camera |
| Image to 3D |
| Model this |
| 3D model of this |

**Pipeline:** camera snapshot → `upload_image` → `image_to_textured_mesh` → wait (minutes) → GLB on DGX.

---

## Text → 3D mesh (text-to-textured-mesh)

No camera required. Describe the object in words:

| Example phrase |
|----------------|
| **Hey agent, make a 3D model of a red sports car** |
| Agent, create a 3D mesh of a wooden chair |
| Text to 3D: a small dragon figurine |
| Generate a 3D model of a coffee mug |

**Tip:** Avoid words like *this*, *camera*, or *what I see* — those route to the camera pipeline instead.

---

## Camera → explorable world (image-to-world)

Point the camera at a scene, then say:

| Example phrase |
|----------------|
| **Hey agent, make a 3D world from this** |
| Agent, create an environment from the camera view |
| Build an explorable scene from what I see |
| Image to world |
| Generate a world from this view |

**Pipeline:** camera snapshot → `image_to_world` → world package (splat env + props).

Load in OpenNexus3DStudio after **Sync DGX** (image-to-world task type).

---

## Auto-rigging (generate-rig)

Rigs the **last 3D model** from this session (camera or text mesh):

| Example phrase |
|----------------|
| **Hey agent, rig this** |
| Agent, auto rig the model |
| Add a skeleton to it |
| Rig the model |

**Requires:** a completed mesh job earlier in the same session. Uses `rig_mode=template` (humanoid VRM pipeline).

---

## Open in OpenNexus3DStudio (automatic handoff)

When you ask for a 3D model from the camera, the agent **queues the job on DGX** and shows an **Open OpenNexus3DStudio** link on the XR page.

### Recommended flow (Galaxy XR browser → Surface Vite)

1. On the **XR AI hub** (`dgx-spark…ts.net`), say **“Hey agent, make a 3D model of this.”**
2. Tap **Open OpenNexus3DStudio** on the banner (or open the spoken URL).
3. The studio opens on **Surface PC** (`https://100.94.108.18:3000/?jobId=…&autoLoad=1&tasks=1` over Tailscale).
4. Task Manager shows the job as **• XR • DGX** with the same progress bar as a native task.
5. When DGX finishes, the model **auto-loads** in the viewport.

Configure the studio URL in `3daigc_vlm_example_worker.yaml`:

```yaml
opennexus_studio_url: https://100.94.108.18:3000
```

Surface must run **Vite with HTTPS** (`npm run dev`) and `VITE_API_ENDPOINT` (or dev proxy) must reach DGX `:7842`.

### Manual fallback (PC only)

1. Open **OpenNexus3DStudio** on Surface
2. **Task Manager → Sync DGX**
3. Click the **• DGX** row to load

---

## Vision Q&A (no 3D generation)

| Example phrase |
|----------------|
| Hey agent, what am I looking at? |
| Describe this |
| What color is the lamp? |
| What does this sign say? |
| Is the door open? |

---

## General knowledge (ignores camera)

| Example phrase |
|----------------|
| Hey agent, what's the capital of France? |
| Tell me a joke |
| Explain entropy |

---

## Data channel topics (advanced)

The worker publishes JSON on these topics when jobs finish:

| Topic | Feature |
|-------|---------|
| `3daigc.meshResult` | image/text → textured mesh |
| `3daigc.worldResult` | image → world |
| `3daigc.rigResult` | auto-rig |
| `3daigc.jobResult` | all features (generic) |

OpenNexus3DStudio today uses **API job history sync**, not these data messages.

---

## Troubleshooting

| Issue | Check |
|-------|--------|
| Agent ignores you | Start with `Hey agent` |
| Camera mesh fails | Camera on, passthrough view, stable connection |
| Text mesh routes to camera | Don't say *this* / *what I see* |
| Rig fails | Run a mesh command first in the same session |
| Nothing in studio after sync | API connected, job `completed`, **Sync DGX**, task < 24h old |
| Headset can't reach DGX | Use Surface proxy bookmark, media relay on Surface |

---

## Related docs

- [NVIDIA_XR_AI_INTEGRATION.md](NVIDIA_XR_AI_INTEGRATION.md) — stack architecture
- [api/api.md](api/api.md) — 3DAIGC-API REST reference
- Worker config: `xr-ai/agent-samples/3daigc-vlm-example/yaml/3daigc_vlm_example_worker.yaml`
