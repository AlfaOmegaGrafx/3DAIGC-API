Agent entry point: read `CLAUDE.md` in full — it is the operating manual — then follow its
Session protocol. Everything else (state, workflows, project knowledge) is routed from there
via the `.agent/` directory.

Optional MindLink memory lives in `.brain/` (see `.agent/areas/mindlink.md`). Prefer `.agent/`
for engineering state; use `.brain/` for durable cross-session notes when MindLink is active.
