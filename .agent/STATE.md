# State — rewritten in full at session END. Cap: 40 lines.

Session: 23
Focus: Moat — memory-bank fully local
Active: shipped — git rm --cached memory-bank/; .gitignore memory-bank/; files remain on disk for agents
Next: include untrack + gitignore in next API commit when user asks
Blocked: none

## Watch-outs
- memory-bank/ is moat — scp sync only
- GNM origin = AlfaOmegaGrafx/GNM
- Do not drain GPU jobs unless queue empty

## Recently shipped
- memory-bank/ fully gitignored
- Prior: GNM fork remote; MESH_WRAP root moat
