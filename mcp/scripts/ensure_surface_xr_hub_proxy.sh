#!/usr/bin/env bash
# Start Surface xr-hub-proxy (:8443 → DGX :8088) for Galaxy XR client isolation.
# MUST use Windows scheduled task — Start-Process / start /B via SSH dies within seconds.
#
# Avoids nested `powershell -Command "..."` quoting (breaks over SSH). Flow:
#   1) write runner + boot PS1 locally
#   2) scp to Surface logs/
#   3) ssh powershell.exe -File boot.ps1  (registers + runs scheduled task)
set -euo pipefail

SURFACE_SSH="${SURFACE_SSH:-Surface-PC-Tailscale}"
SURFACE_ROOT="${SURFACE_ROOT:-C:/Users/alfao/Documents/GitHub/OpenNexus3DStudio}"
SURFACE_ROOT_WIN="${SURFACE_ROOT_WIN:-C:\\Users\\alfao\\Documents\\GitHub\\OpenNexus3DStudio}"
SURFACE_IP="${SURFACE_IP:-10.0.0.32}"
PROXY_PORT="${XR_PROXY_PORT:-8443}"
DGX_HUB="${XR_SPARK_HUB_URL:-https://10.0.0.158:8088}"
TASK_NAME="${XR_HUB_PROXY_TASK:-OpenNexusXRHubProxy}"

probe() {
  curl -sk --connect-timeout 5 -o /dev/null -w '%{http_code}' "https://${SURFACE_IP}:${PROXY_PORT}/" 2>/dev/null || echo "000"
}

code="$(probe)"
if [[ "$code" == "200" ]]; then
  echo "OK  Surface XR proxy https://${SURFACE_IP}:${PROXY_PORT} → ${DGX_HUB} (HTTP ${code})"
  exit 0
fi

echo "==> Surface XR proxy not responding (got HTTP ${code}) — starting via scheduled task"

if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$SURFACE_SSH" "echo ok" >/dev/null 2>&1; then
  echo "FAIL: Cannot SSH to ${SURFACE_SSH}" >&2
  exit 1
fi

TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

# Long-lived proxy process (invoked by scheduled task).
cat >"$TMPDIR_LOCAL/start-xr-hub-proxy.ps1" <<EOF
Set-Location '${SURFACE_ROOT_WIN}'
\$env:XR_SPARK_HUB_URL = '${DGX_HUB}'
\$env:XR_PROXY_PORT = '${PROXY_PORT}'
New-Item -ItemType Directory -Force -Path (Join-Path '${SURFACE_ROOT_WIN}' 'logs') | Out-Null
node scripts/xr-spark-hub-proxy.mjs *>> logs/xr-hub-proxy.log
EOF

# One-shot boot: firewall + register task + run (no nested SSH quoting).
cat >"$TMPDIR_LOCAL/boot-xr-hub-proxy.ps1" <<EOF
\$ErrorActionPreference = 'Continue'
\$r = '${SURFACE_ROOT_WIN}'
\$port = ${PROXY_PORT}
\$task = '${TASK_NAME}'
\$runner = Join-Path \$r 'logs\\start-xr-hub-proxy.ps1'

New-Item -ItemType Directory -Force -Path (Join-Path \$r 'logs') | Out-Null

if (-not (Get-NetFirewallRule -DisplayName 'OpenNexus XR Hub Proxy' -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -DisplayName 'OpenNexus XR Hub Proxy' -Direction Inbound -Action Allow -Protocol TCP -LocalPort \$port -Profile Private,Domain | Out-Null
  Write-Output 'Added firewall rule OpenNexus XR Hub Proxy'
}

# Drop stale listeners on the proxy port (keep other owners if not node/powershell).
Get-NetTCPConnection -LocalPort \$port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
  \$p = Get-Process -Id \$_.OwningProcess -ErrorAction SilentlyContinue
  if (\$p -and (\$p.ProcessName -match '^(node|powershell|pwsh)$')) {
    try { Stop-Process -Id \$p.Id -Force -ErrorAction SilentlyContinue } catch {}
  }
}

\$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -ExecutionPolicy Bypass -File "' + \$runner + '"') -WorkingDirectory \$r
\$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddSeconds(2))
\$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName \$task -Action \$action -Trigger \$trigger -Settings \$settings -Force | Out-Null
schtasks /Run /TN \$task | Out-Null
Write-Output 'Scheduled task started'
EOF

scp -o ConnectTimeout=15 \
  "$TMPDIR_LOCAL/start-xr-hub-proxy.ps1" \
  "$TMPDIR_LOCAL/boot-xr-hub-proxy.ps1" \
  "${SURFACE_SSH}:${SURFACE_ROOT}/logs/" >/dev/null

ssh -o ConnectTimeout=15 "$SURFACE_SSH" \
  "powershell.exe -NoProfile -ExecutionPolicy Bypass -File ${SURFACE_ROOT_WIN}\\logs\\boot-xr-hub-proxy.ps1"

for _ in $(seq 1 12); do
  sleep 2
  code="$(probe)"
  if [[ "$code" == "200" ]]; then
    echo "OK  Surface XR proxy https://${SURFACE_IP}:${PROXY_PORT} → ${DGX_HUB} (HTTP ${code})"
    exit 0
  fi
done

echo "FAIL: Surface proxy still not responding (HTTP ${code})" >&2
echo "Check: ssh ${SURFACE_SSH} powershell.exe -NoProfile -Command \"Get-Content ${SURFACE_ROOT_WIN}\\logs\\xr-hub-proxy.log -Tail 20\"" >&2
exit 1
