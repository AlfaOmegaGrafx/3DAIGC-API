#!/usr/bin/env bash
# Append MSF / RP1 env vars to 3DAIGC-API .env from rp1-spatial-fabric config.
set -euo pipefail

RP1_ENV="/home/sifr/.config/rp1-spatial-fabric/rp1.env"
API_ENV="/home/sifr/3DAIGC-API/.env"
source "$RP1_ENV"

if [[ -n "${MSF_BROWSER_PUBLIC_URL:-}" ]]; then
  PUBLIC_BASE="${MSF_BROWSER_PUBLIC_URL%/}"
else
  PUBLIC_BASE="https://${MSF_PUBLIC_HOST}"
  if [[ "${MSF_PUBLIC_PORT:-443}" != "443" ]]; then
    PUBLIC_BASE="${PUBLIC_BASE}:${MSF_PUBLIC_PORT}"
  fi
fi

touch "$API_ENV"
grep -v '^MSF_PUBLIC_BASE_URL=' "$API_ENV" | grep -v '^MSF_FABRIC_MSF_URL=' | grep -v '^MSF_OBJECTS_DIR=' | grep -v '^RP1_COMPANY_ID=' > "${API_ENV}.tmp" || true
mv "${API_ENV}.tmp" "$API_ENV"

cat >> "$API_ENV" <<EOF
MSF_PUBLIC_BASE_URL=${PUBLIC_BASE}
MSF_FABRIC_MSF_URL=${PUBLIC_BASE}/fabric/sample.msf
MSF_OBJECTS_DIR=/home/sifr/MSF_Map_Svc/dist/web/objects
RP1_COMPANY_ID=${RP1_COMPANY_ID}
EOF

echo "Updated $API_ENV with spatial fabric settings"
