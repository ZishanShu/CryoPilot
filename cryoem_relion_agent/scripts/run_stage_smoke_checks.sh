#!/usr/bin/env bash
# Run scan-project, run-next --dry-run, and optionally generate-report on each smoke fixture.
#
# Usage:
#   ./scripts/run_stage_smoke_checks.sh <SMOKE_BASE_DIR> [OUT_DIR]
#
# Environment:
#   WITH_REPORT=1   Also run generate-report (markdown + json per fixture).
#   OUT_DIR         Default if second arg omitted: /tmp/cryoem_stage_smoke_<timestamp>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${#}" -lt 1 ]]; then
  echo "Usage: $0 <SMOKE_BASE_DIR> [OUT_DIR]" >&2
  exit 1
fi

SMOKE_BASE="$(cd "${1}" && pwd)"
OUT_DIR="${2:-${OUT_DIR:-/tmp/cryoem_stage_smoke_$(date +%Y%m%d_%H%M%S)}}"
WITH_REPORT="${WITH_REPORT:-0}"

FIXTURES=(
  picking_ready
  extract_ready
  class2d_ready
  init3d_ready
  refine3d_ready
  postprocess_ready
)

mkdir -p "${OUT_DIR}"
echo "[smoke_checks] SMOKE_BASE=${SMOKE_BASE}"
echo "[smoke_checks] OUT_DIR=${OUT_DIR}"
echo "[smoke_checks] WITH_REPORT=${WITH_REPORT}"

cd "${AGENT_ROOT}"

for name in "${FIXTURES[@]}"; do
  proj="${SMOKE_BASE}/${name}"
  if [[ ! -d "${proj}" ]]; then
    echo "[smoke_checks] skip (missing): ${proj}" >&2
    continue
  fi
  sub="${OUT_DIR}/${name}"
  mkdir -p "${sub}"
  echo "[smoke_checks] --- ${name} ---"
  python3 -m app.main scan-project --project "${proj}" > "${sub}/scan.json" 2> "${sub}/scan.stderr" || true
  python3 -m app.main run-next --project "${proj}" --dry-run > "${sub}/run_next.json" 2> "${sub}/run_next.stderr" || true
  if [[ "${WITH_REPORT}" == "1" ]]; then
    python3 -m app.main generate-report \
      --project "${proj}" \
      --out "${sub}/report.md" \
      --out-json "${sub}/report.json" \
      2> "${sub}/report.stderr" || true
  fi
done

echo "[smoke_checks] done. Summarize with:"
echo "  cd ${AGENT_ROOT} && python3 -m app.stage_smoke_layout summarize --out-dir ${OUT_DIR}"
