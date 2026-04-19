#!/usr/bin/env bash
# Pretty-print a one-line table from scan.json / run_next.json produced by run_stage_smoke_checks.sh.
#
# Usage:
#   ./scripts/summarize_stage_smoke.sh <OUT_DIR>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${#}" -ne 1 ]]; then
  echo "Usage: $0 <OUT_DIR>" >&2
  exit 1
fi

cd "${AGENT_ROOT}"
exec python3 -m app.stage_smoke_layout summarize --out-dir "${1}"
