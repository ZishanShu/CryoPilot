#!/usr/bin/env bash
# Create six stage smoke copies from a full RELION project (see docs/stage_smoke_validation.md).
#
# Usage:
#   ./scripts/make_stage_smoke_projects.sh <SOURCE_RELION_PROJECT> <DEST_BASE_DIR>
#
# DEST_BASE_DIR will contain: picking_ready/, extract_ready/, ... each a pruned copy.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${#}" -ne 2 ]]; then
  echo "Usage: $0 <SOURCE_RELION_PROJECT> <DEST_BASE_DIR>" >&2
  exit 1
fi

SOURCE="${1}"
DEST_BASE="${2}"

cd "${AGENT_ROOT}"
exec python3 -m app.stage_smoke_layout create --source "${SOURCE}" --dest-base "${DEST_BASE}"
