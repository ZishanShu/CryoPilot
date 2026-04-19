#!/usr/bin/env python3
"""CLI: build failure_audit.json / failure_audit.md from latest.json + smoke_run_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.failure_audit import (
    attach_audit_sources,
    audit_dict_to_json_bytes,
    build_failure_audit,
    render_failure_audit_markdown,
)


def _load_json(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit a failed smoke run (local, no LLM API).")
    p.add_argument("--latest-json", required=True, type=Path, help="Path to .cryoem_agent/benchmarks/latest.json")
    p.add_argument("--smoke-summary", required=True, type=Path, help="Path to smoke_run_summary.json")
    p.add_argument("--output-json", type=Path, default=None, help="Default: sibling of smoke summary")
    p.add_argument("--output-md", type=Path, default=None, help="Default: sibling of smoke summary")
    args = p.parse_args(argv)

    latest_path = args.latest_json.resolve()
    smoke_path = args.smoke_summary.resolve()

    latest_payload = _load_json(latest_path)
    smoke_summary = _load_json(smoke_path)

    audit = build_failure_audit(latest_payload, smoke_summary)
    attach_audit_sources(audit, latest_path=str(latest_path), smoke_summary_path=str(smoke_path))

    out_dir = smoke_path.parent
    out_json = args.output_json or (out_dir / "failure_audit.json")
    out_md = args.output_md or (out_dir / "failure_audit.md")

    out_json.write_bytes(audit_dict_to_json_bytes(audit))
    out_md.write_text(render_failure_audit_markdown(audit), encoding="utf-8")
    print(str(out_json))
    print(str(out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
