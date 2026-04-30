from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def extract_json_from_markdown(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    root = args.run_root.resolve()
    summary = load_json(root / "workflow_summary.json")
    rule_rows = load_jsonl(root / "rule_based/stage_decisions.jsonl")

    # 优先读 stagewise LLM；如果没有，就退回旧版全局 LLM。
    llm_stagewise = load_json(root / "llm_supervised/llm_stagewise_result.json")
    llm_global = extract_json_from_markdown(root / "llm_supervised/llm_supervised_result.md")

    metrics = summary.get("metrics", {})
    artifacts = summary.get("artifacts", {})
    stage_status = summary.get("stage_status", {})

    rule_by_stage = {r.get("stage", ""): r for r in rule_rows}

    # 如果有真正逐阶段 LLM 结果，读它。
    llm_by_stage = {}
    if isinstance(llm_stagewise.get("stage_assessments"), list):
        for r in llm_stagewise["stage_assessments"]:
            llm_by_stage[r.get("stage", "")] = r

    # 旧版 LLM 只有全局 judgment，所以这里作为 fallback。
    global_accept_exec = llm_global.get("accept_backend_execution", "")
    global_accept_downstream = llm_global.get("accept_for_downstream", "")
    global_risk = llm_global.get("risk_level", "")
    global_issue = llm_global.get("main_issue", "")
    global_action = llm_global.get("recommended_next_action", "")

    stages = ["particle_picking", "particle_extraction", "reconstruction", "map2pdb"]

    # 兼容你 rule JSONL 里面的 stage 名。
    alias = {
        "particle_picking": "particle_picking",
        "particle_extraction": "particle_extraction",
        "reconstruction": "reconstruction",
        "map2pdb": "map2pdb",
    }

    stage_metric_map = {
        "particle_picking": {
            "primary_metric": f"topaz_coords_used={metrics.get('total_topaz_coords_used')}",
            "artifact": "Topaz .coord / converted .box coordinates",
        },
        "particle_extraction": {
            "primary_metric": f"extracted_particles={metrics.get('n_extracted_particles')}; particle_stacks={metrics.get('n_particle_stacks')}",
            "artifact": artifacts.get("particles_star", ""),
        },
        "reconstruction": {
            "primary_metric": f"map_exists={metrics.get('reconstruction_map_exists')}; map_size_bytes={metrics.get('reconstruction_map_size_bytes')}",
            "artifact": artifacts.get("reconstruction_map", ""),
        },
        "map2pdb": {
            "primary_metric": "dry_run_recorded=true; cif_or_pdb_generated=false",
            "artifact": artifacts.get("map2pdb_action_history", ""),
        },
    }

    rows = []
    for stage in stages:
        rule = rule_by_stage.get(alias[stage], {})
        llm = llm_by_stage.get(stage, {})

        if llm:
            llm_accept_exec = llm.get("accept_backend_execution", "")
            llm_accept_downstream = llm.get("accept_for_downstream", "")
            llm_risk = llm.get("risk_level", "")
            llm_issue = llm.get("main_issue", "")
            llm_action = llm.get("recommended_next_action", "")
        else:
            llm_accept_exec = global_accept_exec
            llm_accept_downstream = global_accept_downstream
            llm_risk = global_risk
            llm_issue = "Global LLM audit only. " + str(global_issue)
            llm_action = global_action

        rows.append({
            "stage": stage,
            "real_backend": rule.get("selected_backend", ""),
            "rule_based_status": rule.get("status", stage_status.get(stage.replace("particle_", ""), "")),
            "rule_based_reason": rule.get("reason", ""),
            "rule_return_code": rule.get("return_code", ""),
            "rule_runtime_sec": rule.get("runtime_sec", ""),
            "primary_metric": stage_metric_map[stage]["primary_metric"],
            "main_artifact": stage_metric_map[stage]["artifact"],
            "llm_accept_backend_execution": llm_accept_exec,
            "llm_accept_for_downstream": llm_accept_downstream,
            "llm_risk_level": llm_risk,
            "llm_main_issue": llm_issue,
            "llm_recommended_next_action": llm_action,
        })

    out_md = args.out_md or (root / "stage_compare_report.md")
    out_csv = args.out_csv or (root / "stage_compare_table.csv")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = []
    md.append("# Same-Dataset Agent Stage Comparison Report\n")
    md.append("## Case summary\n")
    md.append(f"- Dataset: {summary.get('dataset_id', '')}")
    md.append(f"- Workflow: {summary.get('workflow', '')}")
    md.append(f"- Run root: `{root}`")
    md.append(f"- Micrographs: {metrics.get('n_micrographs')}")
    md.append(f"- Topaz coordinates used: {metrics.get('total_topaz_coords_used')}")
    md.append(f"- Extracted particles: {metrics.get('n_extracted_particles')}")
    md.append(f"- Particle stack count: {metrics.get('n_particle_stacks')}")
    md.append(f"- Reconstruction map exists: {metrics.get('reconstruction_map_exists')}")
    md.append(f"- Reconstruction map size bytes: {metrics.get('reconstruction_map_size_bytes')}")
    md.append("")

    md.append("## Stage-by-stage comparison\n")
    md.append("| Stage | Backend | Rule-based status | Primary metric | LLM accepts execution | LLM accepts downstream | LLM risk |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        md.append(
            f"| {r['stage']} | {r['real_backend']} | {r['rule_based_status']} | "
            f"{r['primary_metric']} | {r['llm_accept_backend_execution']} | "
            f"{r['llm_accept_for_downstream']} | {r['llm_risk_level']} |"
        )

    md.append("")
    md.append("## Detailed stage notes\n")
    for r in rows:
        md.append(f"### {r['stage']}\n")
        md.append(f"- Real backend: `{r['real_backend']}`")
        md.append(f"- Rule-based status: `{r['rule_based_status']}`")
        md.append(f"- Rule-based reason: {r['rule_based_reason']}")
        md.append(f"- Primary metric: `{r['primary_metric']}`")
        md.append(f"- Main artifact: `{r['main_artifact']}`")
        md.append(f"- LLM accepts backend execution: `{r['llm_accept_backend_execution']}`")
        md.append(f"- LLM accepts downstream scientific use: `{r['llm_accept_for_downstream']}`")
        md.append(f"- LLM risk level: `{r['llm_risk_level']}`")
        md.append(f"- LLM main issue: {r['llm_main_issue']}")
        md.append(f"- LLM recommended next action: {r['llm_recommended_next_action']}")
        md.append("")

    md.append("## Interpretation\n")
    md.append(
        "This report is data-oriented. It separates real backend execution, rule-based workflow progression, "
        "and LLM-supervised scientific auditing. A stage can be technically successful while still being rejected "
        "for scientific downstream use by the LLM supervisor."
    )
    md.append("")
    md.append(
        "If Map2PDB was run in dry-run mode, this report records routing and audit evidence only. "
        "A real map-to-model result requires `--map2pdb-mode real` and a generated CIF/PDB artifact."
    )
    md.append("")
    md.append(f"- CSV table: `{out_csv}`")

    out_md.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "out_md": str(out_md),
        "out_csv": str(out_csv),
        "n_rows": len(rows),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
