from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


def sanitize_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    key = key.replace("“", "").replace("”", "")
    m = re.search(r"(sk-[A-Za-z0-9_\-]+)", key)
    if m:
        key = m.group(1)
    key.encode("ascii")
    if not key.startswith("sk-"):
        raise ValueError("OPENAI_API_KEY does not look valid")
    return key


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


def extract_json(text: str) -> Dict[str, Any]:
    # 支持 ```json ... ``` 或普通 JSON
    m = re.search(r"```json\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            text = m.group(0)
    return json.loads(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--model", default=os.environ.get("CRYOAGENT_LLM_MODEL", "gpt-5.5"))
    args = ap.parse_args()

    root = args.run_root.resolve()
    summary_path = root / "workflow_summary.json"
    rule_path = root / "rule_based/stage_decisions.jsonl"
    out_dir = root / "llm_supervised"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rule_rows = load_jsonl(rule_path)

    prompt = {
        "task": "Perform a stage-wise LLM-supervised audit for a same-dataset cryo-EM agent workflow.",
        "strict_output_format": {
            "stage_assessments": [
                {
                    "stage": "particle_picking | particle_extraction | reconstruction | map2pdb",
                    "accept_backend_execution": "boolean",
                    "accept_for_downstream": "boolean",
                    "risk_level": "low | medium | high",
                    "main_issue": "string",
                    "recommended_next_action": "string"
                }
            ],
            "overall_assessment": {
                "accept_backend_execution": "boolean",
                "accept_for_scientific_claim": "boolean",
                "risk_level": "low | medium | high",
                "summary": "string"
            }
        },
        "important_instruction": (
            "Separate technical backend execution from scientific acceptance. "
            "A stage may be accepted as successfully executed but rejected for downstream scientific use. "
            "Return JSON only. No markdown."
        ),
        "workflow_summary": summary,
        "rule_based_stage_decisions": rule_rows,
    }

    prompt_path = out_dir / "llm_stagewise_prompt.json"
    prompt_path.write_text(json.dumps(prompt, indent=2, ensure_ascii=False), encoding="utf-8")

    os.environ["OPENAI_API_KEY"] = sanitize_key()

    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=args.model,
        input=[
            {
                "role": "system",
                "content": "You are a conservative cryo-EM workflow auditor. Return valid JSON only."
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False, indent=2)
            }
        ],
    )

    text = getattr(response, "output_text", "") or str(response)

    raw_path = out_dir / "llm_stagewise_raw.txt"
    raw_path.write_text(text, encoding="utf-8")

    try:
        parsed = extract_json(text)
    except Exception as e:
        parsed = {
            "parse_error": repr(e),
            "raw_text": text,
        }

    result_path = out_dir / "llm_stagewise_result.json"
    result_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = ["# LLM Stage-wise Supervised Audit\n"]
    if "stage_assessments" in parsed:
        md_lines.append("| Stage | Accept execution | Accept downstream | Risk | Main issue |")
        md_lines.append("| --- | --- | --- | --- | --- |")
        for r in parsed["stage_assessments"]:
            md_lines.append(
                f"| {r.get('stage')} | {r.get('accept_backend_execution')} | "
                f"{r.get('accept_for_downstream')} | {r.get('risk_level')} | "
                f"{str(r.get('main_issue', '')).replace('|', '/')} |"
            )
        md_lines.append("")
        md_lines.append("## Overall assessment\n")
        md_lines.append("```json")
        md_lines.append(json.dumps(parsed.get("overall_assessment", {}), indent=2, ensure_ascii=False))
        md_lines.append("```")
    else:
        md_lines.append("Failed to parse stage-wise JSON. See raw output.")
        md_lines.append("```text")
        md_lines.append(text)
        md_lines.append("```")

    md_path = out_dir / "llm_stagewise_result.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "prompt": str(prompt_path),
        "raw": str(raw_path),
        "json": str(result_path),
        "md": str(md_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
