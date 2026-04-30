from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Defensive parser. The prompt asks for strict JSON, but this also handles
    accidental markdown fences.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"Could not parse JSON from model output:\n{text[:2000]}")


def build_prompt(rule_result: Dict[str, Any]) -> str:
    metrics = rule_result.get("metrics", {})
    input_state = rule_result.get("input_state", {})
    decision_trace = rule_result.get("decision_trace", [])

    payload = {
        "task": "LLM-supervised validation for a CryoAgent particle-picking decision",
        "important_constraints": [
            "You are a supervisor, not the executor.",
            "Do not claim that you ran Topaz, RELION, crYOLO, or any numerical cryo-EM backend.",
            "Evaluate whether the rule-based decision and the actual backend output should be accepted for downstream reconstruction.",
            "Separate runtime success from scientific acceptability.",
            "Return strict JSON only. Do not wrap it in markdown.",
        ],
        "dataset": rule_result.get("dataset", "unknown_dataset"),
        "stage": rule_result.get("stage", "particle_picking"),
        "rule_based_decision": {
            "selected_backend": rule_result.get("selected_backend"),
            "final_status": rule_result.get("final_status"),
            "decision_trace": decision_trace,
        },
        "input_state": input_state,
        "real_execution_metrics": metrics,
        "expected_output_schema": {
            "protocol": "llm_supervised_agent",
            "dataset": "unknown_dataset",
            "stage": "particle_picking",
            "llm_role": "decision_supervisor_not_executor",
            "rule_backend_selection_assessment": {
                "accept_rule_backend_selection": "boolean",
                "reason": "string",
            },
            "scientific_output_assessment": {
                "accept_backend_output_for_downstream": "boolean",
                "risk_level": "low|medium|high",
                "main_issue": "string",
                "qc_interpretation": "string",
            },
            "recommended_next_action": {
                "action": "string",
                "reason": "string",
                "fallback_candidates": ["string"],
            },
            "audit_notes": ["string"],
            "final_status": "string",
        },
    }

    return json.dumps(payload, indent=2)


def call_openai(prompt: str, model: str) -> Dict[str, Any]:
    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a cryo-EM workflow supervisor for CryoAgent. "
                    "You validate rule-based workflow decisions using structured evidence. "
                    "You must be conservative, factual, and auditable. "
                    "You must return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    text = getattr(response, "output_text", None)
    if not text:
        # Fallback for SDK variants.
        text_parts = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    text_parts.append(value)
        text = "\n".join(text_parts)

    if not text:
        raise RuntimeError("OpenAI response did not contain output_text.")

    return _extract_json_object(text)


def write_markdown(result: Dict[str, Any], out_md: Path) -> None:
    rb = result.get("rule_backend_selection_assessment", {})
    sci = result.get("scientific_output_assessment", {})
    rec = result.get("recommended_next_action", {})

    lines = []
    lines.append("# Real API LLM-supervised Agent Result: unknown_dataset\n\n")
    lines.append("## Role\n\n")
    lines.append("The LLM-supervised branch validates the rule-based decision. It does not execute numerical cryo-EM computation.\n\n")

    lines.append("## Rule backend selection assessment\n\n")
    lines.append(f"- Accept rule backend selection: {rb.get('accept_rule_backend_selection')}\n")
    lines.append(f"- Reason: {rb.get('reason')}\n\n")

    lines.append("## Scientific output assessment\n\n")
    lines.append(f"- Accept backend output for downstream: {sci.get('accept_backend_output_for_downstream')}\n")
    lines.append(f"- Risk level: {sci.get('risk_level')}\n")
    lines.append(f"- Main issue: {sci.get('main_issue')}\n")
    lines.append(f"- QC interpretation: {sci.get('qc_interpretation')}\n\n")

    lines.append("## Recommended next action\n\n")
    lines.append(f"- Action: {rec.get('action')}\n")
    lines.append(f"- Reason: {rec.get('reason')}\n")
    fallbacks = rec.get("fallback_candidates") or []
    if fallbacks:
        lines.append(f"- Fallback candidates: {', '.join(map(str, fallbacks))}\n")

    lines.append("\n## Audit notes\n\n")
    for note in result.get("audit_notes", []):
        lines.append(f"- {note}\n")

    lines.append(f"\n## Final status\n\n- {result.get('final_status')}\n")

    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", required=True, type=Path)
    parser.add_argument(
        "--model",
        default=os.environ.get("CRYOAGENT_LLM_MODEL", "gpt-5.5"),
    )
    args = parser.parse_args()

    exp_root: Path = args.exp_root
    rule_json = exp_root / "rule_based" / "rule_based_result.json"
    out_dir = exp_root / "llm_supervised"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running this script.")

    if not rule_json.is_file():
        raise SystemExit(f"Missing rule result JSON: {rule_json}")

    rule_result = _read_json(rule_json)
    prompt = build_prompt(rule_result)

    prompt_path = out_dir / "llm_supervised_prompt.json"
    prompt_path.write_text(prompt, encoding="utf-8")

    result = call_openai(prompt, args.model)

    # Attach provenance without changing the model judgement.
    result["protocol"] = "llm_supervised_agent"
    result.setdefault("dataset", "unknown_dataset")
    result["stage"] = rule_result.get("stage", "particle_picking")
    result["api_call_provenance"] = {
        "provider": "openai",
        "model": args.model,
        "input_rule_result": str(rule_json),
        "prompt_file": str(prompt_path),
    }

    out_json = out_dir / "llm_supervised_result.json"
    out_md = out_dir / "llm_supervised_result.md"

    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out_md)

    print(json.dumps({
        "status": "ok",
        "model": args.model,
        "out_json": str(out_json),
        "out_md": str(out_md),
        "prompt": str(prompt_path),
    }, indent=2))


if __name__ == "__main__":
    main()
