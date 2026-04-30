from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_cmd(cmd: List[str], log_path: Path, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()

        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    runtime = round(time.time() - t0, 4)
    text = log_path.read_text(errors="ignore")

    return {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else None,
        "return_code": p.returncode,
        "runtime_sec": runtime,
        "log": str(log_path),
        "has_error_pattern": bool(re.search(r"ERROR|Traceback|Cannot|failed|not found|Exception|segmentation", text, re.I)),
    }


def read_pairs(pairs_csv: Path) -> List[Dict[str, str]]:
    with pairs_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_topaz_coord(coord_path: Path, max_particles: int) -> List[tuple[float, float]]:
    pts: List[tuple[float, float]] = []
    if not coord_path.is_file():
        return pts

    with coord_path.open(errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"[\s,]+", s)
            low = [p.lower() for p in parts]
            if "x_coord" in low and "y_coord" in low:
                continue

            # Topaz coord: image_name x_coord y_coord score
            if len(parts) >= 3:
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                except Exception:
                    continue
                pts.append((x, y))

            if len(pts) >= max_particles:
                break

    return pts


def read_mrc_apix(mrc_path: Path) -> float:
    try:
        import mrcfile  # type: ignore
        with mrcfile.open(mrc_path, permissive=True) as m:
            vx = float(m.voxel_size.x)
            if vx > 0:
                return vx
    except Exception:
        pass
    return 1.0


def write_micrographs_star(project: Path, micrograph_rel_paths: List[str], apix: float, voltage: float, cs: float, amp: float) -> Path:
    star = project / "micrographs.star"

    rows = "\n".join(f"{p} 1" for p in micrograph_rel_paths)

    star.write_text(
        f"""# version 50001

data_optics

loop_
_rlnOpticsGroupName #1
_rlnOpticsGroup #2
_rlnMicrographPixelSize #3
_rlnVoltage #4
_rlnSphericalAberration #5
_rlnAmplitudeContrast #6
opticsGroup1 1 {apix:.6f} {voltage:.1f} {cs:.3f} {amp:.3f}

data_micrographs

loop_
_rlnMicrographName #1
_rlnOpticsGroup #2
{rows}
""",
        encoding="utf-8",
    )
    return star


def write_box(box_path: Path, coords: List[tuple[float, float]], box_size: int) -> None:
    box_path.parent.mkdir(parents=True, exist_ok=True)
    half = box_size / 2.0
    with box_path.open("w", encoding="utf-8") as f:
        for x, y in coords:
            f.write(f"{x - half:.1f}\t{y - half:.1f}\t{box_size}\t{box_size}\n")


def count_particle_rows(particles_star: Path) -> int:
    if not particles_star.is_file():
        return 0
    n = 0
    for line in particles_star.read_text(errors="ignore").splitlines():
        if ".mrcs" in line or ".mrc" in line:
            n += 1
    return n


def write_rule_decision(path: Path, decision: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")


def safe_openai_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    key = key.replace("“", "").replace("”", "")
    m = re.search(r"(sk-[A-Za-z0-9_\-]+)", key)
    if m:
        key = m.group(1)
    try:
        key.encode("ascii")
    except UnicodeEncodeError:
        return None
    if not key.startswith("sk-"):
        return None
    return key


def run_llm_supervisor(summary: Dict[str, Any], out_dir: Path, model: str) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = {
        "role": "LLM-supervised scientific workflow auditor",
        "task": (
            "Review this same-dataset cryo-EM agent workflow trace. "
            "Separate runtime success from scientific acceptance. "
            "Return a concise JSON-like assessment with accept_backend_execution, "
            "accept_for_downstream, risk_level, main_issue, and recommended_next_action."
        ),
        "workflow_summary": summary,
    }

    (out_dir / "llm_supervised_prompt.json").write_text(
        json.dumps(prompt, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    key = safe_openai_key()
    if not key:
        result = {
            "status": "skipped",
            "reason": "OPENAI_API_KEY missing or invalid after ASCII sanitization",
        }
        (out_dir / "llm_supervised_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out_dir / "llm_supervised_result.md").write_text(
            "# LLM-supervised Result\n\nSkipped: OPENAI_API_KEY missing or invalid.\n",
            encoding="utf-8",
        )
        return result

    os.environ["OPENAI_API_KEY"] = key

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You are a careful cryo-EM workflow auditor. Be conservative and evidence-based.",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False, indent=2),
                },
            ],
        )

        text = getattr(response, "output_text", "") or str(response)
        result = {
            "status": "ok",
            "model": model,
            "raw_text": text,
        }

    except Exception as e:
        result = {
            "status": "error",
            "model": model,
            "error": repr(e),
        }
        text = f"LLM call failed:\n\n{repr(e)}"

    (out_dir / "llm_supervised_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "llm_supervised_result.md").write_text(
        "# LLM-supervised Result\n\n" + text + "\n",
        encoding="utf-8",
    )

    return result


def maybe_run_map2pdb(args: argparse.Namespace, project: Path, recon_map: Path, out_root: Path, env: Dict[str, str]) -> Dict[str, Any]:
    map2pdb_dir = out_root / "map2pdb"
    map2pdb_dir.mkdir(parents=True, exist_ok=True)

    state_dir = project / ".cryoem_agent"
    state_dir.mkdir(parents=True, exist_ok=True)

    sample_state = {
        "project_id": args.project_id,
        "project_path": str(project),
        "stage": "PostProcess",
        "status": "running",
        "map_path": str(recon_map),
        "sequence_path": str(args.sequence) if args.sequence else "",
        "extra": {
            "same_dataset_chain": {
                "dataset_id": args.dataset_id,
                "particle_picking_backend": "topaz",
                "particle_extraction_backend": "relion_preprocess",
                "reconstruction_backend": "relion_reconstruct",
                "evidence_root": str(out_root),
            }
        },
    }

    (state_dir / "sample_state.json").write_text(
        json.dumps(sample_state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.map2pdb_mode == "skip":
        return {"status": "skipped", "reason": "map2pdb_mode=skip"}

    cmd = [sys.executable, "-m", "app.main", "run-next", "--project", str(project)]
    if args.sequence:
        cmd += ["--sequence", str(args.sequence)]
    if args.map2pdb_mode == "dry-run":
        cmd.append("--dry-run")

    result = run_cmd(
        cmd,
        map2pdb_dir / f"run_next_map2pdb_{args.map2pdb_mode}.log",
        cwd=Path.cwd(),
        env=env,
    )

    result["sample_state"] = str(state_dir / "sample_state.json")
    result["action_history"] = str(state_dir / "action_history.jsonl")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", default="EMPIAR-10017")
    ap.add_argument("--project-id", default="empiar10017_same_dataset_agent_case")
    ap.add_argument("--pairs-csv", required=True, type=Path)
    ap.add_argument("--topaz-pred-dir", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--max-particles-per-micrograph", type=int, default=200)
    ap.add_argument("--box-size", type=int, default=256)
    ap.add_argument("--bg-radius", type=int, default=80)
    ap.add_argument("--voltage", type=float, default=300.0)
    ap.add_argument("--cs", type=float, default=2.7)
    ap.add_argument("--amp-contrast", type=float, default=0.1)
    ap.add_argument("--relion-bin", type=Path, default=Path.home() / "conda_envs/relion/bin")
    ap.add_argument("--sequence", type=Path, default=None)
    ap.add_argument("--map2pdb-mode", choices=["skip", "dry-run", "real"], default="dry-run")
    ap.add_argument("--run-llm", action="store_true")
    ap.add_argument("--llm-model", default=os.environ.get("CRYOAGENT_LLM_MODEL", "gpt-5.5"))
    args = ap.parse_args()

    out_root = args.out_root.resolve()
    project = out_root / "relion_project"
    logs = out_root / "logs"
    rule_dir = out_root / "rule_based"
    llm_dir = out_root / "llm_supervised"

    for d in [
        project / "Micrographs",
        project / "Coords/Micrographs",
        project / "Extract/job001_box/Particles",
        logs,
        rule_dir,
        llm_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = str(args.relion_bin) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(args.relion_bin.parent / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    pairs = read_pairs(args.pairs_csv)
    selected: List[Dict[str, Any]] = []

    for r in pairs:
        mid = r.get("micrograph_id") or Path(r.get("local_mrc_path", "")).stem
        mrc = Path(r.get("local_mrc_path") or r.get("mrc_path") or "")
        pred = args.topaz_pred_dir / f"{mid}.coord"

        if not mrc.is_file() or not pred.is_file():
            continue

        coords = parse_topaz_coord(pred, args.max_particles_per_micrograph)
        if not coords:
            continue

        selected.append({
            "micrograph_id": mid,
            "mrc": mrc,
            "pred": pred,
            "n_coords": len(coords),
            "coords": coords,
        })

        if len(selected) >= args.limit:
            break

    if not selected:
        raise SystemExit("No usable micrographs found with both MRC and Topaz coord predictions.")

    micrograph_rel_paths: List[str] = []
    apix_values: List[float] = []

    for item in selected:
        mid = item["micrograph_id"]
        src_mrc = Path(item["mrc"])
        dst_mrc = project / "Micrographs" / f"{mid}.mrc"

        if dst_mrc.exists() or dst_mrc.is_symlink():
            dst_mrc.unlink()
        dst_mrc.symlink_to(src_mrc)

        micrograph_rel = dst_mrc.relative_to(project).as_posix()
        micrograph_rel_paths.append(micrograph_rel)

        # RELION with --coord_dir Coords/ and micrograph path Micrographs/X.mrc
        # expects Coords/Micrographs/X.box
        box_path = project / "Coords" / "Micrographs" / f"{mid}.box"
        write_box(box_path, item["coords"], args.box_size)

        apix_values.append(read_mrc_apix(src_mrc))

    apix = apix_values[0] if apix_values else 1.0

    micrographs_star = write_micrographs_star(
        project=project,
        micrograph_rel_paths=micrograph_rel_paths,
        apix=apix,
        voltage=args.voltage,
        cs=args.cs,
        amp=args.amp_contrast,
    )

    rule_jsonl = rule_dir / "stage_decisions.jsonl"
    if rule_jsonl.exists():
        rule_jsonl.unlink()

    picking_decision = {
        "stage": "particle_picking",
        "agent": "rule_based",
        "selected_backend": "topaz",
        "status": "accepted_for_extraction" if selected else "failed",
        "n_micrographs": len(selected),
        "total_coords_used": sum(int(x["n_coords"]) for x in selected),
        "reason": "Topaz coordinate files exist and contain candidate particles; pass to RELION extraction smoke.",
    }
    write_rule_decision(rule_jsonl, picking_decision)

    extract_cmd = [
        str(args.relion_bin / "relion_preprocess"),
        "--i", "micrographs.star",
        "--coord_dir", "Coords/",
        "--coord_suffix", ".box",
        "--part_dir", "Extract/job001_box/Particles/",
        "--part_star", "Extract/job001_box/particles.star",
        "--extract",
        "--extract_size", str(args.box_size),
        "--bg_radius", str(args.bg_radius),
        "--norm",
        "--white_dust", "-1",
        "--black_dust", "-1",
    ]

    extract_result = run_cmd(
        extract_cmd,
        logs / "relion_preprocess_extract.log",
        cwd=project,
        env=env,
    )

    particles_star = project / "Extract/job001_box/particles.star"
    n_particles = count_particle_rows(particles_star)
    stacks = sorted((project / "Extract/job001_box").rglob("*.mrcs"))

    extraction_decision = {
        "stage": "particle_extraction",
        "agent": "rule_based",
        "selected_backend": "relion_preprocess",
        "status": "accepted_for_reconstruction" if n_particles > 0 and stacks else "failed_or_empty",
        "n_particles": n_particles,
        "n_particle_stacks": len(stacks),
        "particles_star": str(particles_star),
        "return_code": extract_result["return_code"],
        "runtime_sec": extract_result["runtime_sec"],
        "reason": "Extraction accepted if particles.star contains particle image rows and particle stack files exist.",
    }
    write_rule_decision(rule_jsonl, extraction_decision)

    recon_dir = out_root / "reconstruct"
    recon_dir.mkdir(parents=True, exist_ok=True)
    recon_map = recon_dir / "reconstruct_from_topaz_extract_box_smoke.mrc"

    if n_particles > 0:
        recon_cmd = [
            str(args.relion_bin / "relion_reconstruct"),
            "--i", "Extract/job001_box/particles.star",
            "--o", str(recon_map),
        ]
        recon_result = run_cmd(
            recon_cmd,
            logs / "relion_reconstruct.log",
            cwd=project,
            env=env,
        )
    else:
        recon_result = {
            "return_code": None,
            "runtime_sec": 0,
            "log": "",
            "skipped": True,
            "reason": "No extracted particles.",
        }

    reconstruction_decision = {
        "stage": "reconstruction",
        "agent": "rule_based",
        "selected_backend": "relion_reconstruct",
        "status": "accepted_for_map2pdb" if recon_map.is_file() and recon_map.stat().st_size > 0 else "failed",
        "map_path": str(recon_map),
        "map_size_bytes": recon_map.stat().st_size if recon_map.is_file() else 0,
        "return_code": recon_result.get("return_code"),
        "runtime_sec": recon_result.get("runtime_sec"),
        "reason": "Reconstruction accepted if RELION writes a non-empty MRC map.",
    }
    write_rule_decision(rule_jsonl, reconstruction_decision)

    map2pdb_result = maybe_run_map2pdb(args, project, recon_map, out_root, env)

    map2pdb_decision = {
        "stage": "map2pdb",
        "agent": "rule_based",
        "mode": args.map2pdb_mode,
        "status": "recorded",
        "result": map2pdb_result,
        "reason": "Map2PDB is invoked through CryoAgent run-next if requested; dry-run is allowed for audit trail.",
    }
    write_rule_decision(rule_jsonl, map2pdb_decision)

    summary = {
        "dataset_id": args.dataset_id,
        "project_id": args.project_id,
        "workflow": "same_dataset_particle_picking_to_map2pdb_agent_case",
        "execution_mode": "real_backends_for_picking_extraction_reconstruction",
        "out_root": str(out_root),
        "relion_project": str(project),
        "selected_micrographs": [
            {
                "micrograph_id": x["micrograph_id"],
                "mrc": str(x["mrc"]),
                "topaz_coord": str(x["pred"]),
                "n_coords_used": int(x["n_coords"]),
            }
            for x in selected
        ],
        "artifacts": {
            "micrographs_star": str(micrographs_star),
            "particles_star": str(particles_star),
            "particle_stacks": [str(p) for p in stacks],
            "reconstruction_map": str(recon_map),
            "rule_based_decisions": str(rule_jsonl),
            "map2pdb_sample_state": str(project / ".cryoem_agent/sample_state.json"),
            "map2pdb_action_history": str(project / ".cryoem_agent/action_history.jsonl"),
        },
        "metrics": {
            "n_micrographs": len(selected),
            "total_topaz_coords_used": sum(int(x["n_coords"]) for x in selected),
            "n_extracted_particles": n_particles,
            "n_particle_stacks": len(stacks),
            "reconstruction_map_exists": recon_map.is_file(),
            "reconstruction_map_size_bytes": recon_map.stat().st_size if recon_map.is_file() else 0,
        },
        "stage_status": {
            "picking": picking_decision["status"],
            "extraction": extraction_decision["status"],
            "reconstruction": reconstruction_decision["status"],
            "map2pdb": map2pdb_decision["status"],
        },
        "logs": {
            "extract": str(logs / "relion_preprocess_extract.log"),
            "reconstruct": str(logs / "relion_reconstruct.log"),
            "map2pdb": str((out_root / "map2pdb" / f"run_next_map2pdb_{args.map2pdb_mode}.log")),
        },
    }

    (out_root / "workflow_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rule_md = f"""# Rule-based Same-Dataset Agent Workflow Result

## Status

- Dataset: {args.dataset_id}
- Micrographs: {summary["metrics"]["n_micrographs"]}
- Topaz coordinates used: {summary["metrics"]["total_topaz_coords_used"]}
- Extracted particles: {summary["metrics"]["n_extracted_particles"]}
- Particle stack count: {summary["metrics"]["n_particle_stacks"]}
- Reconstruction map exists: {summary["metrics"]["reconstruction_map_exists"]}
- Map2PDB mode: {args.map2pdb_mode}

## Artifacts

- RELION project: `{project}`
- Micrographs STAR: `{micrographs_star}`
- Particles STAR: `{particles_star}`
- Reconstruction map: `{recon_map}`
- Rule decisions JSONL: `{rule_jsonl}`

## Interpretation

This run records a same-dataset agent workflow from Topaz particle picking coordinates to RELION particle extraction and RELION reconstruction. Map2PDB is invoked according to the selected mode.

This is a technical workflow smoke/batch case. Scientific quality still requires downstream QC.
"""
    (rule_dir / "rule_based_result.md").write_text(rule_md, encoding="utf-8")

    if args.run_llm:
        llm_result = run_llm_supervisor(summary, llm_dir, args.llm_model)
        summary["llm_supervised"] = llm_result
        (out_root / "workflow_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps({
        "status": "ok",
        "out_root": str(out_root),
        "summary": str(out_root / "workflow_summary.json"),
        "rule_result": str(rule_dir / "rule_based_result.md"),
        "llm_result": str(llm_dir / "llm_supervised_result.md") if args.run_llm else "",
        "reconstruction_map": str(recon_map),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
