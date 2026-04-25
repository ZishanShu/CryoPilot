"""Centralized backend resolution for picking, heterogeneity, and model-building domains."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from backends.reconstruction.cryodrgn_ai_contract import (
    CRYODRGN_AI_BACKEND_METADATA,
    CRYODRGN_AI_INPUT_CONTRACT,
)

logger = logging.getLogger(__name__)

# Unified input contract per model_building backend (single source for preflight + registry).
MODEL_BUILDING_INPUT_SPECS: Dict[str, Dict[str, Any]] = {
    "modelangelo": {
        "backend_name": "modelangelo",
        "required_fields": ("map_path", "sequence_path"),
        "optional_fields": (
            "output_dir",
            "run_id",
            "chain_count",
            "device",
            "gpu",
            "use_gpu",
            "weights_path",
            "model_dir",
        ),
        "defaults": {},
    },
    "emodelx": {
        "backend_name": "emodelx",
        "required_fields": ("map_path", "sequence_path", "output_dir", "emodelx_repo_dir"),
        "optional_fields": (
            "run_id",
            "chain_count",
            "emodelx_python",
            "emodelx_protocol",
            "emodelx_weights_dir",
            "run_pulchra",
            "pulchra_path",
            "run_phenix",
            "phenix_act",
            "resolution",
            "template_dir",
            "download_afdb",
            "afdb_allow_seq_id",
        ),
        "defaults": {"chain_count": 1, "emodelx_protocol": "temp_free"},
    },
    "mica": {
        "backend_name": "mica",
        "required_fields": ("map_path", "sequence_path", "output_dir", "mica_repo_dir", "af3_dir"),
        "optional_fields": (
            "af3_result_dir",
            "run_id",
            "chain_count",
            "mica_python",
            "mica_weights_dir",
            "mica_config_path",
            "mica_device",
            "mica_resolution",
            "run_pulchra",
            "pulchra_path",
            "run_phenix",
            "phenix_act",
        ),
        "defaults": {"mica_device": "cuda", "protocol": "AF3_struct"},
    },
}

# Unified metadata for model_building domain backends (policy / reports / tests).
MODEL_BUILDING_BACKEND_METADATA: Dict[str, Dict[str, Any]] = {
    "modelangelo": {
        "backend_name": "modelangelo",
        "category": "model_building",
        "capability_summary": "Sequence-guided map-to-model with ModelAngelo",
        "availability": "available",
        "experimental": False,
        "placeholder_execution": False,
    },
    "emodelx": {
        "backend_name": "emodelx",
        "category": "model_building",
        "capability_summary": "Repo-oriented EModelX map-to-model via python run.py",
        "availability": "available",
        "experimental": False,
        "placeholder_execution": False,
    },
    "mica": {
        "backend_name": "mica",
        "category": "model_building",
        "capability_summary": "Repo-oriented MICA AF3-aware map-to-model via python run.py",
        "availability": "available",
        "experimental": False,
        "placeholder_execution": False,
    },
}

for _bid, _meta in MODEL_BUILDING_BACKEND_METADATA.items():
    _sp = MODEL_BUILDING_INPUT_SPECS.get(_bid, {})
    _meta["required_fields"] = list(_sp.get("required_fields", ()))
    _meta["optional_fields"] = list(_sp.get("optional_fields", ()))

PICKING_INPUT_SPECS: Dict[str, Dict[str, Any]] = {
    "topaz": {
        "backend_name": "topaz",
        "required_fields": ("micrographs_star_path", "particle_diameter_angstrom", "output_dir"),
        "optional_fields": (
            "topaz_executable",
            "topaz_env_activate",
            "topaz_model_path",
            "topaz_threshold",
            "topaz_num_particles_per_micrograph",
            "topaz_device",
            "topaz_train_mode",
            "use_pretrained_model",
            "picked_particles_star",
            "training_particles_star",
            "ctf_micrographs_star",
            "input_micrographs_star",
            "run_id",
            "micrograph_pixel_size_angstrom",
            "topaz_output_coords",
        ),
        "defaults": {},
    },
}

PICKING_BACKEND_METADATA: Dict[str, Dict[str, Any]] = {
    "topaz": {
        "backend_name": "topaz",
        "category": "picking",
        "capability_summary": "Topaz neural particle picking via external topaz CLI",
        "real_execution": True,
    },
}

for _pbid, _pmeta in PICKING_BACKEND_METADATA.items():
    _psp = PICKING_INPUT_SPECS.get(_pbid, {})
    _pmeta["required_fields"] = list(_psp.get("required_fields", ()))
    _pmeta["optional_fields"] = list(_psp.get("optional_fields", ()))

HETEROGENEITY_INPUT_SPECS: Dict[str, Dict[str, Any]] = {
    "cryodrgn": {
        "backend_name": "cryodrgn",
        # output_dir always; training inputs are either RELION particles.star or external mrcs+pose+ctf (preflight enforces one).
        "required_fields": ("output_dir",),
        "optional_fields": (
            "particles_star_path",
            "pose_pkl_path",
            "ctf_pkl_path",
            "particles_mrcs_path",
            "box_size",
            "apix",
            "downsample_box_size",
            "cryodrgn_executable",
            "cryodrgn_device",
            "cryodrgn_zdim",
            "cryodrgn_epochs",
            "cryodrgn_multigpu",
            "cryodrgn_analyze_after_train",
            "datadir",
            "run_id",
            "cryodrgn_train_subdir",
        ),
        "defaults": {},
    },
}

HETEROGENEITY_BACKEND_METADATA: Dict[str, Dict[str, Any]] = {
    "cryodrgn": {
        "backend_name": "cryodrgn",
        "category": "heterogeneity",
        "capability_summary": "cryoDRGN VAE latent heterogeneity (external cryodrgn CLI)",
        "real_execution": True,
    },
}

for _hbid, _hmeta in HETEROGENEITY_BACKEND_METADATA.items():
    _hsp = HETEROGENEITY_INPUT_SPECS.get(_hbid, {})
    _hmeta["required_fields"] = list(_hsp.get("required_fields", ()))
    _hmeta["optional_fields"] = list(_hsp.get("optional_fields", ()))

RECONSTRUCTION_INPUT_SPECS: Dict[str, Dict[str, Any]] = {
    "relion": {
        "backend_name": "relion",
        "required_fields": (),
        "optional_fields": ("particles_star_path", "initial_model_path", "map_path", "output_dir"),
        "defaults": {},
    },
    "cryodrgn_ai": {
        "backend_name": "cryodrgn_ai",
        "required_fields": tuple(CRYODRGN_AI_INPUT_CONTRACT["required_inputs"]),
        "optional_fields": tuple(CRYODRGN_AI_INPUT_CONTRACT["optional_inputs"]),
        "defaults": {},
    },
}

RECONSTRUCTION_BACKEND_METADATA: Dict[str, Dict[str, Any]] = {
    "relion": {
        "backend_name": "relion",
        "category": "reconstruction",
        "subcategory": "relion_initial_model_refinement",
        "capability_summary": "RELION-native initial model, 3D classification, and refinement path",
        "execution_modes": ["dry_run", "real"],
    },
    "cryodrgn_ai": dict(CRYODRGN_AI_BACKEND_METADATA),
}

for _rbid, _rmeta in RECONSTRUCTION_BACKEND_METADATA.items():
    _rsp = RECONSTRUCTION_INPUT_SPECS.get(_rbid, {})
    _rmeta["required_fields"] = list(_rsp.get("required_fields", ()))
    _rmeta["optional_fields"] = list(_rsp.get("optional_fields", ()))

DEFAULT_DOMAINS: Dict[str, Dict[str, Any]] = {
    "picking": {
        "default_backend": "relion_log",
        "allowed_backends": ["relion_log", "topaz", "cryolo"],
        "project_overrides": {},
    },
    "heterogeneity": {
        "default_backend": "human_review",
        "allowed_backends": ["human_review", "cryodrgn", "cryodrgn_ai"],
        "project_overrides": {},
    },
    "reconstruction": {
        "default_backend": "relion",
        "allowed_backends": ["relion", "cryodrgn_ai"],
        "project_overrides": {},
    },
    "model_building": {
        "default_backend": "modelangelo",
        "allowed_backends": ["modelangelo", "emodelx", "mica"],
        "project_overrides": {},
    },
}

BACKEND_CAPABILITY_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "relion_log": {
        "category": "picking",
        "maturity_level": "legacy",
        "execution_mode_supported": ["log_only"],
        "required_inputs": [],
        "optional_inputs": [],
        "required_dependencies": ["relion"],
        "optional_dependencies": [],
        "primary_outputs": ["relion_autopick_logs"],
        "typical_use_case": "RELION-native autopick/log-only fallback",
        "upstream_assumptions": ["ctf_completed"],
        "fallback_candidates": [],
        "supports_gpu": False,
        "supports_af3_prior": False,
        "supports_sequence_input": False,
        "supports_map_input": False,
        "supports_particle_input": False,
        "supports_consensus_pose": False,
        "notes": "Legacy fallback when neural pickers unavailable.",
    },
    "topaz": {
        "maturity_level": "real",
        "execution_mode_supported": ["real"],
        "required_dependencies": ["topaz_executable"],
        "optional_dependencies": ["topaz_model_path"],
        "primary_outputs": ["picked_particles_star", "topaz_output_coords"],
        "typical_use_case": "First-line neural particle picking",
        "upstream_assumptions": ["ctf_completed", "micrographs_star_path"],
        "fallback_candidates": ["relion_log"],
        "supports_gpu": True,
        "supports_af3_prior": False,
        "supports_sequence_input": False,
        "supports_map_input": False,
        "supports_particle_input": False,
        "supports_consensus_pose": False,
        "notes": "Best when micrographs and diameter are available.",
    },
    "cryolo": {
        "category": "picking",
        "maturity_level": "placeholder",
        "execution_mode_supported": ["dry_run_placeholder"],
        "required_inputs": [],
        "optional_inputs": [],
        "required_dependencies": ["cryolo"],
        "optional_dependencies": [],
        "primary_outputs": ["cryolo_predictions"],
        "typical_use_case": "Alternative picker path",
        "upstream_assumptions": ["ctf_completed"],
        "fallback_candidates": ["relion_log"],
        "supports_gpu": True,
        "supports_af3_prior": False,
        "supports_sequence_input": False,
        "supports_map_input": False,
        "supports_particle_input": False,
        "supports_consensus_pose": False,
        "notes": "Not yet first-party integrated.",
    },
    "human_review": {
        "category": "heterogeneity",
        "maturity_level": "manual",
        "execution_mode_supported": ["manual_review"],
        "required_inputs": [],
        "optional_inputs": [],
        "required_dependencies": [],
        "optional_dependencies": [],
        "primary_outputs": ["manual_decision"],
        "typical_use_case": "Fallback for ambiguous heterogeneity decisions",
        "upstream_assumptions": ["heterogeneity_warning"],
        "fallback_candidates": [],
        "supports_gpu": False,
        "supports_af3_prior": False,
        "supports_sequence_input": False,
        "supports_map_input": False,
        "supports_particle_input": True,
        "supports_consensus_pose": False,
        "notes": "Human-in-the-loop gate.",
    },
    "cryodrgn": {
        "maturity_level": "real",
        "execution_mode_supported": ["real", "unavailable"],
        "required_dependencies": ["cryodrgn_executable"],
        "optional_dependencies": ["gpu"],
        "primary_outputs": ["pose_pkl", "ctf_pkl", "train_weights", "latent_analysis"],
        "typical_use_case": "Consensus-pose heterogeneity modeling",
        "upstream_assumptions": ["particles_star_path or external stack"],
        "fallback_candidates": ["human_review"],
        "supports_gpu": True,
        "supports_af3_prior": False,
        "supports_sequence_input": False,
        "supports_map_input": False,
        "supports_particle_input": True,
        "supports_consensus_pose": True,
        "notes": "Preferred when STAR/pose/ctf are available.",
    },
    "cryodrgn_ai": {
        "category": "reconstruction",
        "subcategory": "neural_abinitio_heterogeneity",
        "maturity_level": "real",
        "execution_mode_supported": ["dry_run", "real"],
        "required_inputs": ["particles_star_or_mrcs", "ctf_metadata"],
        "optional_inputs": [
            "datadir",
            "apix",
            "box_size",
            "downsample_box",
            "zdim",
            "gpu_id",
            "num_epochs",
            "max_particles",
        ],
        "required_dependencies": ["cryodrgn_executable", "cryodrgn abinit"],
        "optional_dependencies": ["gpu"],
        "primary_outputs": [
            "run_dir",
            "command_manifest",
            "stdout_log",
            "stderr_log",
            "latent_embeddings",
            "model_weights",
            "sampled_volumes",
        ],
        "typical_use_case": "Neural ab-initio reconstruction when RELION initial model/classification is unreliable",
        "upstream_assumptions": ["particles.star+datadir+apix+box_size or particles.mrcs+ctf.pkl"],
        "fallback_candidates": ["relion_initial_model", "relion_3d_classification", "human_review"],
        "supports_gpu": True,
        "supports_af3_prior": False,
        "supports_sequence_input": False,
        "supports_map_input": False,
        "supports_particle_input": True,
        "supports_consensus_pose": False,
        "notes": "Optional reconstruction / heterogeneity branch; RELION remains the default path.",
    },
    "modelangelo": {
        "maturity_level": "real",
        "execution_mode_supported": ["real"],
        "required_dependencies": ["model_angelo_executable"],
        "optional_dependencies": ["gpu"],
        "primary_outputs": ["atomic_model"],
        "typical_use_case": "General map+sequence structure build",
        "upstream_assumptions": ["map_path", "sequence_path"],
        "fallback_candidates": [],
        "supports_gpu": True,
        "supports_af3_prior": False,
        "supports_sequence_input": True,
        "supports_map_input": True,
        "supports_particle_input": False,
        "supports_consensus_pose": False,
        "notes": "Current baseline model-building backend.",
    },
    "emodelx": {
        "maturity_level": "real",
        "execution_mode_supported": ["real", "unavailable"],
        "required_dependencies": ["emodelx_repo_dir", "emodelx_weights", "emodelx_python"],
        "optional_dependencies": ["pulchra", "phenix"],
        "primary_outputs": ["emodelx_models"],
        "typical_use_case": "Repo-oriented map+sequence modeling",
        "upstream_assumptions": ["map_path", "sequence_path"],
        "fallback_candidates": ["modelangelo"],
        "supports_gpu": True,
        "supports_af3_prior": False,
        "supports_sequence_input": True,
        "supports_map_input": True,
        "supports_particle_input": False,
        "supports_consensus_pose": False,
        "notes": "Temp_free first; temp_flex optional.",
    },
    "mica": {
        "maturity_level": "real",
        "execution_mode_supported": ["real", "unavailable"],
        "required_dependencies": ["mica_repo_dir", "mica_weights", "mica_python"],
        "optional_dependencies": ["pulchra", "phenix"],
        "primary_outputs": ["mica_models"],
        "typical_use_case": "AF3-aware map+sequence model building",
        "upstream_assumptions": ["map_path", "sequence_path", "af3_dir"],
        "fallback_candidates": ["emodelx", "modelangelo"],
        "supports_gpu": True,
        "supports_af3_prior": True,
        "supports_sequence_input": True,
        "supports_map_input": True,
        "supports_particle_input": False,
        "supports_consensus_pose": False,
        "notes": "Requires AF3 directory populated with structures/results.",
    },
}


class BackendRegistry:
    """Resolve a valid backend per domain using defaults, allow-lists, and project overrides."""

    def __init__(self, domains: Dict[str, Dict[str, Any]]) -> None:
        self._domains = domains

    @classmethod
    def from_policy_and_file(
        cls,
        policy_rules: Dict[str, Any],
        registry_yaml: Optional[Path] = None,
    ) -> "BackendRegistry":
        # Optional deployment defaults first; policy_rules merge last so policy always wins.
        merged = deepcopy(DEFAULT_DOMAINS)
        if registry_yaml and registry_yaml.exists():
            try:
                data = yaml.safe_load(registry_yaml.read_text(encoding="utf-8")) or {}
                for key in ("picking", "heterogeneity", "reconstruction", "model_building"):
                    if isinstance(data.get(key), dict):
                        merged[key] = {**merged.get(key, {}), **data[key]}
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("Could not load backend_registry.yaml: %s", exc)
        for key in ("picking", "heterogeneity", "reconstruction", "model_building"):
            sec = policy_rules.get(key) if isinstance(policy_rules, dict) else None
            if isinstance(sec, dict) and sec:
                merged[key] = {**merged.get(key, {}), **sec}
        if isinstance(policy_rules, dict):
            legacy = policy_rules.get("picker")
            if isinstance(legacy, dict) and legacy:
                merged["picking"] = {**merged.get("picking", {}), **legacy}
        return cls(merged)

    def section(self, domain: str) -> Dict[str, Any]:
        return dict(self._domains.get(domain, {}))

    @staticmethod
    def _canonical_id(domain: str, raw: Optional[str]) -> Optional[str]:
        """Return canonical backend id if ``raw`` names a known backend, else ``None``."""
        s = str(raw or "").strip().lower()
        if not s:
            return None
        if domain == "picking":
            return s if s in {"topaz", "cryolo", "relion_log"} else None
        if domain == "heterogeneity":
            return s if s in {"human_review", "cryodrgn", "cryodrgn_ai"} else None
        if domain == "reconstruction":
            alias = {
                "relion_initial_model": "relion",
                "relion_3d_classification": "relion",
                "run_cryodrgn_ai": "cryodrgn_ai",
            }
            if s in {"relion", "cryodrgn_ai"}:
                return s
            return alias.get(s)
        if domain == "model_building":
            if s in {"modelangelo", "emodelx", "mica"}:
                return s
            alias = {
                "run_modelangelo": "modelangelo",
                "run_emodelx": "emodelx",
                "run_mica": "mica",
            }
            return alias.get(s)
        return None

    def _effective_default(self, domain: str) -> str:
        """Domain ``default_backend`` from config, canonicalized; falls back to code defaults."""
        section = self._domains.get(domain, {})
        raw = str(section.get("default_backend", "")).strip()
        cid = self._canonical_id(domain, raw)
        if cid is not None:
            return cid
        fb = DEFAULT_DOMAINS.get(domain, {}).get("default_backend")
        return str(fb) if fb is not None else ""

    def _constrain(self, domain: str, normalized: str) -> Tuple[str, Optional[str]]:
        section = self._domains.get(domain, {})
        allowed = section.get("allowed_backends")
        if not isinstance(allowed, list) or not allowed:
            return normalized, None
        allowed_set = {str(x).strip().lower() for x in allowed}
        if normalized in allowed_set:
            return normalized, None
        default_b = str(section.get("default_backend", "")).strip().lower()
        if default_b in allowed_set:
            return default_b, "fallback_invalid_backend"
        return sorted(allowed_set)[0], "fallback_invalid_backend"

    def resolve(
        self,
        domain: str,
        preferred: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Return ``(backend, note)`` where ``note`` is set when falling back from a disallowed value.

        Precedence: explicit ``preferred`` (only if allowed) > project override > ``default_backend``.
        Unknown or disallowed explicit values fall back to the domain default.
        """
        section = self._domains.get(domain, {})
        allowed = section.get("allowed_backends")
        allowed_set = {str(x).strip().lower() for x in allowed} if isinstance(allowed, list) else set()
        has_allowlist = isinstance(allowed, list) and len(allowed) > 0

        if preferred is not None and str(preferred).strip():
            cid = self._canonical_id(domain, str(preferred).strip())
            if cid is None:
                d, _ = self._constrain(domain, self._effective_default(domain))
                return d, "fallback_invalid_backend"
            if not has_allowlist or cid in allowed_set:
                return cid, None
            d, _ = self._constrain(domain, self._effective_default(domain))
            return d, "fallback_invalid_backend"

        overrides = section.get("project_overrides") if isinstance(section, dict) else None
        if project_id and isinstance(overrides, dict) and project_id in overrides:
            cid = self._canonical_id(domain, str(overrides.get(project_id)))
            if cid is not None and (not has_allowlist or cid in allowed_set):
                return cid, None

        return self._constrain(domain, self._effective_default(domain))


def model_action_to_backend(action_name: str) -> str:
    """Map structure-modeling action name to model_building domain backend id."""
    return {
        "run_modelangelo": "modelangelo",
        "run_emodelx": "emodelx",
        "run_mica": "mica",
    }.get(action_name, "modelangelo")


def model_building_backend_to_action_name(backend_id: str) -> str:
    """Map canonical model_building backend id to ``ActionName`` string value."""
    return {
        "modelangelo": "run_modelangelo",
        "emodelx": "run_emodelx",
        "mica": "run_mica",
    }.get(str(backend_id).strip().lower(), "run_modelangelo")


def iter_model_building_backend_ids() -> Tuple[str, ...]:
    """Stable ordering for registry / tests."""
    return ("modelangelo", "emodelx", "mica")


def iter_reconstruction_backend_ids() -> Tuple[str, ...]:
    """Stable ordering for reconstruction registry / tests."""
    return ("relion", "cryodrgn_ai")


def get_backend_capability_matrix() -> Dict[str, Dict[str, Any]]:
    """Unified backend capability matrix for selection/report/benchmark."""
    matrix: Dict[str, Dict[str, Any]] = {}
    specs_by_name: Dict[str, Dict[str, Any]] = {}
    specs_by_name.update(PICKING_INPUT_SPECS)
    specs_by_name.update(HETEROGENEITY_INPUT_SPECS)
    specs_by_name.update(RECONSTRUCTION_INPUT_SPECS)
    specs_by_name.update(MODEL_BUILDING_INPUT_SPECS)
    metadata_by_name: Dict[str, Dict[str, Any]] = {}
    metadata_by_name.update(PICKING_BACKEND_METADATA)
    metadata_by_name.update(HETEROGENEITY_BACKEND_METADATA)
    metadata_by_name.update(RECONSTRUCTION_BACKEND_METADATA)
    metadata_by_name.update(MODEL_BUILDING_BACKEND_METADATA)
    for name, override in BACKEND_CAPABILITY_OVERRIDES.items():
        spec = specs_by_name.get(name, {})
        meta = metadata_by_name.get(name, {})
        row: Dict[str, Any] = {
            "backend_name": name,
            "category": override.get("category") or meta.get("category"),
            "maturity_level": override.get("maturity_level", "placeholder"),
            "execution_mode_supported": list(override.get("execution_mode_supported", [])),
            "required_inputs": list(spec.get("required_fields", override.get("required_inputs", []))),
            "optional_inputs": list(spec.get("optional_fields", override.get("optional_inputs", []))),
            "required_dependencies": list(override.get("required_dependencies", [])),
            "optional_dependencies": list(override.get("optional_dependencies", [])),
            "primary_outputs": list(override.get("primary_outputs", [])),
            "typical_use_case": str(override.get("typical_use_case", "")),
            "upstream_assumptions": list(override.get("upstream_assumptions", [])),
            "fallback_candidates": list(override.get("fallback_candidates", [])),
            "supports_gpu": bool(override.get("supports_gpu", False)),
            "supports_af3_prior": bool(override.get("supports_af3_prior", False)),
            "supports_sequence_input": bool(override.get("supports_sequence_input", False)),
            "supports_map_input": bool(override.get("supports_map_input", False)),
            "supports_particle_input": bool(override.get("supports_particle_input", False)),
            "supports_consensus_pose": bool(override.get("supports_consensus_pose", False)),
            "notes": str(override.get("notes", "")),
        }
        matrix[name] = row
    return matrix
