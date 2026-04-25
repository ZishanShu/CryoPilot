"""Contract and metadata for the cryodrgn_ai reconstruction backend."""

from __future__ import annotations

from typing import Any, Dict

BACKEND_NAME = "cryodrgn_ai"

CRYODRGN_AI_INPUT_CONTRACT: Dict[str, Any] = {
    "backend_name": BACKEND_NAME,
    "category": "reconstruction",
    "subcategory": "neural_abinitio_heterogeneity",
    "required_inputs": ("particles_star_or_mrcs", "ctf_metadata"),
    "optional_inputs": (
        "datadir",
        "apix",
        "box_size",
        "downsample_box",
        "zdim",
        "gpu_id",
        "num_epochs",
        "max_particles",
    ),
}

CRYODRGN_AI_BACKEND_METADATA: Dict[str, Any] = {
    "backend_name": BACKEND_NAME,
    "category": "reconstruction",
    "subcategory": "neural_abinitio_heterogeneity",
    "capability_summary": "cryoDRGN-AI ab-initio neural reconstruction and latent heterogeneity via cryodrgn abinit",
    "execution_modes": ["dry_run", "real"],
    "required_inputs": list(CRYODRGN_AI_INPUT_CONTRACT["required_inputs"]),
    "optional_inputs": list(CRYODRGN_AI_INPUT_CONTRACT["optional_inputs"]),
    "primary_outputs": [
        "run_dir",
        "command_manifest",
        "stdout_log",
        "stderr_log",
        "latent_embeddings",
        "model_weights",
        "sampled_volumes",
    ],
    "fallback_candidates": ["relion_initial_model", "relion_3d_classification", "human_review"],
    "required_dependencies": ["cryodrgn_executable", "cryodrgn abinit"],
    "optional_dependencies": ["gpu"],
    "supports_gpu": True,
    "supports_particle_input": True,
    "supports_consensus_pose": False,
    "notes": "Optional branch under the RELION-centered workflow; does not replace RELION refinement.",
}


def get_contract() -> Dict[str, Any]:
    """Return a defensive copy of the backend contract."""

    return {
        **CRYODRGN_AI_INPUT_CONTRACT,
        "required_inputs": list(CRYODRGN_AI_INPUT_CONTRACT["required_inputs"]),
        "optional_inputs": list(CRYODRGN_AI_INPUT_CONTRACT["optional_inputs"]),
    }


def get_metadata() -> Dict[str, Any]:
    """Return a defensive copy of the backend metadata."""

    out = dict(CRYODRGN_AI_BACKEND_METADATA)
    for key in ("execution_modes", "required_inputs", "optional_inputs", "primary_outputs", "fallback_candidates"):
        out[key] = list(out.get(key, []))
    return out
