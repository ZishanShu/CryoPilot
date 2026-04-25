from agent.backend_registry import RECONSTRUCTION_BACKEND_METADATA, RECONSTRUCTION_INPUT_SPECS
from backends.reconstruction.cryodrgn_ai_contract import get_metadata


def test_cryodrgn_ai_contract_metadata() -> None:
    spec = RECONSTRUCTION_INPUT_SPECS["cryodrgn_ai"]
    assert spec["required_fields"] == ("particles_star_or_mrcs", "ctf_metadata")
    for field in ("datadir", "apix", "box_size", "downsample_box", "zdim", "gpu_id", "num_epochs", "max_particles"):
        assert field in spec["optional_fields"]

    meta = get_metadata()
    assert meta["category"] == "reconstruction"
    assert meta["subcategory"] == "neural_abinitio_heterogeneity"
    assert meta["execution_modes"] == ["dry_run", "real"]
    assert "command_manifest" in meta["primary_outputs"]
    assert meta["fallback_candidates"] == ["relion_initial_model", "relion_3d_classification", "human_review"]
    assert RECONSTRUCTION_BACKEND_METADATA["cryodrgn_ai"]["category"] == "reconstruction"
