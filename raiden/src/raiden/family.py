"""RAIDEN foundation family. Two SKUs. Flash is the only wired Stage I loader."""

from __future__ import annotations

FORBIDDEN_BASE_REPOS = (
    "dealignai/GLM-5.3-CYBERSECURITY-FP8",
    "dealignai/GLM-5.3-UNCENSORED-FP8",
    "dealignai/GLM-5.3-Flash-UNCENSORED-FP8",
    "dealignai/GLM-5.3-ABLITERATED-NVFP4",
    "JANGQ-AI/GLM-5.3-FP8",
)

FORBIDDEN_BASE_MARKERS = (
    "cybersecurity-fp8",
    "uncensored",
    "abliterat",
    "/crack",
    "glm-5.3-crack",
)

FORBIDDEN_BASE_REASON = (
    "RAIDEN Stage I is identity QLoRA on official Z.ai BF16 weights. "
    "Abliterated / CRACK / uncensored third-party dumps are not a RAIDEN base: "
    "they are native FP8 (QLoRA-incompatible), a different method (weight edit, not LoRA), "
    "and they are tuned to drop refusals for offensive-security content. "
    "Second SKU train weights: zai-org/GLM-5.3-BF16. "
    "H200 TP8 serving of FP8 753B is infra evidence, not a base-model swap."
)

TRACK_FLASH = "flash"
TRACK_GLM53 = "glm53"


def normalize_repo_id(model_id: str) -> str:
    return str(model_id).strip().replace("\\", "/").rstrip("/")


def forbidden_base_hit(model_id: str) -> str | None:
    raw = normalize_repo_id(model_id)
    lowered = raw.lower()
    for repo in FORBIDDEN_BASE_REPOS:
        if lowered == repo.lower() or lowered.endswith("/" + repo.split("/")[-1].lower()):
            return repo
    for marker in FORBIDDEN_BASE_MARKERS:
        if marker in lowered:
            return marker
    return None
