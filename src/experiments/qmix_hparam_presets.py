from __future__ import annotations

from typing import Tuple


DEFAULT_QMIX_CONFIG = "qmix"
LBF_PAPER_TUNED_QMIX_CONFIG = "qmix_lbf_paper_tuned"
SUPPORTED_QMIX_HPARAM_PRESETS = ("default", "lbf_paper_tuned")


def list_qmix_hparam_presets() -> tuple[str, ...]:
    return SUPPORTED_QMIX_HPARAM_PRESETS


def resolve_qmix_hparam_preset(
    train_config: str | None,
    qmix_hparam_preset: str | None,
) -> Tuple[str, str]:
    config_name = str(train_config or DEFAULT_QMIX_CONFIG).strip()
    preset_name = str(qmix_hparam_preset or "").strip() or "default"
    if preset_name not in SUPPORTED_QMIX_HPARAM_PRESETS:
        available = ", ".join(SUPPORTED_QMIX_HPARAM_PRESETS)
        raise ValueError(
            f"unknown qmix_hparam_preset: {qmix_hparam_preset}. available: {available}"
        )

    if config_name == LBF_PAPER_TUNED_QMIX_CONFIG:
        return (LBF_PAPER_TUNED_QMIX_CONFIG, "lbf_paper_tuned")
    if config_name != DEFAULT_QMIX_CONFIG:
        if preset_name != "default":
            raise ValueError(
                "qmix_hparam_preset is only supported when train_config is qmix or "
                "qmix_lbf_paper_tuned"
            )
        return (config_name, "default")

    if preset_name == "lbf_paper_tuned":
        return (LBF_PAPER_TUNED_QMIX_CONFIG, "lbf_paper_tuned")
    return (DEFAULT_QMIX_CONFIG, "default")
