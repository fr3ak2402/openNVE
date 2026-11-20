"""Camera models supported by openNVE (all devices of neuromorphic_drivers)."""

from __future__ import annotations

import dataclasses
import types

import neuromorphic_drivers as nd


@dataclasses.dataclass(frozen=True)
class CameraModel:
    """A camera model, for instance SE1-S4.

    Attributes:
        id: Short model name used in JSON files, for instance "SE1-S4".
        name: Device name reported by neuromorphic_drivers, for instance "SENSING SE1-S4-USB".
        module: The neuromorphic_drivers module of the device, for instance nd.sensing_se1_s4.
    """

    id: str
    name: str
    module: types.ModuleType

    @property
    def configuration_type(self) -> type:
        return self.module.Configuration


# Short model names, the other devices use their neuromorphic_drivers module name
_SHORT_IDS = {
    "sensing_se1_s4": "SE1-S4",
    "prophesee_evk4": "EVK4",
    "prophesee_evk3_hd": "EVK3-HD",
    "prophesee_evk1_gen4": "EVK1-GEN4",
    "inivation_davis346": "DAVIS346",
    "inivation_dvxplorer": "DVXPLORER",
    "centuryarks_vga": "CENTURYARKS-VGA",
    "lucid_triton": "TRITON",
}


def _build_models() -> tuple[CameraModel, ...]:
    models = []
    for name in nd.Name:
        module_name = name.name.lower()
        module = getattr(nd, module_name, None)
        if module is None or not hasattr(module, "Configuration"):
            continue
        models.append(
            CameraModel(
                id=_SHORT_IDS.get(module_name, module_name.upper().replace("_", "-")),
                name=name.value,
                module=module,
            )
        )
    return tuple(models)


MODELS: tuple[CameraModel, ...] = _build_models()


def _normalize(value: str) -> str:
    return value.strip().upper().replace("_", "-").replace(" ", "-")


def get_model(model: str) -> CameraModel:
    """Find a model by short name ("SE1-S4"), device name ("SENSING SE1-S4-USB")
    or neuromorphic_drivers module name ("sensing_se1_s4"). Case-insensitive."""
    key = _normalize(model)
    for candidate in MODELS:
        if key in (
            _normalize(candidate.id),
            _normalize(candidate.name),
            _normalize(candidate.module.__name__.rsplit(".", 1)[-1]),
        ):
            return candidate
    raise ValueError(
        f"unknown camera model {model!r}, supported models: {', '.join(m.id for m in MODELS)}"
    )


def model_from_device_name(name: "nd.Name") -> CameraModel:
    """Model of a device listed by neuromorphic_drivers."""
    for candidate in MODELS:
        if candidate.name == name.value:
            return candidate
    raise ValueError(f"unsupported device {name.value!r}")
