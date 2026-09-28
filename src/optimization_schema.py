"""Shared schema for parameters exposed to the numerical optimizer.

The registry still owns the ODE topology and kinetic rates.  This module only
describes the case-level unknowns that an optimizer may vary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    default: float
    lower: float
    upper: float
    trainable: bool = True
    llm_allowed: bool = True
    description: str = ""

    def clamp(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = self.default
        if number != number or number in (float("inf"), float("-inf")):
            number = self.default
        return max(self.lower, min(self.upper, number))


OPTIMIZATION_PARAMETERS: Tuple[ParameterSpec, ...] = (
    ParameterSpec(
        "log10_intake", 4.0, 0.0, 12.0, description="log10 of acute intake in Bq",
    ),
    ParameterSpec(
        "intake_time_d", 0.5, -3650.0, 3650.0,
        description="time of intake relative to the declared observation origin",
    ),
)

PARAMETER_BLOCKS: Dict[str, Tuple[str, ...]] = {
    "BASIC": ("log10_intake", "intake_time_d"),
}

PARAMETER_BY_NAME = {item.name: item for item in OPTIMIZATION_PARAMETERS}
LLM_ALLOWED_PARAMETER_NAMES = tuple(
    item.name for item in OPTIMIZATION_PARAMETERS if item.llm_allowed
)


def parameter_defaults(overrides: Mapping[str, Any] | None = None) -> Dict[str, float]:
    values = {item.name: item.default for item in OPTIMIZATION_PARAMETERS}
    for name, value in (overrides or {}).items():
        if name in PARAMETER_BY_NAME:
            values[name] = PARAMETER_BY_NAME[name].clamp(value)
    return values


def parameter_bounds(
    overrides: Mapping[str, Any] | None = None,
    names: Sequence[str] | None = None,
) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {}
    requested = tuple(names or (item.name for item in OPTIMIZATION_PARAMETERS))
    overrides = overrides or {}
    for name in requested:
        spec = PARAMETER_BY_NAME.get(name)
        if spec is None:
            continue
        value = overrides.get(name)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            low = max(spec.lower, float(value[0]))
            high = min(spec.upper, float(value[1]))
            if low < high:
                result[name] = [low, high]
                continue
        result[name] = [spec.lower, spec.upper]
    return result


def validate_parameter_values(
    values: Mapping[str, Any],
    required: Iterable[str] | None = None,
) -> Dict[str, float]:
    allowed = set(LLM_ALLOWED_PARAMETER_NAMES)
    required_names = tuple(required or ("log10_intake", "intake_time_d"))
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown or protected optimization parameters: {sorted(unknown)}")
    missing = [name for name in required_names if name not in values]
    if missing:
        raise ValueError(f"Missing optimization parameters: {missing}")
    result = parameter_defaults(values)
    return result


def describe_parameters() -> List[Dict[str, Any]]:
    return [
        {
            "name": item.name,
            "default": item.default,
            "min": item.lower,
            "max": item.upper,
            "trainable": item.trainable,
            "llm_allowed": item.llm_allowed,
            "description": item.description,
        }
        for item in OPTIMIZATION_PARAMETERS
    ]
