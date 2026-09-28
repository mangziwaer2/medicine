"""Shared components for multi-nuclide inverse bioassay experiments.

The canonical user-facing ODE experiment is ``run_ode_dataset_pipeline.py``.
This module keeps the reusable case schema, LLM planner, numerical inversion,
and compartment-ODE backend in one place.

The fixed-model mainline separates two optimization layers:

1. The numerical inner loop estimates intake and intake time from bioassay
   observations.
2. The LLM outer loop proposes a whitelisted optimization operator and, when
   several verified models are compatible, a registered model subset.

The outer loop may select only registered ODE model IDs. It cannot create or
modify compartment graphs, measurements, transfer paths, or kinetic rates.
The current registry is transcribed from ICRP Publication 137 and is used for
theoretical validation, not clinical dose assessment.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize, minimize_scalar

try:
    from .optimization_schema import (
        LLM_ALLOWED_PARAMETER_NAMES,
        PARAMETER_BLOCKS,
        PARAMETER_BY_NAME,
        parameter_defaults,
        parameter_bounds,
        validate_parameter_values,
    )
except ImportError:
    from optimization_schema import (
        LLM_ALLOWED_PARAMETER_NAMES,
        PARAMETER_BLOCKS,
        PARAMETER_BY_NAME,
        parameter_defaults,
        parameter_bounds,
        validate_parameter_values,
    )


SUPPORTED_OPTIMIZERS = (
    "candidate_points",
    "differential_evolution",
    "differential_evolution_then_lbfgsb",
    "lbfgsb",
    "profile_likelihood",
)

try:
    from .compartment_ode import (
        CompartmentODEForwardModel,
        ODEModelRegistry,
        ODE_MODEL_ID,
        SUPPORTED_SOLVERS,
    )
except ImportError:
    from compartment_ode import (
        CompartmentODEForwardModel,
        ODEModelRegistry,
        ODE_MODEL_ID,
        SUPPORTED_SOLVERS,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "Qwen3-1.7B"
DEFAULT_LLM_GATE_PATH = ROOT / "models" / "llm_gate_mlp_icrp.pt"
DEFAULT_LLM_GATE_THRESHOLD = 0.22
OBSERVATION_NODE_TYPES = {
    "24-h Urine": "urine",
    "24-h Feces": "feces",
    "T-body": "whole_body",
    "Thyroid-Bioa": "thyroid",
    "Skelet-Bioa": "skeleton",
}

OBSERVATION_ALIASES = {
    "urine": "24-h Urine",
    "24 h urine": "24-h Urine",
    "24-h urine": "24-h Urine",
    "feces": "24-h Feces",
    "24 h feces": "24-h Feces",
    "24-h feces": "24-h Feces",
    "whole body": "T-body",
    "whole-body": "T-body",
    "t-body": "T-body",
    "thyroid": "Thyroid-Bioa",
    "thyroid-bioa": "Thyroid-Bioa",
    "skeleton": "Skelet-Bioa",
    "skelet-bioa": "Skelet-Bioa",
}


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def normalize_observation_type(value: Any) -> str:
    text = str(value).strip()
    return OBSERVATION_ALIASES.get(text.lower(), text)


def intake_time_bounds_for_case(
    case: "CaseInput",
    earliest_observation_d: float,
) -> Tuple[float, float]:
    """Return legal intake-time bounds for the case's declared time origin.

    ``time_d`` is an absolute coordinate on the declared origin.  Negative
    intake times are therefore legal when the origin is the first observation
    or another external reference, but not when the exposure window itself is
    defined to start at zero.  A post-intake coordinate fixes intake time at
    zero and should not be used to estimate an independent time shift.
    """
    context = case.exposure_context if isinstance(case.exposure_context, dict) else {}
    reference = str(
        context.get("time_reference", "exposure_window_origin")
    ).strip().lower()
    if reference in {"post_intake", "since_intake", "intake_origin"}:
        lower = upper = 0.0
    elif reference in {"first_observation", "first_detection", "calendar_reference", "external_reference"}:
        lower = _finite_float(
            context.get("minimum_intake_time_d", -14.0), -14.0
        ) or -14.0
        upper = earliest_observation_d
    else:
        lower = _finite_float(
            context.get("minimum_intake_time_d", 0.0), 0.0
        ) or 0.0
        upper = earliest_observation_d
    requested_upper = _finite_float(context.get("maximum_intake_time_d"))
    if requested_upper is not None:
        upper = min(upper, requested_upper)
    lower = max(-3650.0, min(float(lower), float(upper)))
    upper = min(float(earliest_observation_d), max(float(lower), float(upper)))
    if upper <= lower:
        if reference in {"post_intake", "since_intake", "intake_origin"}:
            return (0.0, 1e-9)
        if reference in {"exposure_window_origin", "exposure_origin"}:
            return (0.0, 1e-9)
        return (-14.0, float(earliest_observation_d))
    return (lower, upper)


@dataclass
class Observation:
    time_d: float
    type: str
    value_bq: float
    sigma_bq: Optional[float] = None
    detection_limit_bq: Optional[float] = None
    is_censored: bool = False
    measurement_id: str = ""
    unit: str = "Bq"
    sample_volume_l: Optional[float] = None
    sample_window_d: Optional[float] = None
    measurement_method: str = ""
    quality_flag: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], index: int = 0) -> "Observation":
        value = data.get("value_bq", data.get("value_Bq", data.get("value")))
        sigma = data.get("sigma_bq", data.get("sigma_Bq", data.get("uncertainty_bq")))
        detection_limit = data.get(
            "detection_limit_bq",
            data.get("detection_limit_Bq", data.get("lod_bq")),
        )
        is_censored = bool(data.get("is_censored", False))
        return cls(
            time_d=float(data["time_d"]),
            type=normalize_observation_type(data["type"]),
            value_bq=float(value),
            sigma_bq=_finite_float(sigma),
            detection_limit_bq=_finite_float(detection_limit),
            is_censored=is_censored,
            measurement_id=str(data.get("measurement_id", f"obs-{index + 1}")),
            unit=str(data.get("unit", "Bq")),
            sample_volume_l=_finite_float(data.get("sample_volume_l")),
            sample_window_d=_finite_float(data.get("sample_window_d")),
            measurement_method=str(data.get("measurement_method", "")),
            quality_flag=str(data.get("quality_flag", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.sigma_bq is None:
            result.pop("sigma_bq")
        if self.detection_limit_bq is None:
            result.pop("detection_limit_bq")
        if not self.is_censored:
            result.pop("is_censored")
        for key in ("sample_volume_l", "sample_window_d"):
            if result.get(key) is None:
                result.pop(key, None)
        for key in ("measurement_method", "quality_flag"):
            if not result.get(key):
                result.pop(key, None)
        return result


@dataclass
class NuclideCase:
    nuclide: str
    observations: List[Observation]
    # Every value must name a pre-registered fixed ICRP model.
    candidate_model_ids: List[str] = field(default_factory=list)
    route_candidates: List[str] = field(default_factory=lambda: ["unknown"])
    chemical_form: str = "unknown"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NuclideCase":
        model_ids = data.get("candidate_model_ids", [])
        if isinstance(model_ids, str):
            model_ids = [model_ids]
        observations = [
            Observation.from_dict(item, index=index)
            for index, item in enumerate(data.get("observations", []))
        ]
        routes = data.get("route_candidates", data.get("routes", ["unknown"]))
        if isinstance(routes, str):
            routes = [routes]
        return cls(
            nuclide=str(data["nuclide"]),
            observations=observations,
            candidate_model_ids=[str(item) for item in model_ids],
            route_candidates=[str(item) for item in routes] or ["unknown"],
            chemical_form=str(data.get("chemical_form", "unknown")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nuclide": self.nuclide,
            "candidate_model_ids": self.candidate_model_ids,
            "route_candidates": self.route_candidates,
            "chemical_form": self.chemical_form,
            "observations": [item.to_dict() for item in self.observations],
        }


@dataclass
class CaseInput:
    case_id: str
    nuclides: List[NuclideCase]
    subject: Dict[str, Any] = field(default_factory=dict)
    exposure_context: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseInput":
        exposure_context = dict(data.get("exposure_context", {}))
        if "route_candidates" in data and "route_candidates" not in exposure_context:
            exposure_context["route_candidates"] = data["route_candidates"]
        return cls(
            case_id=str(data.get("case_id", "case-unknown")),
            nuclides=[NuclideCase.from_dict(item) for item in data.get("nuclides", [])],
            subject=dict(data.get("subject", {})),
            exposure_context=exposure_context,
            raw_text=str(data.get("raw_text", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "subject": self.subject,
            "exposure_context": self.exposure_context,
            "nuclides": [item.to_dict() for item in self.nuclides],
        }


def validate_case(
    case: CaseInput,
    dataset_root: Optional[Path] = None,
    forward_model_kind: str = "ode",
    ode_registry: Optional[ODEModelRegistry] = None,
) -> None:
    if not case.nuclides:
        raise ValueError("The case must contain at least one nuclide.")
    if forward_model_kind != "ode":
        raise ValueError("Only the compartment ODE backend is supported.")
    if ode_registry is None or not ode_registry.models:
        raise ValueError("A non-empty verified ICRP ODE registry is required.")
    registry = ode_registry

    for nuclide_case in case.nuclides:
        if not registry.compatible_model_ids(
            nuclide_case.nuclide,
            (item.type for item in nuclide_case.observations),
        ):
            raise ValueError(
                f"No ODE scenario model supports {nuclide_case.nuclide} and "
                "its requested observation types."
            )
        if not nuclide_case.observations:
            raise ValueError(f"{nuclide_case.nuclide} has no observations.")
        time_reference = str(
            case.exposure_context.get("time_reference", "exposure_window_origin")
            if isinstance(case.exposure_context, dict) else "exposure_window_origin"
        ).strip().lower()
        for observation in nuclide_case.observations:
            if (
                observation.time_d < 0
                and time_reference in {
                    "exposure_window_origin", "exposure_origin",
                    "post_intake", "since_intake", "intake_origin",
                }
            ):
                raise ValueError("Observation time_d must be non-negative.")
            if observation.value_bq <= 0:
                raise ValueError(
                    f"Observation {observation.measurement_id} must have "
                    "a positive value_bq for log-loss inversion."
                )


@dataclass
class GraphNode:
    node_id: str
    kind: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    attributes: Dict[str, Any] = field(default_factory=dict)


def build_biokinetic_graph(nuclide_case: NuclideCase) -> Dict[str, Any]:
    """Build an observation-oriented compartment graph for one nuclide.

    The authoritative graph topology and rates are stored in the ODE registry.
    This observation-oriented graph is a compact planner view.
    """

    nodes = [
        GraphNode("intake", "input", {"nuclide": nuclide_case.nuclide}),
        GraphNode("blood", "compartment"),
    ]
    edges = [
        GraphEdge("intake", "blood", "absorption", {"route_candidates": nuclide_case.route_candidates}),
    ]
    seen_nodes = {"intake", "blood"}

    for observation in nuclide_case.observations:
        node_id = OBSERVATION_NODE_TYPES.get(observation.type, observation.type.lower().replace(" ", "_"))
        if node_id not in seen_nodes:
            nodes.append(
                GraphNode(
                    node_id,
                    "compartment_or_measurement",
                    {"observation_types": [observation.type]},
                )
            )
            seen_nodes.add(node_id)
        edges.append(
            GraphEdge(
                "blood",
                node_id,
                "measurement_projection",
                {"measurement_type": observation.type},
            )
        )

    return {
        "nodes": [asdict(item) for item in nodes],
        "edges": [asdict(item) for item in edges],
    }


class StructuredKnowledgeBase:
    """Verified ICRP ODE registry plus page-traceable retrieval context."""

    def __init__(
        self,
        dataset_root: Optional[Path] = None,
        forward_model_kind: str = "ode",
        ode_registry: Optional[ODEModelRegistry] = None,
        icrp_root: Optional[Path] = None,
    ):
        if forward_model_kind != "ode":
            raise ValueError("Only the compartment ODE backend is supported.")
        self.dataset_root = dataset_root
        self.forward_model_kind = forward_model_kind
        self.model_id = ODE_MODEL_ID
        if ode_registry is None or not ode_registry.models:
            raise ValueError("A non-empty verified ICRP ODE registry is required.")
        self.ode_registry = ode_registry
        self.icrp_root = Path(icrp_root) if icrp_root else None
        self.icrp = None
        if self.icrp_root is not None:
            try:
                from .icrp_knowledge import ICRPKnowledgeBase
            except ImportError:
                from icrp_knowledge import ICRPKnowledgeBase
            self.icrp = ICRPKnowledgeBase(self.icrp_root)

    def available_nuclides(self) -> List[str]:
        return sorted({nuclide for nuclide, _ in self.ode_registry.models})

    def compatible_model_ids(
        self,
        nuclide_case: NuclideCase,
    ) -> List[str]:
        required_types = {item.type for item in nuclide_case.observations}
        requested = nuclide_case.candidate_model_ids
        available = self.ode_registry.compatible_model_ids(
            nuclide_case.nuclide,
            required_types,
        )
        if requested:
            available = [item for item in available if item in requested]
        return available

    def retrieve(self, nuclide_case: NuclideCase) -> Dict[str, Any]:
        compatible = self.compatible_model_ids(nuclide_case)
        return {
            "nuclide": nuclide_case.nuclide,
            "model_candidates": [
                self.ode_registry.describe(nuclide_case.nuclide, model_id)
                for model_id in compatible
            ],
            "compatible_model_ids": compatible,
            "graph": build_biokinetic_graph(nuclide_case),
        }

    def retrieve_case(self, case: CaseInput) -> Dict[str, Any]:
        result = {
            "dataset_root": str(self.dataset_root),
            "forward_model_kind": self.forward_model_kind,
            "model_id": self.model_id,
            "available_nuclide_count": len(self.available_nuclides()),
            "nuclides": [self.retrieve(item) for item in case.nuclides],
        }
        if self.icrp is not None:
            result["icrp"] = self.retrieve_icrp(case)
        return result

    def retrieve_icrp(self, case: CaseInput, top_k: int = 3) -> Dict[str, Any]:
        """Retrieve bounded ICRP context with explicit source provenance."""
        if self.icrp is None:
            return {"enabled": False, "records": []}
        records: List[Dict[str, Any]] = []
        for item in case.nuclides:
            compatible = self.compatible_model_ids(item)
            source_metadata = [
                self.ode_registry.describe(item.nuclide, model_id).get("metadata", {})
                for model_id in compatible
            ]
            tables = sorted({
                str(metadata.get("meta_source_table", ""))
                for metadata in source_metadata
                if metadata.get("meta_source_table")
            })
            publications = sorted({
                str(metadata.get("meta_parameter_source", ""))
                for metadata in source_metadata
                if metadata.get("meta_parameter_source")
            })
            nuclide_terms = {
                "I-131": "iodine systemic",
                "Cs-137": "caesium cesium systemic",
            }.get(item.nuclide, item.nuclide)
            query = f"{nuclide_terms} {' '.join(tables)} transfer coefficient"
            hits = self.icrp.search(
                query,
                top_k=max(top_k * 4, 12),
                publication=publications[0] if len(publications) == 1 else None,
            )
            # Prefer the exact source table recorded in the verified registry.
            # Generic lexical hits from another nuclide/model are useful only
            # as secondary context and should not displace the table itself.
            if tables:
                table_terms = [table.lower() for table in tables]
                exact = [
                    hit for hit in hits
                    if any(term in str(hit.get("text", "")).lower() for term in table_terms)
                ]
                remainder = [hit for hit in hits if hit not in exact]
                hits = (exact + remainder)[:top_k]
            else:
                hits = hits[:top_k]
            records.append({
                "nuclide": item.nuclide,
                "query": query,
                "hits": [
                    {
                        "chunk_id": hit["chunk_id"],
                        "publication": hit["publication"],
                        "page": hit["page"],
                        "category": hit["category"],
                        "retrieval_score": hit["retrieval_score"],
                        "text": hit["text"],
                        "official_url": hit["official_url"],
                    }
                    for hit in hits
                ],
            })
        return {
            "enabled": True,
            "index_root": str(self.icrp_root),
            "manifest_counts": self.icrp.manifest.get("counts", {}),
            "records": records,
        }


def _compact_planner_knowledge(
    case: CaseInput,
    knowledge_base: StructuredKnowledgeBase,
    *,
    icrp_text_limit: int = 400,
) -> Dict[str, Any]:
    """Build a bounded planner context for small local language models.

    The executor remains authoritative.  The planner only needs registered
    model IDs, a concise topology summary, and a few source excerpts.  Passing
    every duplicated graph field and full page text wastes context and makes a
    1.7B model less reliable without adding executable information.
    """
    full = knowledge_base.retrieve_case(case)
    compact: Dict[str, Any] = {
        "forward_model_kind": full.get("forward_model_kind"),
        "registered_model_policy": "registered_model_ids_only",
        "nuclides": [],
    }
    for item in full.get("nuclides", []):
        models = []
        for model in item.get("model_candidates", []):
            models.append({
                "model_id": model.get("model_id"),
                "nuclide": model.get("nuclide"),
                "compartment_count": len(model.get("compartments", [])),
                "intake_compartment": model.get("intake_compartment"),
                "observation_types": model.get("observation_types", []),
                "transfer_count": model.get("transfer_count", 0),
                "metadata": {
                    key: value
                    for key, value in model.get("metadata", {}).items()
                    if key.startswith("meta_")
                },
            })
        compact["nuclides"].append({
            "nuclide": item.get("nuclide"),
            "compatible_model_ids": item.get("compatible_model_ids", []),
            "models": models,
        })

    icrp = full.get("icrp")
    if isinstance(icrp, dict):
        compact["icrp"] = {
            "enabled": bool(icrp.get("enabled")),
            "records": [
                {
                    "nuclide": record.get("nuclide"),
                    "hits": [
                        {
                            "publication": hit.get("publication"),
                            "page": hit.get("page"),
                            "chunk_id": hit.get("chunk_id"),
                            "category": hit.get("category"),
                            "retrieval_score": hit.get("retrieval_score"),
                            "text": str(hit.get("text", ""))[:icrp_text_limit],
                            "official_url": hit.get("official_url"),
                        }
                        for hit in record.get("hits", [])[:2]
                    ],
                }
                for record in icrp.get("records", [])
            ],
        }
    return compact

def _measurement_weight(observation: Observation, configured_weights: Dict[str, Any]) -> float:
    configured = _finite_float(configured_weights.get(observation.type), 1.0)
    if configured is None or configured <= 0:
        configured = 1.0
    if observation.sigma_bq is None:
        return configured
    relative_sigma = observation.sigma_bq / max(abs(observation.value_bq), 1e-12)
    precision_weight = min(25.0, max(0.05, 1.0 / max(relative_sigma * relative_sigma, 1e-6)))
    return configured * precision_weight


def _log_residual(observed: float, predicted: float) -> float:
    return math.log(max(predicted, 1e-12)) - math.log(max(observed, 1e-12))


def _observation_residual(observation: Observation, predicted: float) -> float:
    """Residual supporting left-censored measurements at a detection limit."""
    if observation.is_censored and observation.detection_limit_bq is not None:
        limit = max(float(observation.detection_limit_bq), 1e-12)
        # A prediction below the limit is compatible with a non-detect.
        if float(predicted) <= limit:
            return 0.0
        return math.log(max(float(predicted), 1e-12)) - math.log(limit)
    return _log_residual(observation.value_bq, predicted)


def _normalise_bounds(
    action: Dict[str, Any],
    earliest_observation_d: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    raw_bounds = action.get("bounds", {})
    if not isinstance(raw_bounds, dict):
        raw_bounds = {}

    def read_pair(name: str, default: Tuple[float, float]) -> Tuple[float, float]:
        value = raw_bounds.get(name, default)
        if isinstance(value, dict):
            value = [value.get("min", default[0]), value.get("max", default[1])]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return default
        low = _finite_float(value[0], default[0])
        high = _finite_float(value[1], default[1])
        if low is None or high is None:
            return default
        return low, high

    intake_bounds = read_pair("log10_intake", (0.0, 8.0))
    time_bounds = read_pair("intake_time_d", (-14.0, earliest_observation_d))
    intake_bounds = (
        max(0.0, min(intake_bounds)),
        min(12.0, max(intake_bounds)),
    )
    time_bounds = (
        max(-3650.0, min(time_bounds)),
        min(earliest_observation_d, max(time_bounds)),
    )
    if intake_bounds[0] >= intake_bounds[1]:
        intake_bounds = (0.0, 8.0)
    if time_bounds[0] >= time_bounds[1]:
        time_bounds = (-14.0, earliest_observation_d)
    return intake_bounds, time_bounds


def _identifiability_diagnostics(
    parameter_names: Sequence[str],
    parameter_values: Dict[str, float],
    bounds_by_name: Dict[str, Sequence[float]],
    predict_from_values: Any,
    observations: Sequence[Observation],
) -> Dict[str, Any]:
    """Estimate local practical identifiability from a scaled log-output Jacobian."""

    names = [name for name in parameter_names if name in bounds_by_name]
    row_indices = [index for index, item in enumerate(observations) if not item.is_censored]
    if not names or not row_indices:
        return {
            "status": "insufficient_data",
            "parameter_names": names,
            "uncensored_observations": len(row_indices),
            "recommended_parameter_names": list(PARAMETER_BLOCKS["BASIC"]),
        }

    columns = []
    sensitivity_norms: Dict[str, float] = {}
    for name in names:
        low, high = [float(value) for value in bounds_by_name[name]]
        width = max(high - low, 1e-12)
        center = max(low, min(high, float(parameter_values.get(name, PARAMETER_BY_NAME[name].default))))
        step = max(width * 0.02, 1e-6)
        lower_value = max(low, center - step)
        upper_value = min(high, center + step)
        if upper_value <= lower_value:
            column = np.zeros(len(row_indices), dtype=float)
        else:
            lower_parameters = dict(parameter_values)
            upper_parameters = dict(parameter_values)
            lower_parameters[name] = lower_value
            upper_parameters[name] = upper_value
            lower_predictions = predict_from_values(lower_parameters)
            upper_predictions = predict_from_values(upper_parameters)
            denominator = (upper_value - lower_value) / width
            column = np.asarray([
                (
                    math.log(max(float(upper_predictions[index]), 1e-12))
                    - math.log(max(float(lower_predictions[index]), 1e-12))
                ) / max(denominator, 1e-12)
                for index in row_indices
            ], dtype=float)
        columns.append(column)
        sensitivity_norms[name] = float(np.linalg.norm(column))

    jacobian = np.column_stack(columns)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = max(jacobian.shape) * max(largest, 1e-12) * 1e-4
    rank = int(np.sum(singular_values > tolerance))
    smallest = float(singular_values[-1]) if singular_values.size else 0.0
    condition_number = (
        float(largest / smallest)
        if smallest > tolerance and largest > 0.0
        else float("inf")
    )
    correlations: Dict[str, Dict[str, float]] = {name: {} for name in names}
    for left_index, left_name in enumerate(names):
        left = jacobian[:, left_index]
        for right_index, right_name in enumerate(names):
            right = jacobian[:, right_index]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            correlations[left_name][right_name] = (
                float(np.dot(left, right) / denominator) if denominator > 0 else 0.0
            )

    recommended = list(PARAMETER_BLOCKS["BASIC"])
    maximum_norm = max(sensitivity_norms.values(), default=0.0)
    for name in names:
        if name in recommended:
            continue
        sufficiently_sensitive = sensitivity_norms[name] >= max(1e-6, maximum_norm * 0.05)
        not_collinear = all(
            abs(correlations[name].get(selected, 0.0)) < 0.98
            for selected in recommended
        )
        if sufficiently_sensitive and not_collinear and len(row_indices) >= len(recommended) + 1:
            recommended.append(name)

    return {
        "status": "ok",
        "parameter_names": names,
        "uncensored_observations": len(row_indices),
        "jacobian_shape": list(jacobian.shape),
        "rank": rank,
        "condition_number": condition_number,
        "singular_values": [float(value) for value in singular_values],
        "sensitivity_norms": sensitivity_norms,
        "parameter_correlations": correlations,
        "locally_identifiable": bool(
            rank == len(names)
            and len(row_indices) >= len(names)
            and condition_number <= 1e4
        ),
        "recommended_parameter_names": recommended,
    }


def invert_one_nuclide(
    nuclide_case: NuclideCase,
    action: Dict[str, Any],
    forward_model: Any,
    seed: int,
) -> Dict[str, Any]:
    """Run the numerical inner loop for one nuclide."""

    earliest_time = min(item.time_d for item in nuclide_case.observations)
    intake_bounds, time_bounds = _normalise_bounds(action, earliest_time)
    parameter_names = [
        str(name) for name in action.get(
            "parameter_names", ["log10_intake", "intake_time_d"]
        ) if str(name) in LLM_ALLOWED_PARAMETER_NAMES
    ]
    if "log10_intake" not in parameter_names:
        parameter_names.insert(0, "log10_intake")
    if "intake_time_d" not in parameter_names:
        parameter_names.insert(1, "intake_time_d")
    parameter_names = list(dict.fromkeys(parameter_names))
    all_parameter_bounds = parameter_bounds(action.get("parameter_bounds"), parameter_names)
    extra_bounds = {
        name: bounds for name, bounds in all_parameter_bounds.items()
        if name not in {"log10_intake", "intake_time_d"}
    }
    bounds_by_name = {
        "log10_intake": list(intake_bounds),
        "intake_time_d": list(time_bounds),
        **extra_bounds,
    }
    configured_weights = action.get("measurement_weights", {})
    if not isinstance(configured_weights, dict):
        configured_weights = {}
    maxiter = int(max(1, min(120, _finite_float(action.get("maxiter"), 60) or 60)))
    popsize = int(max(5, min(20, _finite_float(action.get("popsize"), 8) or 8)))
    forward_predict_budget = int(max(
        1,
        min(100000, _finite_float(action.get("forward_predict_budget"), 100000) or 100000),
    ))
    optimization_predict_budget = max(1, forward_predict_budget - 1)
    optimizer_name = str(action.get("optimizer", "differential_evolution")).lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        optimizer_name = "differential_evolution"

    candidate_model_ids = action.get("model_ids", [])
    if isinstance(candidate_model_ids, str):
        candidate_model_ids = [candidate_model_ids]
    candidate_model_ids = [str(item) for item in candidate_model_ids]
    ode_solver = str(action.get("ode_solver", "LSODA"))
    candidate_results: List[Dict[str, Any]] = []

    for model_index, model_id in enumerate(candidate_model_ids):
        candidate_started = time.perf_counter()
        forward_predict_calls = 0
        # Keep optimization calls separate from the final scoring call.  The
        # latter is reserved so that a budget-exhausted optimizer still gets a
        # prediction for its selected point.  Diagnostic calls use the same
        # hard cap and are served from the last prediction when no budget
        # remains.
        optimization_phase = True
        last_predictions: Optional[List[float]] = None

        def counted_predict(
            intake_bq: float,
            intake_time_d: float,
            observations: Sequence[Any],
            parameters: Optional[Dict[str, float]] = None,
        ) -> List[float]:
            nonlocal forward_predict_calls, last_predictions
            call_limit = (
                optimization_predict_budget
                if optimization_phase
                else forward_predict_budget
            )
            if forward_predict_calls >= call_limit:
                if last_predictions is not None:
                    return list(last_predictions)
                return [1e-12 for _ in observations]
            forward_predict_calls += 1
            predictions = forward_model.predict(
                nuclide_case.nuclide,
                model_id,
                intake_bq,
                intake_time_d,
                observations,
                solver=ode_solver,
                parameters=parameters,
            )
            last_predictions = [float(value) for value in predictions]
            return predictions

        try:
            supported = set(forward_model.supported_observation_types(
                nuclide_case.nuclide,
                model_id,
            ))
        except (FileNotFoundError, KeyError) as error:
            candidate_results.append({
                "model_id": model_id,
                "status": "invalid",
                "error": str(error),
            })
            continue

        missing = sorted({item.type for item in nuclide_case.observations} - supported)
        if missing:
            candidate_results.append({
                "model_id": model_id,
                "status": "invalid",
                "error": f"Missing observation columns: {missing}",
            })
            continue

        def unpack(parameters_vector: Sequence[float]) -> Dict[str, float]:
            values = parameter_defaults()
            for name, value in zip(parameter_names, parameters_vector):
                values[name] = float(value)
            return values

        best_evaluated: Optional[Tuple[float, List[float]]] = None

        def objective(parameters_vector: Sequence[float]) -> float:
            nonlocal best_evaluated
            if forward_predict_calls >= optimization_predict_budget:
                return 1e12
            values = unpack(parameters_vector)
            log10_intake = values["log10_intake"]
            intake_time_d = values["intake_time_d"]
            intake_bq = 10.0 ** float(log10_intake)
            predictions = counted_predict(
                intake_bq,
                float(intake_time_d),
                nuclide_case.observations,
                values,
            )
            residuals = [
                _observation_residual(observation, prediction)
                for observation, prediction in zip(nuclide_case.observations, predictions)
            ]
            weights = [
                _measurement_weight(observation, configured_weights)
                for observation in nuclide_case.observations
            ]
            score = float(np.average(np.square(residuals), weights=weights))
            vector = [float(value) for value in parameters_vector]
            if best_evaluated is None or score < best_evaluated[0]:
                best_evaluated = (score, vector)
            return score

        candidate_point_rows = []
        seen_candidate_keys = set()
        best_candidate_vector = None
        for raw_point in action.get("candidate_points", []):
            if forward_predict_calls >= optimization_predict_budget:
                break
            if not isinstance(raw_point, dict):
                continue
            point_model_id = str(raw_point.get("model_id", "")).strip()
            if point_model_id and point_model_id != model_id:
                continue
            try:
                point_values = {
                    key: value for key, value in raw_point.items()
                    if key != "model_id"
                }
                point = validate_parameter_values(point_values, required=parameter_names)
                vector = [point[name] for name in parameter_names]
                if any(vector[index] < bounds_by_name[name][0] or vector[index] > bounds_by_name[name][1]
                       for index, name in enumerate(parameter_names)):
                    continue
                key = tuple(round(float(value), 8) for value in vector)
                if key in seen_candidate_keys:
                    continue
                seen_candidate_keys.add(key)
                point_loss = float(objective(vector))
                candidate_point_rows.append({"parameters": point, "objective": point_loss})
                if best_candidate_vector is None or point_loss < best_candidate_vector[0]:
                    best_candidate_vector = (point_loss, vector)
            except (TypeError, ValueError, KeyError):
                continue

        if optimizer_name == "candidate_points":
            if best_candidate_vector is None:
                midpoint = [
                    0.5 * (bounds_by_name[name][0] + bounds_by_name[name][1])
                    for name in parameter_names
                ]
                midpoint_loss = objective(midpoint)
                best_candidate_vector = (midpoint_loss, midpoint)
                candidate_point_rows.append({
                    "parameters": unpack(midpoint),
                    "objective": float(midpoint_loss),
                    "source": "deterministic_midpoint_fallback",
                })
            result = OptimizeResult(
                x=np.asarray(best_candidate_vector[1], dtype=float),
                fun=float(best_candidate_vector[0]),
                nfev=len(candidate_point_rows),
                nit=0,
                success=True,
                message="Candidate points evaluated without a full optimizer.",
            )
            result_values = unpack(result.x)
            log10_intake = result_values["log10_intake"]
            intake_time_d = result_values["intake_time_d"]
        elif optimizer_name == "profile_likelihood" and len(parameter_names) == 2:
            # For uncensored observations, intake is analytically identifiable
            # at a fixed time. With censored observations, profile intake
            # numerically so the detection-limit likelihood is not discarded.
            def best_log10_intake_at_time(intake_time: float) -> float:
                unit_predictions = counted_predict(
                    1.0, float(intake_time), nuclide_case.observations,
                )
                valid_rows = [
                    (obs, max(float(pred), 1e-12), _measurement_weight(obs, configured_weights))
                    for obs, pred in zip(nuclide_case.observations, unit_predictions)
                    if not obs.is_censored
                ]
                # Profile the amplitude analytically from uncensored rows.  A
                # censored row is an upper-limit constraint, not an observed
                # activity value; including its detection limit in a nested
                # scalar amplitude search can exhaust the ODE budget and bias
                # the result.  The outer profile objective still evaluates and
                # penalizes every censored constraint.
                if valid_rows:
                    log_intake = sum(
                        weight * (math.log(max(obs.value_bq, 1e-12)) - math.log(pred))
                        for obs, pred, weight in valid_rows
                    ) / sum(weight for _, _, weight in valid_rows)
                    return min(
                        intake_bounds[1],
                        max(intake_bounds[0], log_intake / math.log(10.0)),
                    )

                def intake_objective(log10_intake: float) -> float:
                    predictions_local = counted_predict(
                        10.0 ** float(log10_intake),
                        float(intake_time),
                        nuclide_case.observations,
                    )
                    residuals_local = [
                        _observation_residual(obs, pred)
                        for obs, pred in zip(nuclide_case.observations, predictions_local)
                    ]
                    weights_local = [
                        _measurement_weight(obs, configured_weights)
                        for obs in nuclide_case.observations
                    ]
                    return float(np.average(np.square(residuals_local), weights=weights_local))

                result_intake = minimize_scalar(
                    intake_objective,
                    bounds=intake_bounds,
                    method="bounded",
                    options={"xatol": 1e-5, "maxiter": maxiter * 4},
                )
                return float(result_intake.x)

            def profile_objective(intake_time: float) -> float:
                log10_intake = best_log10_intake_at_time(float(intake_time))
                intake = 10.0 ** log10_intake
                predictions_local = counted_predict(
                    intake, float(intake_time), nuclide_case.observations,
                )
                residuals_local = [
                    _observation_residual(obs, pred)
                    for obs, pred in zip(nuclide_case.observations, predictions_local)
                ]
                weights_local = [_measurement_weight(obs, configured_weights) for obs in nuclide_case.observations]
                return float(np.average(np.square(residuals_local), weights=weights_local))

            result = minimize_scalar(
                profile_objective,
                bounds=time_bounds,
                method="bounded",
                options={"xatol": 1e-4, "maxiter": maxiter * 10},
            )
            intake_time_d = float(result.x)
            log10_intake = best_log10_intake_at_time(intake_time_d)
        else:
            dimension = len(parameter_names)
            material_budget = max(
                1,
                forward_predict_budget // max(1, len(candidate_model_ids)),
            )
            reserved_calls = max(10, 4 * dimension)
            population_calls = max(5, popsize * dimension)
            budget_limited_maxiter = max(
                1,
                (max(1, material_budget - reserved_calls) // population_calls) - 1,
            )
            maxiter = min(maxiter, budget_limited_maxiter)
            de_kwargs = {
                "bounds": [bounds_by_name[name] for name in parameter_names],
                "seed": seed + model_index,
                "maxiter": maxiter,
                "popsize": popsize,
                "polish": True,
                "updating": "immediate",
                "workers": 1,
            }
            if best_candidate_vector is not None:
                de_kwargs["x0"] = np.asarray(best_candidate_vector[1], dtype=float)
            if optimizer_name == "lbfgsb":
                if best_candidate_vector is not None:
                    initial_vector = np.asarray(best_candidate_vector[1], dtype=float)
                else:
                    initial_vector = np.asarray(
                        [0.5 * (low + high) for low, high in de_kwargs["bounds"]],
                        dtype=float,
                    )
                result = minimize(
                    objective,
                    initial_vector,
                    method="L-BFGS-B",
                    bounds=de_kwargs["bounds"],
                    options={"maxiter": maxiter * 5, "ftol": 1e-12, "maxls": 20},
                )
            else:
                result = differential_evolution(
                    objective,
                    **de_kwargs,
                )
            if (
                forward_predict_calls >= optimization_predict_budget
                and best_evaluated is not None
            ):
                result.x = np.asarray(best_evaluated[1], dtype=float)
                result.fun = float(best_evaluated[0])
            if optimizer_name == "differential_evolution_then_lbfgsb":
                local_result = minimize(
                    objective,
                    np.asarray(result.x, dtype=float),
                    method="L-BFGS-B",
                    bounds=[bounds_by_name[name] for name in parameter_names],
                    options={"maxiter": maxiter * 5, "ftol": 1e-12, "maxls": 20},
                )
                if float(local_result.fun) <= float(result.fun):
                    result = local_result
            result_values = unpack(result.x)
            log10_intake, intake_time_d = result_values["log10_intake"], result_values["intake_time_d"]
        if optimizer_name == "profile_likelihood":
            result.x = np.asarray([log10_intake, intake_time_d], dtype=float)
        # Reserve one call for the final prediction.  This also prevents
        # profile-likelihood's nested scalar minimizers from bypassing the
        # budget through direct calls to counted_predict.
        optimization_phase = False
        intake_bq = 10.0 ** log10_intake
        final_values = unpack(result.x)
        predictions = counted_predict(
            intake_bq,
            intake_time_d,
            nuclide_case.observations,
            final_values,
        )
        residuals = [
            _observation_residual(observation, prediction)
            for observation, prediction in zip(nuclide_case.observations, predictions)
        ]
        weights = [
            _measurement_weight(observation, configured_weights)
            for observation in nuclide_case.observations
        ]
        unweighted_loss = float(np.mean(np.square(residuals)))
        weighted_loss = float(np.average(np.square(residuals), weights=weights))
        identifiability = None
        identifiability_calls = 0
        if bool(action.get("run_identifiability_check", False)):
            diagnostic_names = [
                str(name) for name in action.get(
                    "identifiability_parameter_names",
                    PARAMETER_BLOCKS["BASIC"],
                )
                if str(name) in LLM_ALLOWED_PARAMETER_NAMES
            ]
            diagnostic_bounds = parameter_bounds(
                action.get("parameter_bounds"), diagnostic_names
            )
            diagnostic_bounds["log10_intake"] = list(intake_bounds)
            diagnostic_bounds["intake_time_d"] = list(time_bounds)
            calls_before_diagnostics = forward_predict_calls

            def predict_diagnostic(values: Dict[str, float]) -> List[float]:
                return counted_predict(
                    10.0 ** float(values["log10_intake"]),
                    float(values["intake_time_d"]),
                    nuclide_case.observations,
                    values,
                )

            identifiability = _identifiability_diagnostics(
                diagnostic_names,
                final_values,
                diagnostic_bounds,
                predict_diagnostic,
                nuclide_case.observations,
            )
            identifiability_calls = forward_predict_calls - calls_before_diagnostics
        prediction_rows = []
        for observation, prediction, residual in zip(
            nuclide_case.observations,
            predictions,
            residuals,
        ):
            prediction_rows.append({
                "measurement_id": observation.measurement_id,
                "time_d": observation.time_d,
                "type": observation.type,
                "observed_bq": observation.value_bq,
                "predicted_bq": float(prediction),
                "relative_error": (
                    None if observation.is_censored
                    else float(prediction / observation.value_bq - 1.0)
                ),
                "log_residual": float(residual),
            })
        candidate_results.append({
            "model_id": model_id,
            "route": str(action.get("route", "unknown")),
            "status": "ok",
            "intake_bq_estimate": float(intake_bq),
            "intake_time_d_estimate": float(intake_time_d),
            "parameter_estimates": final_values,
            "unweighted_log_mse": unweighted_loss,
            "weighted_log_mse": weighted_loss,
            "optimizer": {
                "name": optimizer_name,
                "maxiter": maxiter,
                "popsize": popsize,
                "nfev": int(getattr(result, "nfev", 0)),
                "nit": int(getattr(result, "nit", 0)),
                "forward_predict_calls": forward_predict_calls,
                "forward_predict_budget": forward_predict_budget,
                "budget_exhausted": bool(forward_predict_calls >= forward_predict_budget),
                "wall_time_s": float(time.perf_counter() - candidate_started),
                "success": bool(result.success),
                "message": str(result.message),
            },
            "ode_solver": ode_solver,
            "predictions": prediction_rows,
            "candidate_point_evaluations": candidate_point_rows,
            "candidate_point_count": (
                len(candidate_point_rows)
                if bool(action.get("count_as_candidate_points", True)) else 0
            ),
            "candidate_point_best_objective": (
                min(
                    (float(item["objective"]) for item in candidate_point_rows),
                    default=None,
                )
            ),
            "candidate_improved_incumbent": (
                bool(
                    candidate_point_rows
                    and _finite_float(action.get("incumbent_objective")) is not None
                    and min(float(item["objective"]) for item in candidate_point_rows)
                    < float(action["incumbent_objective"]) - 1e-12
                )
                if optimizer_name == "candidate_points"
                and bool(action.get("count_as_candidate_points", True))
                else None
            ),
            "identifiability": identifiability,
            "identifiability_forward_predict_calls": identifiability_calls,
        })

    valid_results = [item for item in candidate_results if item["status"] == "ok"]
    valid_results.sort(key=lambda item: item["weighted_log_mse"])
    if not valid_results:
        return {
            "nuclide": nuclide_case.nuclide,
            "status": "failed",
            "error": "No compatible material type could be optimized.",
            "candidate_results": candidate_results,
        }

    best = valid_results[0]
    return {
        "nuclide": nuclide_case.nuclide,
        "status": "ok",
        "best": best,
        "ranked_candidates": valid_results,
        "candidate_results": candidate_results,
        "search_bounds": {
            "log10_intake": list(intake_bounds),
            "intake_time_d": list(time_bounds),
            "parameter_bounds": bounds_by_name,
        },
    }


def execute_numerical_plan(
    case: CaseInput,
    plan: Dict[str, Any],
    forward_model: CompartmentODEForwardModel,
    seed: int,
) -> Dict[str, Any]:
    numerical_started = time.perf_counter()
    actions = {
        str(item.get("nuclide")): item
        for item in plan.get("actions", [])
        if isinstance(item, dict) and item.get("nuclide")
    }
    per_nuclide = []
    total_evaluations = 0
    total_iterations = 0
    total_forward_predict_calls = 0
    total_candidate_points = 0
    total_candidate_improvements = 0
    for index, nuclide_case in enumerate(case.nuclides):
        action = actions.get(nuclide_case.nuclide)
        if action is None:
            per_nuclide.append({
                "nuclide": nuclide_case.nuclide,
                "status": "failed",
                "error": "No action was supplied for this nuclide.",
            })
            continue
        result = invert_one_nuclide(
            nuclide_case,
            action,
            forward_model,
            seed=seed + index * 1000,
        )
        if result.get("status") == "ok":
            total_candidate_points += sum(
                int(item.get("candidate_point_count", 0))
                for item in result.get("candidate_results", [])
                if item.get("status") == "ok"
            )
            total_candidate_improvements += sum(
                item.get("candidate_improved_incumbent") is True
                for item in result.get("candidate_results", [])
                if item.get("status") == "ok"
            )
            valid_candidates = [
                item for item in result.get("candidate_results", [])
                if item.get("status") == "ok"
            ]
            total_evaluations += sum(
                int(item["optimizer"].get("nfev", 0))
                for item in valid_candidates
            )
            total_iterations += sum(
                int(item["optimizer"].get("nit", 0))
                for item in valid_candidates
            )
            total_forward_predict_calls += sum(
                int(item["optimizer"].get("forward_predict_calls", 0))
                for item in valid_candidates
            )
        per_nuclide.append(result)

    valid = [item for item in per_nuclide if item.get("status") == "ok"]
    failed_count = len(per_nuclide) - len(valid)
    # Keep outer-round scores comparable even if the LLM changes measurement
    # weights: model selection uses the unweighted log-MSE.
    fit_score = sum(item["best"]["unweighted_log_mse"] for item in valid)
    objective_score = float(fit_score + failed_count * 100.0 + total_evaluations * 1e-8)
    return {
        "status": "ok" if failed_count == 0 else "partial",
        "objective_score": objective_score,
        "fit_score": float(fit_score),
        "failed_nuclide_count": failed_count,
        "total_function_evaluations": total_evaluations,
        "total_optimizer_iterations": total_iterations,
        "total_forward_predict_calls": total_forward_predict_calls,
        "total_candidate_points": total_candidate_points,
        "total_candidate_improvements": total_candidate_improvements,
        "numerical_wall_time_s": float(time.perf_counter() - numerical_started),
        "per_nuclide": per_nuclide,
    }


def _default_weights(observations: Iterable[Observation]) -> Dict[str, float]:
    return {item.type: 1.0 for item in observations}


def heuristic_plan(
    case: CaseInput,
    knowledge_base: StructuredKnowledgeBase,
    history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a safe numerical-plan skeleton, not a high-level policy decision.

    The function fills registered models, legal parameter bounds, solver,
    measurement weights, and a cheap default optimizer. Operator planners
    (deterministic baseline or gated LLM) may then transform this skeleton.
    """
    context = case.exposure_context if isinstance(case.exposure_context, dict) else {}
    time_reference = str(
        context.get("time_reference", "exposure_window_origin")
    ).strip().lower()
    previous_best = {}
    if history:
        previous_best = {
            item["nuclide"]: item
            for item in history[-1].get("best_by_nuclide", [])
            if isinstance(item, dict) and item.get("nuclide")
        }

    actions = []
    for nuclide_case in case.nuclides:
        knowledge = knowledge_base.retrieve(nuclide_case)
        compatible = knowledge["compatible_model_ids"]
        if not compatible:
            raise ValueError(
                f"No verified registered model is compatible with {nuclide_case.nuclide}."
            )
        previous = previous_best.get(nuclide_case.nuclide)
        if previous and previous.get("model_id") in compatible:
            model_ids = [previous["model_id"]]
            estimate = _finite_float(previous.get("intake_bq_estimate"), 1e3) or 1e3
            log_estimate = math.log10(max(1.0, estimate))
            intake_bounds = [max(0.0, log_estimate - 1.5), min(12.0, log_estimate + 1.5)]
            time_estimate = _finite_float(previous.get("intake_time_d_estimate"), 0.0) or 0.0
            earliest = min(item.time_d for item in nuclide_case.observations)
            default_lower, _ = intake_time_bounds_for_case(case, earliest)
            time_bounds = [
                max(default_lower, time_estimate - 3.0),
                min(earliest, time_estimate + 3.0),
            ]
        else:
            model_ids = compatible
            intake_bounds = [0.0, 8.0]
            time_bounds = list(
                intake_time_bounds_for_case(
                    case,
                    min(item.time_d for item in nuclide_case.observations),
                )
            )

        actions.append({
            "nuclide": nuclide_case.nuclide,
            "model_id": knowledge_base.model_id,
            "route": nuclide_case.route_candidates[0],
            "model_ids": model_ids,
            "unknown_parameters": ["log10_intake", "intake_time_d"],
            "parameter_names": ["log10_intake", "intake_time_d"],
            "parameter_bounds": {},
            "bounds": {
                "log10_intake": intake_bounds,
                "intake_time_d": time_bounds,
            },
            "optimizer": "profile_likelihood",
            "ode_solver": "LSODA",
            "maxiter": 60,
            "popsize": 8,
            "measurement_weights": _default_weights(nuclide_case.observations),
            "reason": (
                "Evaluate all compatible ODE scenarios, then narrow the next "
                "round to the current best candidate."
                if time_reference in {"exposure_window_origin", "exposure_origin"}
                else "Evaluate all compatible ODE models, then narrow the next round to the current best candidate."
            ),
        })
    return {
        "actions": actions,
        "reasoning_summary": "Deterministic baseline planner used for validation or LLM fallback.",
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found in planner output.")
    depth = 0
    in_string = False
    escaped = False
    end = None
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError("Planner JSON object is incomplete.")
    value = json.loads(cleaned[start:end])
    if not isinstance(value, dict):
        raise ValueError("Planner JSON root must be an object.")
    return value


class QwenClient:
    def __init__(self, model_path: Path, adapter_path: Optional[Path] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        load_kwargs: Dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            load_kwargs["dtype"] = torch.bfloat16
        else:
            load_kwargs["dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
        self.adapter_path = adapter_path
        self.adapter_load_info = None
        if adapter_path is not None:
            adapter_path = Path(adapter_path)
            payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {})
            try:
                from .train_qwen_lora import _replace_target_modules
            except ImportError:
                from train_qwen_lora import _replace_target_modules
            _replace_target_modules(
                self.model,
                int(metadata.get("rank", 8)),
                float(metadata.get("alpha", 16.0)),
                float(metadata.get("dropout", 0.0)),
                tuple(metadata.get("target_modules", ("q_proj", "v_proj"))),
            )
            missing, unexpected = self.model.load_state_dict(
                payload.get("adapter_state", payload), strict=False
            )
            missing_adapter = [name for name in missing if "lora_A" in name or "lora_B" in name]
            if missing_adapter:
                raise RuntimeError(
                    f"Missing tensors while loading Qwen adapter {adapter_path}: {missing_adapter[:5]}"
                )
            self.adapter_load_info = {
                "path": str(adapter_path),
                "unexpected_keys": unexpected,
                "metadata": metadata,
            }
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a scientific optimization planner. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except TypeError:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
        if hasattr(inputs, "items"):
            model_inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
                if hasattr(value, "to")
            }
        else:
            model_inputs = {"input_ids": inputs.to(self.device)}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        input_length = model_inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(
            generated[0][input_length:],
            skip_special_tokens=True,
        ).strip()

OPERATOR_NAMES = (
    "RUN_PROFILE",
    "EVALUATE_CANDIDATE_POINTS",
    "LOCAL_REFINE",
    "RUN_DE_GLOBAL",
    "SHRINK_BOUNDS",
    "EXPAND_BOUNDS",
    "RESTART_SEARCH",
    "REFINE_CURRENT_MODEL",
    "CHECK_IDENTIFIABILITY",
    "STOP_CONVERGED",
    "STOP_UNIDENTIFIABLE",
)

EXPENSIVE_OPERATOR_NAMES = {"RUN_DE_GLOBAL", "EXPAND_BOUNDS", "RESTART_SEARCH"}

OPERATOR_DESCRIPTIONS = {
    "EVALUATE_CANDIDATE_POINTS": "Evaluate a small set of explicit bounded parameter points, then seed local search.",
    "LOCAL_REFINE": "Run a cheap local global-search refinement around the best candidate point.",
    "RUN_DE_GLOBAL": "Run broad differential-evolution search over compatible models.",
    "SHRINK_BOUNDS": "Refine with profile likelihood near the current estimate.",
    "EXPAND_BOUNDS": "Expand parameter bounds and run global search once.",
    "RESTART_SEARCH": "Restart differential evolution with a new deterministic seed.",
    "REFINE_CURRENT_MODEL": "Keep the current verified model and run a cheap profile refinement.",
    "CHECK_IDENTIFIABILITY": "Compute a finite-difference log-output Jacobian, singular values, condition number, and a safe parameter block.",
    "STOP_CONVERGED": "Stop because fit and identifiability are sufficient.",
    "STOP_UNIDENTIFIABLE": "Stop because more computation is not justified or identifiable.",
}


def _allowed_operators_for_state(
    state: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    model_gap_threshold: float,
    convergence_loss: float,
    max_forward_predict_calls: int,
    force_easy_stop: bool = True,
) -> set[str]:
    """Apply non-learned identifiability, repetition, and budget constraints."""
    rows = [item for item in state.get("per_nuclide", []) if item.get("status") == "ok"]
    if not rows or any(
        int(item.get("uncensored_observation_count", 0)) == 0 for item in rows
    ):
        return {"STOP_UNIDENTIFIABLE"}

    allowed = set(OPERATOR_NAMES) - {"RUN_PROFILE", "STOP_CONVERGED"}
    diagnostics = [
        item.get("identifiability") for item in rows
        if isinstance(item.get("identifiability"), dict)
    ]
    allowed.discard("CHECK_IDENTIFIABILITY")
    score = float(state.get("objective_score", float("inf")))
    ambiguous = any(
        item.get("relative_model_gap") is not None
        and float(item["relative_model_gap"]) <= model_gap_threshold
        for item in rows
    )
    boundary_hit = any(
        item.get("intake_boundary_hit") or item.get("time_boundary_hit")
        for item in rows
    )
    if (
        force_easy_stop
        and score <= convergence_loss
        and not ambiguous
        and not boundary_hit
    ):
        return {"STOP_CONVERGED"}
    if score <= convergence_loss:
        allowed.add("STOP_CONVERGED")
    if ambiguous:
        allowed.discard("REFINE_CURRENT_MODEL")
        allowed.discard("SHRINK_BOUNDS")

    used = {str(item.get("plan", {}).get("operator", "")) for item in history}
    if used & EXPENSIVE_OPERATOR_NAMES:
        return {
            "STOP_CONVERGED" if score <= convergence_loss else "STOP_UNIDENTIFIABLE"
        }

    spent_calls = sum(
        int(item.get("tool_result", {}).get("total_forward_predict_calls", 0))
        for item in history
    )
    estimated_calls = max(1, int(state.get("forward_predict_calls", 0)))
    if spent_calls >= max_forward_predict_calls:
        return {
            "STOP_CONVERGED" if score <= convergence_loss else "STOP_UNIDENTIFIABLE"
        }
    if spent_calls + estimated_calls > max_forward_predict_calls:
        allowed -= EXPENSIVE_OPERATOR_NAMES
    allowed -= {name for name in used if name not in {"RUN_PROFILE"}}
    if not allowed:
        allowed = {
            "STOP_CONVERGED" if score <= convergence_loss else "STOP_UNIDENTIFIABLE"
        }
    return allowed


class AdaptiveOperatorPlanner:
    """Deterministic operator policy used before training MLP/using an LLM.

    It deliberately makes decisions from the structured numerical feedback,
    so its traces can become supervised/RL data for a learned policy later.
    The numerical executor remains unchanged and validates all bounds/models.
    """

    def __init__(
        self,
        knowledge_base: StructuredKnowledgeBase,
        model_gap_threshold: float = 0.10,
        convergence_loss: float = 0.01,
        minimum_relative_improvement: float = 0.01,
        stagnation_rounds: int = 2,
        max_forward_predict_calls: int = 500,
    ):
        self.knowledge_base = knowledge_base
        self.model_gap_threshold = float(model_gap_threshold)
        self.convergence_loss = float(convergence_loss)
        self.minimum_relative_improvement = float(minimum_relative_improvement)
        self.stagnation_rounds = max(1, int(stagnation_rounds))
        self.max_forward_predict_calls = max(1, int(max_forward_predict_calls))

    @staticmethod
    def _near_boundary(value: float, bounds: Sequence[float], fraction: float = 0.05) -> bool:
        low, high = float(bounds[0]), float(bounds[1])
        width = max(high - low, 1e-12)
        return value <= low + fraction * width or value >= high - fraction * width

    def _build_operator_plan(
        self,
        case: CaseInput,
        history: Sequence[Dict[str, Any]],
        operator: str,
        reason: str,
        planner_info: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Translate a high-level operator into a validated numerical plan."""
        if operator.startswith("STOP_"):
            if planner_info.get("raw_llm_operator") is not None:
                planner_info["executed_operator"] = operator
                planner_info["raw_executed_match"] = (
                    str(planner_info["raw_llm_operator"]) == operator
                )
            return {
                "actions": [],
                "operator": operator,
                "stop": True,
                "operator_reason": reason,
                "reasoning_summary": f"{operator}: {reason}",
            }, planner_info

        base = heuristic_plan(case, self.knowledge_base, history)
        proposal_points = planner_info.get("candidate_points", [])
        proposal_models = planner_info.get("model_ids_by_nuclide", {})
        proposal_parameters = planner_info.get("parameter_names")
        proposal_bounds = planner_info.get("parameter_bounds")
        if isinstance(proposal_bounds, dict):
            for action in base["actions"]:
                action["parameter_bounds"].update(proposal_bounds)
        candidate_capable_operators = {
            "EVALUATE_CANDIDATE_POINTS", "LOCAL_REFINE", "RUN_DE_GLOBAL",
            "RESTART_SEARCH",
        }
        if isinstance(proposal_points, list) and operator in candidate_capable_operators:
            by_nuclide = {}
            for point in proposal_points:
                if isinstance(point, dict):
                    nuclide_name = str(point.get("nuclide", ""))
                    clean_point = {
                        key: value for key, value in point.items() if key != "nuclide"
                    }
                    by_nuclide.setdefault(nuclide_name, []).append(clean_point)
            for action in base["actions"]:
                action["candidate_points"] = by_nuclide.get(action["nuclide"], [])
        if isinstance(proposal_models, dict):
            for action in base["actions"]:
                requested = proposal_models.get(action["nuclide"], [])
                if isinstance(requested, str):
                    requested = [requested]
                compatible = set(self.knowledge_base.compatible_model_ids(
                    next(item for item in case.nuclides if item.nuclide == action["nuclide"])
                ))
                selected = [str(model_id) for model_id in requested if str(model_id) in compatible]
                if selected:
                    action["model_ids"] = list(dict.fromkeys(selected))
        previous = history[-1] if history else None
        spent_forward_calls = sum(
            int(item.get("tool_result", {}).get("total_forward_predict_calls", 0))
            for item in history
        )
        remaining_forward_calls = max(
            1, self.max_forward_predict_calls - spent_forward_calls
        )
        per_nuclide_budget = max(
            20, remaining_forward_calls // max(1, len(case.nuclides))
        )
        case_by_nuclide = {item.nuclide: item for item in case.nuclides}
        for action in base["actions"]:
            action["forward_predict_budget"] = per_nuclide_budget
            nuclide = action["nuclide"]
            nuclide_case = case_by_nuclide[nuclide]
            prior = next(
                (
                    item
                    for item in (previous or {}).get("best_by_nuclide", [])
                    if item.get("nuclide") == nuclide
                ),
                None,
            )
            recommended_parameters = list(PARAMETER_BLOCKS["BASIC"])
            if prior and isinstance(prior.get("identifiability"), dict):
                recommended_parameters = list(
                    prior["identifiability"].get(
                        "recommended_parameter_names", recommended_parameters
                    )
                )
            action["recommended_parameter_names"] = recommended_parameters
            if proposal_parameters:
                requested = [
                    str(name) for name in proposal_parameters
                    if str(name) in recommended_parameters
                ]
                for required_name in reversed(PARAMETER_BLOCKS["BASIC"]):
                    if required_name not in requested:
                        requested.insert(0, required_name)
                action["parameter_names"] = list(dict.fromkeys(requested))
            if operator == "RUN_DE_GLOBAL":
                action["optimizer"] = "differential_evolution"
                action["maxiter"] = min(80, max(30, int(action.get("maxiter", 60))))
                action["model_ids"] = self.knowledge_base.compatible_model_ids(
                    nuclide_case
                )
            elif operator == "EVALUATE_CANDIDATE_POINTS":
                action["optimizer"] = "candidate_points"
                action["maxiter"] = 1
                if prior:
                    prior_values = prior.get("parameter_estimates", {})
                    incumbent_point = {
                        name: prior_values.get(
                            name,
                            (
                                math.log10(max(float(prior.get("intake_bq_estimate", 1e3)), 1e-12))
                                if name == "log10_intake"
                                else float(prior.get("intake_time_d_estimate", 0.0))
                            ),
                        )
                        for name in action["parameter_names"]
                    }
                    existing_keys = {
                        tuple(round(float(point.get(name, float("nan"))), 8) for name in action["parameter_names"])
                        for point in action.get("candidate_points", [])
                        if isinstance(point, dict)
                        and all(
                            name in point and _finite_float(point.get(name)) is not None
                            for name in action["parameter_names"]
                        )
                    }
                    incumbent_key = tuple(
                        round(float(incumbent_point[name]), 8)
                        for name in action["parameter_names"]
                    )
                    novel_keys = {
                        key for key in existing_keys if key != incumbent_key
                    }
                    action["novel_candidate_count"] = len(novel_keys)
                    if incumbent_key not in existing_keys:
                        action.setdefault("candidate_points", []).append(incumbent_point)
                    action["incumbent_objective"] = prior.get("weighted_log_mse")
            elif operator == "LOCAL_REFINE":
                action["optimizer"] = "differential_evolution_then_lbfgsb"
                action["maxiter"] = 20
                action["popsize"] = 5
                if prior:
                    estimate = math.log10(max(float(prior.get("intake_bq_estimate", 1e3)), 1e-12))
                    t_estimate = float(prior.get("intake_time_d_estimate", 0.0))
                    earliest = min(item.time_d for item in nuclide_case.observations)
                    time_lower, _ = intake_time_bounds_for_case(case, earliest)
                    action["bounds"] = {
                        "log10_intake": [max(0.0, estimate - 0.5), min(12.0, estimate + 0.5)],
                        "intake_time_d": [max(time_lower, t_estimate - 0.5), min(earliest, t_estimate + 0.5)],
                    }
            elif operator == "RESTART_SEARCH":
                action["optimizer"] = "differential_evolution"
                action["maxiter"] = min(70, max(25, int(action.get("maxiter", 60))))
            elif operator == "SHRINK_BOUNDS" and prior:
                estimate = math.log10(
                    max(float(prior.get("intake_bq_estimate", 1e3)), 1e-12)
                )
                t_estimate = float(prior.get("intake_time_d_estimate", 0.0))
                earliest = min(item.time_d for item in nuclide_case.observations)
                time_lower, _ = intake_time_bounds_for_case(case, earliest)
                action["model_ids"] = [str(prior.get("model_id"))]
                action["bounds"] = {
                    "log10_intake": [
                        max(0.0, estimate - 0.75),
                        min(12.0, estimate + 0.75),
                    ],
                    "intake_time_d": [
                        max(time_lower, t_estimate - 1.0),
                        min(earliest, t_estimate + 1.0),
                    ],
                }
            elif operator == "EXPAND_BOUNDS":
                time_lower, _ = intake_time_bounds_for_case(
                    case, min(item.time_d for item in nuclide_case.observations)
                )
                action["bounds"] = {
                    "log10_intake": [0.0, 12.0],
                    "intake_time_d": [
                        time_lower,
                        min(item.time_d for item in nuclide_case.observations),
                    ],
                }
                action["optimizer"] = "differential_evolution"
            elif operator == "REFINE_CURRENT_MODEL" and prior:
                action["model_ids"] = [str(prior.get("model_id"))]
                action["optimizer"] = "profile_likelihood"
            elif operator == "CHECK_IDENTIFIABILITY":
                action["optimizer"] = "candidate_points"
                action["parameter_names"] = list(PARAMETER_BLOCKS["BASIC"])
                if prior:
                    prior_values = prior.get("parameter_estimates", {})
                    action["candidate_points"] = [{
                        "log10_intake": prior_values.get(
                            "log10_intake",
                            math.log10(max(float(prior.get("intake_bq_estimate", 1e3)), 1e-12)),
                        ),
                        "intake_time_d": prior_values.get(
                            "intake_time_d", prior.get("intake_time_d_estimate", 0.0)
                        ),
                    }]
                action["run_identifiability_check"] = True
                action["count_as_candidate_points"] = False
        if operator == "EVALUATE_CANDIDATE_POINTS" and not any(
            int(action.get("novel_candidate_count", 0)) > 0
            for action in base["actions"]
        ):
            operator = "CHECK_IDENTIFIABILITY"
            reason = (
                "All LLM candidate points duplicated the incumbent; run the "
                "Jacobian/SVD diagnostic instead."
            )
            for action in base["actions"]:
                action["optimizer"] = "candidate_points"
                action["parameter_names"] = list(PARAMETER_BLOCKS["BASIC"])
                prior = next(
                    (
                        item for item in (previous or {}).get("best_by_nuclide", [])
                        if item.get("nuclide") == action["nuclide"]
                    ),
                    None,
                )
                prior_values = prior.get("parameter_estimates", {}) if prior else {}
                action["candidate_points"] = [{
                    "log10_intake": prior_values.get(
                        "log10_intake",
                        math.log10(max(float((prior or {}).get("intake_bq_estimate", 1e3)), 1e-12)),
                    ),
                    "intake_time_d": prior_values.get(
                        "intake_time_d", (prior or {}).get("intake_time_d_estimate", 0.0)
                    ),
                }]
                action["run_identifiability_check"] = True
                action["count_as_candidate_points"] = False
                action["identifiability_parameter_names"] = list(PARAMETER_BLOCKS["BASIC"])
            planner_info.update({
                "operator": operator,
                "reason": reason,
                "candidate_rejection": "duplicate_incumbent",
            })
        base["operator"] = operator
        base["operator_reason"] = reason
        base["reasoning_summary"] = f"{operator}: {reason}"
        if planner_info.get("raw_llm_operator") is not None:
            planner_info["executed_operator"] = operator
            planner_info["raw_executed_match"] = (
                str(planner_info["raw_llm_operator"]) == operator
            )
        return base, planner_info

    def propose(
        self,
        case: CaseInput,
        history: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        operator = "RUN_PROFILE"
        reason = "Use profile likelihood as the low-cost initial probe."
        previous = history[-1] if history else None
        stop = False
        if previous:
            tool = previous.get("tool_result", {})
            score = float(tool.get("objective_score", float("inf")))
            stagnant = False
            relative_improvement = None
            if len(history) >= 2:
                prior_score = float(history[-2].get("tool_result", {}).get("objective_score", float("inf")))
                relative_improvement = (prior_score - score) / max(abs(prior_score), 1e-12)
            if len(history) >= self.stagnation_rounds:
                recent = [
                    float(item.get("tool_result", {}).get("objective_score", float("inf")))
                    for item in history[-self.stagnation_rounds:]
                ]
                stagnant = max(recent) - min(recent) < 1e-6
            boundary_hit = False
            ambiguous = False
            insufficient_observations = False
            diagnostic_available = False
            for item in tool.get("per_nuclide", []):
                if item.get("status") != "ok":
                    continue
                best = item.get("best", {})
                if isinstance(best.get("identifiability"), dict):
                    diagnostic_available = True
                predictions = best.get("predictions", [])
                uncensored_count = sum(
                    1 for prediction in predictions
                    if prediction.get("relative_error") is not None
                )
                if uncensored_count == 0:
                    insufficient_observations = True
                bounds = item.get("search_bounds", {})
                log_intake = math.log10(max(float(best.get("intake_bq_estimate", 1e-12)), 1e-12))
                if bounds and self._near_boundary(
                    log_intake,
                    bounds.get("log10_intake", [0, 8]),
                ):
                    boundary_hit = True
                if bounds and self._near_boundary(
                    float(best.get("intake_time_d_estimate", 0.0)),
                    bounds.get("intake_time_d", [0, 1]),
                ):
                    boundary_hit = True
                ranked = item.get("ranked_candidates", [])
                if len(ranked) >= 2:
                    first = float(ranked[0].get("weighted_log_mse", float("inf")))
                    second = float(ranked[1].get("weighted_log_mse", float("inf")))
                    relative_gap = (second - first) / max(abs(first), 1e-12)
                    if relative_gap <= self.model_gap_threshold:
                        ambiguous = True
            prior_operator = str(previous.get("plan", {}).get("operator", ""))
            spent_calls = sum(
                int(item.get("tool_result", {}).get("total_forward_predict_calls", 0))
                for item in history
            )
            estimated_next_calls = int(tool.get("total_forward_predict_calls", 0))
            if insufficient_observations:
                operator = "STOP_UNIDENTIFIABLE"
                reason = "No uncensored observation is available for parameter identification."
                stop = True
            elif score <= self.convergence_loss and not ambiguous and not boundary_hit:
                operator = "STOP_CONVERGED"
                reason = "The loss is below tolerance and the best model is separated."
                stop = True
            elif prior_operator == "RUN_DE_GLOBAL" and ambiguous and (
                relative_improvement is None
                or relative_improvement < self.minimum_relative_improvement
            ):
                operator = "STOP_UNIDENTIFIABLE"
                reason = "Global search did not resolve the model ambiguity efficiently."
                stop = True
            elif prior_operator in {"RUN_DE_GLOBAL", "EXPAND_BOUNDS", "RESTART_SEARCH"}:
                operator = "STOP_UNIDENTIFIABLE"
                reason = "One expensive recovery operator was insufficient; stop within budget."
                stop = True
            elif spent_calls + estimated_next_calls > self.max_forward_predict_calls:
                operator = "STOP_UNIDENTIFIABLE"
                reason = "The estimated next operator would exceed the numerical budget."
                stop = True
            elif not diagnostic_available:
                operator = "CHECK_IDENTIFIABILITY"
                reason = "Run a low-cost Jacobian/SVD probe before enlarging the parameter block."
            elif len(history) >= 2 and relative_improvement is not None and (
                relative_improvement < self.minimum_relative_improvement
            ):
                operator = "STOP_UNIDENTIFIABLE"
                reason = "The last operator produced too little improvement for its cost."
                stop = True
            elif boundary_hit:
                operator = "EXPAND_BOUNDS"
                reason = "The previous optimum is close to a search boundary."
            elif ambiguous:
                operator = "RUN_DE_GLOBAL"
                reason = "Candidate models remain close, so global exploration is needed."
            elif stagnant:
                operator = "RESTART_SEARCH"
                reason = "The objective has stagnated across recent rounds."
            else:
                operator = "SHRINK_BOUNDS"
                reason = "The previous round found a stable basin; refine around it."

        return self._build_operator_plan(
            case,
            history,
            operator,
            reason,
            {
                "source": "adaptive_rule",
                "operator": operator,
                "reason": reason,
                "raw_output": None,
            },
        )


def _operator_selection_prompt(
    case: CaseInput,
    knowledge_base: StructuredKnowledgeBase,
    state: Dict[str, Any],
    allowed_operators: Sequence[str],
    history: Sequence[Dict[str, Any]],
) -> str:
    operator_library = {
        name: OPERATOR_DESCRIPTIONS[name] for name in allowed_operators
    }
    compact_history = [{
        "round": item.get("round"),
        "operator": item.get("plan", {}).get("operator"),
        "objective_score": item.get("tool_result", {}).get("objective_score"),
        "forward_predict_calls": item.get("tool_result", {}).get(
            "total_forward_predict_calls"
        ),
        "objective_improvement": item.get("operator_trace", {}).get(
            "objective_improvement"
        ),
        "gain_per_forward_predict": item.get("operator_trace", {}).get(
            "gain_per_forward_predict"
        ),
    } for item in history]
    visible_case = {
        "case_id": case.case_id,
        "nuclides": [{
            "nuclide": item.nuclide,
            "candidate_model_ids": item.candidate_model_ids,
            "observation_count": len(item.observations),
            "observation_types": sorted({obs.type for obs in item.observations}),
            "censored_count": sum(obs.is_censored for obs in item.observations),
        } for item in case.nuclides],
    }
    example_nuclide = case.nuclides[0].nuclide if case.nuclides else "Cs-137"
    example_models = knowledge_base.compatible_model_ids(case.nuclides[0]) if case.nuclides else []
    example_model = example_models[0] if example_models else "registered_model_id"
    return f"""
You are the high-level optimization reasoner for an inverse ODE problem.
Profile likelihood has already produced a structured numerical state. Select
exactly one operator from the supplied whitelist. You do not determine final
intake estimates, change medical/ODE parameters, invent observations, or
choose an unregistered model. You may provide bounded candidate points for
the executor, but you do not report them as final estimates. A deterministic
executor will translate the operator into bounded numerical actions.

Decision priorities:
1. Improve fit or resolve model ambiguity only when the likely benefit
   justifies additional ODE evaluations.
2. Prefer a cheap refinement over global search when sufficient.
3. Stop when further computation is unlikely to be cost-effective.
4. Respect that an expensive recovery operator can run at most once.
5. Candidate points must contain exactly the selected parameter_names and
   stay within registered bounds.
6. Candidate points must be informative local alternatives to the incumbent.
   Do not copy the incumbent and do not use generic constants such as 4.0/0.3
   when the state already provides case-specific estimates.
7. ICRP excerpts are reference evidence, not clinical truth for this synthetic
   benchmark. Do not invent parameter values from memory. If you use an ICRP
   excerpt, cite its publication, PDF page, and chunk_id in the reason.

Return only this JSON schema:
{{"operator": "ONE_ALLOWED_OPERATOR", "parameter_names": ["log10_intake", "intake_time_d"], "parameter_bounds": {{}}, "model_ids_by_nuclide": {{"{example_nuclide}": ["{example_model}"]}}, "candidate_points": [{{"nuclide": "{example_nuclide}", "model_id": "{example_model}", "log10_intake": 5.0, "intake_time_d": 0.1}}], "reason": "short technical reason"}}

Allowed operator library:
{json.dumps(operator_library, ensure_ascii=True)}

Visible case summary:
{json.dumps(visible_case, ensure_ascii=True)}

Current optimization state:
{json.dumps(state, ensure_ascii=True)}

Previous numerical rounds:
{json.dumps(compact_history, ensure_ascii=True)}

Bounded registered-model and ICRP context:
{json.dumps(_compact_planner_knowledge(case, knowledge_base), ensure_ascii=True)}
"""


class LLMGatedOperatorPlanner(AdaptiveOperatorPlanner):
    """Binary MLP call gate followed by LLM-based operator reasoning."""

    def __init__(
        self,
        knowledge_base: StructuredKnowledgeBase,
        qwen_client: QwenClient,
        gate_checkpoint_path: Path,
        gate_threshold: float = DEFAULT_LLM_GATE_THRESHOLD,
        allow_fallback: bool = True,
        **kwargs: Any,
    ):
        super().__init__(knowledge_base, **kwargs)
        try:
            from .operator_policy import LLMGatePredictor
        except ImportError:
            from operator_policy import LLMGatePredictor

        self.qwen_client = qwen_client
        self.gate = LLMGatePredictor(Path(gate_checkpoint_path))
        self.gate_threshold = min(1.0, max(0.0, float(gate_threshold)))
        self.allow_fallback = bool(allow_fallback)

    def _safe_stop(
        self,
        case: CaseInput,
        history: Sequence[Dict[str, Any]],
        state: Dict[str, Any],
        reason: str,
        planner_info: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        operator = (
            "STOP_CONVERGED"
            if float(state.get("objective_score", float("inf"))) <= self.convergence_loss
            else "STOP_UNIDENTIFIABLE"
        )
        planner_info.update({"operator": operator, "reason": reason})
        return self._build_operator_plan(
            case, history, operator, reason, planner_info
        )

    def propose(
        self,
        case: CaseInput,
        history: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not history:
            reason = "Mandatory profile probe before deciding whether LLM reasoning is needed."
            return self._build_operator_plan(
                case,
                history,
                "RUN_PROFILE",
                reason,
                {
                    "source": "profile_probe",
                    "operator": "RUN_PROFILE",
                    "reason": reason,
                    "llm_invoked": False,
                },
            )

        state = _optimization_state(history[-1].get("tool_result"))
        if state is None:
            raise RuntimeError("LLM gate requires a numerical state after profile probing.")
        allowed = _allowed_operators_for_state(
            state,
            history,
            self.model_gap_threshold,
            self.convergence_loss,
            self.max_forward_predict_calls,
            force_easy_stop=False,
        )
        if len(allowed) == 1 and next(iter(allowed)).startswith("STOP_"):
            operator = next(iter(allowed))
            reason = "Deterministic safety/identifiability rule requires termination."
            return self._build_operator_plan(
                case,
                history,
                operator,
                reason,
                {
                    "source": "safety_gate",
                    "operator": operator,
                    "reason": reason,
                    "llm_invoked": False,
                    "allowed_operators": sorted(allowed),
                },
            )

        gate_ranking = self.gate.rank(state)
        call_probability = dict(gate_ranking)["CALL_LLM"]
        gate_info = {
            "gate_ranking": [
                {"decision": name, "probability": probability}
                for name, probability in gate_ranking
            ],
            "gate_threshold": self.gate_threshold,
            "allowed_operators": sorted(allowed),
        }
        if call_probability < self.gate_threshold:
            reason = (
                f"MLP gate skipped LLM: CALL_LLM probability "
                f"{call_probability:.4f} < {self.gate_threshold:.4f}."
            )
            return self._safe_stop(
                case,
                history,
                state,
                reason,
                {"source": "mlp_gate", "llm_invoked": False, **gate_info},
            )

        llm_allowed = {
            operator for operator in allowed if not operator.startswith("STOP_")
        }
        if not llm_allowed:
            reason = "The gate requested LLM reasoning, but no executable operator remains."
            return self._safe_stop(
                case,
                history,
                state,
                reason,
                {"source": "safety_gate", "llm_invoked": False, **gate_info},
            )
        gate_info["llm_operator_whitelist"] = sorted(llm_allowed)

        prompt = _operator_selection_prompt(
            case,
            self.knowledge_base,
            state,
            sorted(llm_allowed),
            history,
        )
        raw_text = self.qwen_client.generate(prompt, max_new_tokens=256)
        try:
            proposal = _extract_json_object(raw_text)
            operator = str(proposal.get("operator", ""))
            if operator not in llm_allowed:
                raise ValueError(
                    f"LLM operator {operator!r} is outside whitelist "
                    f"{sorted(llm_allowed)}."
                )
            reason = str(proposal.get("reason", "LLM selected a whitelisted operator."))
            parameter_names = proposal.get("parameter_names", ["log10_intake", "intake_time_d"])
            if isinstance(parameter_names, str):
                parameter_names = [parameter_names]
            parameter_names = [
                str(name) for name in parameter_names
                if str(name) in LLM_ALLOWED_PARAMETER_NAMES
            ]
            if "log10_intake" not in parameter_names:
                parameter_names.insert(0, "log10_intake")
            if "intake_time_d" not in parameter_names:
                parameter_names.insert(1, "intake_time_d")
            points = proposal.get("candidate_points", [])
            if not isinstance(points, list):
                points = []
            model_ids_by_nuclide = proposal.get("model_ids_by_nuclide", {})
            if not isinstance(model_ids_by_nuclide, dict):
                model_ids_by_nuclide = {}
            return self._build_operator_plan(
                case,
                history,
                operator,
                reason,
                {
                    "source": "qwen_operator",
                    "operator": operator,
                    "reason": reason,
                    "raw_llm_proposal": proposal,
                    "raw_llm_operator": operator,
                    "raw_llm_reason": reason,
                    "raw_llm_parameter_names": list(dict.fromkeys(parameter_names)),
                    "raw_llm_parameter_bounds": proposal.get("parameter_bounds", {}),
                    "raw_llm_candidate_points": points[:8],
                    "model_ids_by_nuclide": model_ids_by_nuclide,
                    "parameter_names": list(dict.fromkeys(parameter_names)),
                    "parameter_bounds": proposal.get("parameter_bounds", {}),
                    "candidate_points": points[:8],
                    "llm_invoked": True,
                    "prompt": prompt,
                    "raw_output": raw_text,
                    **gate_info,
                },
            )
        except (ValueError, json.JSONDecodeError, TypeError) as error:
            if not self.allow_fallback:
                raise
            reason = f"Invalid LLM operator output; stopped safely: {error}"
            return self._safe_stop(
                case,
                history,
                state,
                reason,
                {
                    "source": "qwen_operator_invalid",
                    "llm_invoked": True,
                    "prompt": prompt,
                    "raw_output": raw_text,
                    "error": str(error),
                    **gate_info,
                },
            )


def _best_round_summary(tool_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    summaries = []
    for item in tool_result.get("per_nuclide", []):
        if item.get("status") != "ok":
            summaries.append({
                "nuclide": item.get("nuclide"),
                "status": item.get("status"),
            })
            continue
        best = item["best"]
        summaries.append({
            "nuclide": item["nuclide"],
            "status": "ok",
            "model_id": best["model_id"],
            "route": best.get("route"),
            "intake_bq_estimate": best["intake_bq_estimate"],
            "intake_time_d_estimate": best["intake_time_d_estimate"],
            "parameter_estimates": best.get("parameter_estimates", {}),
            "identifiability": best.get("identifiability"),
            "unweighted_log_mse": best["unweighted_log_mse"],
            "weighted_log_mse": best["weighted_log_mse"],
        })
    return summaries


def _optimization_state(tool_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a compact, fixed-schema state for rule, LLM and MLP policies."""
    if not tool_result:
        return None
    per_nuclide = []
    for item in tool_result.get("per_nuclide", []):
        if item.get("status") != "ok":
            per_nuclide.append({
                "nuclide": item.get("nuclide"),
                "status": item.get("status"),
            })
            continue
        best = item.get("best", {})
        predictions = best.get("predictions", [])
        observation_count = len(predictions)
        uncensored_count = sum(
            1 for prediction in predictions
            if prediction.get("relative_error") is not None
        )
        ranked = item.get("ranked_candidates", [])
        model_gap = None
        relative_model_gap = None
        if len(ranked) >= 2:
            first = float(ranked[0].get("weighted_log_mse", float("inf")))
            second = float(ranked[1].get("weighted_log_mse", float("inf")))
            model_gap = second - first
            relative_model_gap = model_gap / max(abs(first), 1e-12)
        bounds = item.get("search_bounds", {})
        identifiability = best.get("identifiability")
        log_intake = math.log10(max(float(best.get("intake_bq_estimate", 1e-12)), 1e-12))
        intake_bounds = bounds.get("log10_intake", [0.0, 8.0])
        time_bounds = bounds.get("intake_time_d", [0.0, 1.0])
        per_nuclide.append({
            "nuclide": item.get("nuclide"),
            "status": "ok",
            "best_model_id": best.get("model_id"),
            "best_loss": best.get("unweighted_log_mse"),
            "weighted_loss": best.get("weighted_log_mse"),
            "log10_intake": log_intake,
            "intake_time_d": best.get("intake_time_d_estimate"),
            "parameter_estimates": best.get("parameter_estimates", {}),
            "candidate_point_count": int(best.get("candidate_point_count", 0)),
            "candidate_improved_incumbent": best.get("candidate_improved_incumbent"),
            "identifiability": identifiability,
            "identifiability_condition_number": (
                identifiability.get("condition_number")
                if isinstance(identifiability, dict) else None
            ),
            "recommended_parameter_names": (
                identifiability.get("recommended_parameter_names", [])
                if isinstance(identifiability, dict) else []
            ),
            "model_gap": model_gap,
            "relative_model_gap": relative_model_gap,
            "observation_count": observation_count,
            "uncensored_observation_count": uncensored_count,
            "censored_fraction": (
                float(observation_count - uncensored_count) / observation_count
                if observation_count else 1.0
            ),
            "intake_boundary_hit": AdaptiveOperatorPlanner._near_boundary(log_intake, intake_bounds),
            "time_boundary_hit": AdaptiveOperatorPlanner._near_boundary(
                float(best.get("intake_time_d_estimate", 0.0)), time_bounds
            ),
        })
    return {
        "objective_score": tool_result.get("objective_score"),
        "fit_score": tool_result.get("fit_score"),
        "failed_nuclide_count": tool_result.get("failed_nuclide_count"),
        "function_evaluations": tool_result.get("total_function_evaluations"),
        "optimizer_iterations": tool_result.get("total_optimizer_iterations"),
        "forward_predict_calls": tool_result.get("total_forward_predict_calls"),
        "numerical_wall_time_s": tool_result.get("numerical_wall_time_s"),
        "per_nuclide": per_nuclide,
    }


def run_outer_loop(
    case: CaseInput,
    knowledge_base: StructuredKnowledgeBase,
    planner: AdaptiveOperatorPlanner,
    rounds: int,
    seed: int,
    dataset_root: Path,
    forward_model_kind: str = "ode",
) -> Dict[str, Any]:
    if forward_model_kind != "ode":
        raise ValueError("Only the compartment ODE backend is supported.")
    forward_model: Any = CompartmentODEForwardModel(knowledge_base.ode_registry)
    run_started = time.perf_counter()
    history: List[Dict[str, Any]] = []
    terminal_decision: Optional[Dict[str, Any]] = None
    planner_decision_count = 0
    total_planner_wall_time_s = 0.0
    for round_index in range(max(1, rounds)):
        round_started = time.perf_counter()
        planner_started = time.perf_counter()
        plan, planner_info = planner.propose(case, history)
        planner_wall_time_s = float(time.perf_counter() - planner_started)
        planner_decision_count += 1
        total_planner_wall_time_s += planner_wall_time_s
        if bool(plan.get("stop", False)):
            terminal_decision = {
                "round": round_index,
                "operator": plan.get("operator"),
                "reason": plan.get("operator_reason"),
                "planner": planner_info,
                "planner_wall_time_s": planner_wall_time_s,
                "state": _optimization_state(
                    history[-1].get("tool_result") if history else None
                ),
            }
            break
        state_before = _optimization_state(
            history[-1].get("tool_result") if history else None
        )
        tool_result = execute_numerical_plan(
            case,
            plan,
            forward_model,
            seed=seed + round_index * 10000,
        )
        state_after = _optimization_state(tool_result)
        previous_score = (
            float(state_before["objective_score"])
            if state_before and state_before.get("objective_score") is not None
            else None
        )
        current_score = float(tool_result.get("objective_score", float("inf")))
        improvement = None if previous_score is None else previous_score - current_score
        relative_improvement = (
            None
            if improvement is None
            else improvement / max(abs(previous_score), 1e-12)
        )
        forward_cost = int(tool_result.get("total_forward_predict_calls", 0))
        operator_trace = {
            "operator": plan.get("operator", "DIRECT_OPTIMIZATION"),
            "reason": plan.get("operator_reason", plan.get("reasoning_summary", "")),
            "state_before": state_before,
            "state_after": state_after,
            "objective_improvement": improvement,
            "relative_objective_improvement": relative_improvement,
            "forward_predict_cost": forward_cost,
            "gain_per_forward_predict": (
                None
                if improvement is None
                else improvement / max(forward_cost, 1)
            ),
        }
        history.append({
            "round": round_index,
            "plan": plan,
            "planner": planner_info,
            "planner_wall_time_s": planner_wall_time_s,
            "tool_result": tool_result,
            "best_by_nuclide": _best_round_summary(tool_result),
            "operator_trace": operator_trace,
            "round_wall_time_s": float(time.perf_counter() - round_started),
        })

    best_round = min(
        history,
        key=lambda item: item["tool_result"].get("objective_score", float("inf")),
    )
    final_tool_result = best_round["tool_result"]
    llm_sources = {
        "qwen", "heuristic_fallback", "qwen_operator", "qwen_operator_invalid"
    }
    llm_call_count = sum(
        1 for item in history
        if item.get("planner", {}).get("llm_invoked") is True
        or item.get("planner", {}).get("source") in llm_sources
    )
    if terminal_decision and (
        terminal_decision.get("planner", {}).get("llm_invoked") is True
        or terminal_decision.get("planner", {}).get("source") in llm_sources
    ):
        llm_call_count += 1
    planner_decisions_with_info = [
        item.get("planner", {}) for item in history
    ]
    if terminal_decision:
        planner_decisions_with_info.append(terminal_decision.get("planner", {}))
    audited_llm_decisions = [
        info for info in planner_decisions_with_info
        if info.get("raw_llm_operator") is not None
    ]
    raw_executed_match_count = sum(
        info.get("raw_executed_match") is True for info in audited_llm_decisions
    )
    raw_executed_mismatch_count = sum(
        info.get("raw_executed_match") is False for info in audited_llm_decisions
    )
    strategy_correction_count = sum(
        bool(info.get("strategy_correction")) for info in audited_llm_decisions
    )
    candidate_rejection_count = sum(
        bool(info.get("candidate_rejection")) for info in audited_llm_decisions
    )
    run_metrics = {
        "outer_rounds": len(history),
        "planner_decisions": planner_decision_count,
        "llm_call_count": llm_call_count,
        "llm_audited_decision_count": len(audited_llm_decisions),
        "llm_raw_executed_match_count": int(raw_executed_match_count),
        "llm_raw_executed_mismatch_count": int(raw_executed_mismatch_count),
        "llm_strategy_correction_count": int(strategy_correction_count),
        "llm_candidate_rejection_count": int(candidate_rejection_count),
        "llm_strategy_correction_rate": (
            float(raw_executed_mismatch_count) / len(audited_llm_decisions)
            if audited_llm_decisions else 0.0
        ),
        "total_planner_wall_time_s": float(total_planner_wall_time_s),
        "total_numerical_wall_time_s": float(sum(item["tool_result"].get("numerical_wall_time_s", 0.0) for item in history)),
        "total_function_evaluations_all_rounds": int(sum(item["tool_result"].get("total_function_evaluations", 0) for item in history)),
        "total_optimizer_iterations_all_rounds": int(sum(item["tool_result"].get("total_optimizer_iterations", 0) for item in history)),
        "total_forward_predict_calls_all_rounds": int(sum(item["tool_result"].get("total_forward_predict_calls", 0) for item in history)),
        "total_candidate_points_all_rounds": int(sum(item["tool_result"].get("total_candidate_points", 0) for item in history)),
        "total_candidate_improvements_all_rounds": int(sum(item["tool_result"].get("total_candidate_improvements", 0) for item in history)),
        "total_wall_time_s": float(time.perf_counter() - run_started),
    }
    return {
        "task": "multi_nuclide_llm_guided_inverse_bioassay",
        "forward_model": {
            "kind": forward_model_kind,
            "model_id": knowledge_base.model_id,
        },
        "case": case.to_dict(),
        "graph_by_nuclide": {
            item.nuclide: build_biokinetic_graph(item)
            for item in case.nuclides
        },
        "knowledge_snapshot": knowledge_base.retrieve_case(case),
        "rounds": history,
        "operator_trace": [item["operator_trace"] for item in history],
        "terminal_decision": terminal_decision,
        "best_round": best_round["round"],
        "final_numerical_result": final_tool_result,
        "run_metrics": run_metrics,
        "dosimetry": {
            "status": "not_implemented",
            "intake_estimation_status": "available",
            "reason": (
                "This prototype estimates intake and intake time from "
                "bioassay observations. Clinical absorbed dose and effective "
                "dose require radionuclide-specific decay energy, tissue "
                "weighting, biokinetic ODE parameters, and dose coefficients."
            ),
        },
    }


def evaluate_intake_estimates(
    result: Dict[str, Any],
    hidden_truth: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare inferred intake/type/time with hidden labels for simulation only.

    This function is deliberately separate from numerical execution so labels
    cannot leak into the planner or objective. It should only be called after
    an experiment has completed.
    """
    rows: List[Dict[str, Any]] = []
    abs_relative_errors: List[float] = []
    log_errors: List[float] = []
    route_matches: List[bool] = []
    model_matches: List[bool] = []
    abs_time_errors: List[float] = []
    parameter_errors: Dict[str, List[float]] = {}
    for item in result.get("final_numerical_result", {}).get("per_nuclide", []):
        nuclide = str(item.get("nuclide"))
        truth = hidden_truth.get(nuclide, {})
        best = item.get("best", {}) if item.get("status") == "ok" else {}
        estimate = _finite_float(best.get("intake_bq_estimate"))
        actual = _finite_float(truth.get("intake_bq"))
        rel_error = None
        log_error = None
        if estimate is not None and actual is not None and actual > 0:
            rel_error = float(estimate / actual - 1.0)
            log_error = float(math.log(estimate) - math.log(actual))
            abs_relative_errors.append(abs(rel_error))
            log_errors.append(abs(log_error))
        model_match = bool(best.get("model_id") == truth.get("model_id"))
        if item.get("status") == "ok":
            model_matches.append(model_match)
        estimated_time = _finite_float(best.get("intake_time_d_estimate"))
        true_time = _finite_float(truth.get("intake_time_d"))
        if estimated_time is not None and true_time is not None:
            abs_time_errors.append(abs(estimated_time - true_time))
        estimates = best.get("parameter_estimates", {})
        for name in parameter_errors:
            estimate_value = _finite_float(estimates.get(name)) if isinstance(estimates, dict) else None
            truth_value = _finite_float(truth.get(name))
            if estimate_value is not None and truth_value is not None:
                parameter_errors[name].append(abs(estimate_value - truth_value))
        rows.append({
            "nuclide": nuclide,
            "status": item.get("status"),
            "true_intake_bq": actual,
            "estimated_intake_bq": estimate,
            "relative_error": rel_error,
            "absolute_log_error": abs(log_error) if log_error is not None else None,
            "true_model_id": truth.get("model_id"),
            "estimated_model_id": best.get("model_id"),
            "model_id_match": model_match,
            "true_intake_time_d": truth.get("intake_time_d"),
            "estimated_intake_time_d": best.get("intake_time_d_estimate"),
            "true_intake_type": truth.get("intake_type"),
        })
    return {
        "rows": rows,
        "mean_absolute_relative_error": (
            float(np.mean(abs_relative_errors)) if abs_relative_errors else None
        ),
        "median_absolute_relative_error": (
            float(np.median(abs_relative_errors)) if abs_relative_errors else None
        ),
        "mean_absolute_log_error": (
            float(np.mean(log_errors)) if log_errors else None
        ),
        "model_id_accuracy": (
            float(np.mean(model_matches)) if model_matches else None
        ),
        "mean_absolute_intake_time_error_d": (
            float(np.mean(abs_time_errors)) if abs_time_errors else None
        ),
        "mean_absolute_parameter_error": {
            name: (float(np.mean(values)) if values else None)
            for name, values in parameter_errors.items()
        },
        "warning": "Metrics are valid only for synthetic hidden labels, not clinical cases.",
    }


def _human_number(value: Any, digits: int = 6) -> str:
    number = _finite_float(value)
    if number is None:
        return str(value)
    if abs(number) >= 10000 or (0 < abs(number) < 0.001):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _human_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_human_log(result: Dict[str, Any]) -> str:
    """Render an audit-oriented, human-readable run log."""

    lines: List[str] = []
    forward_model = result.get("forward_model", {})
    case = result.get("case", {})
    lines.extend([
        "多核素内污染生物测定反演运行日志",
        "=" * 72,
        f"任务: {result.get('task', '')}",
        f"病例ID: {case.get('case_id', '')}",
        f"前向模型: {forward_model.get('kind', '')} / {forward_model.get('model_id', '')}",
        "",
        "一、输入",
        "-" * 72,
        "本次输入是经过结构化的生物测定观测，不包含真实摄入量。",
        f"受试者信息: {_human_json(case.get('subject', {}))}",
        f"暴露信息: {_human_json(case.get('exposure_context', {}))}",
    ])
    for nuclide in case.get("nuclides", []):
        lines.append(f"\n核素 {nuclide.get('nuclide')}:")
        lines.append(
            f"  候选 ICRP 模型: "
            f"{', '.join(nuclide.get('candidate_model_ids', [])) or '未指定'}"
        )
        lines.append(f"  摄入途径候选: {', '.join(nuclide.get('route_candidates', []))}")
        lines.append("  观测数据:")
        for observation in nuclide.get("observations", []):
            uncertainty = ""
            if "sigma_bq" in observation:
                uncertainty = f", sigma={_human_number(observation['sigma_bq'])} Bq"
            lines.append(
                f"    - {observation.get('measurement_id', '')}: "
                f"t={_human_number(observation.get('time_d'))} d, "
                f"type={observation.get('type')}, "
                f"value={_human_number(observation.get('value_bq'))} Bq{uncertainty}"
            )

    lines.extend([
        "",
        "二、图结构与知识检索",
        "-" * 72,
        "图结构把摄入、血液和观测相关区室/测量节点连接起来。",
    ])
    for nuclide, graph in result.get("graph_by_nuclide", {}).items():
        node_names = [
            item.get("node_id")
            for item in graph.get("nodes", [])
            if isinstance(item, dict)
        ]
        edge_names = [
            f"{item.get('source')} -> {item.get('target')}"
            for item in graph.get("edges", [])
            if isinstance(item, dict)
        ]
        lines.append(f"{nuclide}:")
        lines.append(f"  节点: {', '.join(node_names)}")
        lines.append(f"  边: {', '.join(edge_names)}")

    knowledge = result.get("knowledge_snapshot", {})
    lines.append(
        f"知识注册表: {knowledge.get('available_nuclide_count', 0)} 个核素目录；"
        f"当前后端={knowledge.get('forward_model_kind')}"
    )
    for item in knowledge.get("nuclides", []):
        lines.append(
            f"  {item.get('nuclide')}: "
            f"兼容模型={', '.join(item.get('compatible_model_ids', []))}"
        )
        for model in item.get("model_candidates", []):
            if knowledge.get("forward_model_kind") == "ode":
                lines.append(
                    f"    ODE {model.get('model_id')}: "
                    f"compartments={model.get('compartments')}, "
                    f"decay_lambda={_human_number(model.get('physical_decay_constant_per_d'))}/d"
                )
                for transfer in model.get("transfers", []):
                    target = transfer.get("target") or transfer.get("output") or "loss"
                    lines.append(
                        f"      {transfer.get('source')} -> {target}: "
                        f"{_human_number(transfer.get('rate_per_d'))}/d"
                    )

    lines.extend([
        "",
        "三、外层LLM规划与数值工具执行",
        "-" * 72,
        "LLM只提出模型和搜索配置，不直接把摄入量或剂量作为可信答案输出。",
        "数值工具执行: 候选参数 -> ODE前向求解 -> 预测观测 -> 损失 -> 差分进化更新。",
        "ODE状态方程形式: dx/dt = A x(t)，急性摄入在 intake_time_d 时刻注入摄入区室。",
    ])
    for round_item in result.get("rounds", []):
        round_index = round_item.get("round")
        planner = round_item.get("planner", {})
        plan = round_item.get("plan", {})
        tool = round_item.get("tool_result", {})
        lines.extend([
            "",
            f"[第 {round_index} 轮]",
            f"LLM来源: {planner.get('source')}",
            f"LLM摘要: {plan.get('reasoning_summary', '')}",
        ])
        if planner.get("error"):
            lines.append(f"LLM解析/校验信息: {planner.get('error')}")
        raw_output = planner.get("raw_output")
        if raw_output:
            lines.append("LLM原始输出:")
            lines.append(str(raw_output))
        lines.append("经过程序校验后的执行配置:")
        lines.append(_human_json(plan))
        lines.append(
            f"数值工具结果: status={tool.get('status')}, "
            f"objective_score={_human_number(tool.get('objective_score'))}, "
            f"function_evaluations={tool.get('total_function_evaluations')}"
        )
        for item in tool.get("per_nuclide", []):
            lines.append(f"  核素 {item.get('nuclide')}: status={item.get('status')}")
            if item.get("status") != "ok":
                lines.append(f"    error={item.get('error')}")
                continue
            best = item.get("best", {})
            lines.append(
                f"    最优模型={best.get('model_id')}, "
                f"solver={best.get('ode_solver', 'LSODA')}, "
                f"intake={_human_number(best.get('intake_bq_estimate'))} Bq, "
                f"intake_time={_human_number(best.get('intake_time_d_estimate'))} d"
            )
            lines.append(
                f"    loss: unweighted_log_mse={_human_number(best.get('unweighted_log_mse'))}, "
                f"weighted_log_mse={_human_number(best.get('weighted_log_mse'))}"
            )
            lines.append("    观测与预测:")
            for prediction in best.get("predictions", []):
                lines.append(
                    f"      - {prediction.get('measurement_id')}: "
                    f"observed={_human_number(prediction.get('observed_bq'))} Bq, "
                    f"predicted={_human_number(prediction.get('predicted_bq'))} Bq, "
                    f"relative_error={_human_number(prediction.get('relative_error'))}"
                )

    final_result = result.get("final_numerical_result", {})
    lines.extend([
        "",
        "四、最终输出",
        "-" * 72,
        f"选择的最佳轮次: {result.get('best_round')}",
        f"最终状态: {final_result.get('status')}",
        f"最终目标分数: {_human_number(final_result.get('objective_score'))}",
    ])
    for item in final_result.get("per_nuclide", []):
        if item.get("status") != "ok":
            lines.append(f"{item.get('nuclide')}: 反演失败, {item.get('error')}")
            continue
        best = item.get("best", {})
        lines.append(
            f"{item.get('nuclide')}: "
            f"摄入量={_human_number(best.get('intake_bq_estimate'))} Bq, "
            f"摄入时间={_human_number(best.get('intake_time_d_estimate'))} d, "
            f"模型={best.get('model_id')}, "
            f"拟合损失={_human_number(best.get('unweighted_log_mse'))}"
        )
    dosimetry = result.get("dosimetry", {})
    lines.extend([
        "",
        f"剂量学状态: {dosimetry.get('status')}",
        f"剂量学说明: {dosimetry.get('reason')}",
    ])
    offline = result.get("offline_evaluation")
    if offline:
        lines.extend([
            "",
            "五、离线仿真评价",
            "-" * 72,
            "本次运行使用合成观测，因此可以使用隐藏真值进行算法评价；",
            "隐藏真值不是实际测量输入，也不应出现在真实病例报告中。",
            _human_json(offline.get("hidden_ground_truth", {})),
        ])
    lines.extend([
        "",
        "日志结论:",
        "本次运行验证了 LLM 配置规划 -> ODE前向求解 -> 数值反演 -> 反馈汇总闭环。",
        "当前结果是摄入事件重建；器官剂量和承诺有效剂量模块尚未启用。",
    ])
    return "\n".join(lines) + "\n"


def load_case(path: Path) -> CaseInput:
    """Load one normalized observation case from a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return CaseInput.from_dict(json.load(handle))
