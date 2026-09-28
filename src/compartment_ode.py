"""Linear compartment ODE engine for the fixed ICRP registry.

The ODE topology and transfer coefficients are loaded from a source-audited
registry.  No model is generated at runtime and no LLM output is allowed to
modify a compartment, transfer, or rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp


ODE_MODEL_ID = "linear_compartment_ode_v1"
SUPPORTED_SOLVERS = ("LSODA", "Radau", "BDF", "DOP853", "RK45")


@dataclass(frozen=True)
class Transfer:
    source: str
    target: Optional[str]
    rate_per_d: float
    output: Optional[str] = None


@dataclass
class CompartmentModel:
    model_id: str
    nuclide: str
    compartments: List[str]
    intake_compartment: str
    physical_decay_constant_per_d: float
    transfers: List[Transfer]
    measurements: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.intake_compartment not in self.compartments:
            raise ValueError(f"Unknown intake compartment: {self.intake_compartment}")
        known = set(self.compartments)
        for transfer in self.transfers:
            if transfer.source not in known:
                raise ValueError(f"Unknown transfer source: {transfer.source}")
            if transfer.target is not None and transfer.target not in known:
                raise ValueError(f"Unknown transfer target: {transfer.target}")
            if transfer.rate_per_d < 0:
                raise ValueError("Transfer rates must be non-negative.")

    def observation_types(self) -> List[str]:
        return sorted(self.measurements)

    def _matrices(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Build the immutable transition system transcribed from ICRP."""

        size = len(self.compartments)
        index = {name: position for position, name in enumerate(self.compartments)}
        transition = -float(self.physical_decay_constant_per_d) * np.eye(size)
        output_vectors: Dict[str, np.ndarray] = {}

        for transfer in self.transfers:
            source_index = index[transfer.source]
            rate = float(transfer.rate_per_d)
            transition[source_index, source_index] -= rate
            if transfer.target is not None:
                transition[index[transfer.target], source_index] += rate
            if transfer.output:
                output_vectors.setdefault(transfer.output, np.zeros(size))
                output_vectors[transfer.output][source_index] += rate
        return transition, output_vectors

    def _rhs(self):
        transition, output_vectors = self._matrices()
        output_names = sorted(output_vectors)
        state_size = len(self.compartments)

        def rhs(_time: float, state: np.ndarray) -> np.ndarray:
            compartments = state[:state_size]
            derivative = np.zeros_like(state)
            derivative[:state_size] = transition @ compartments
            for offset, output_name in enumerate(output_names):
                derivative[state_size + offset] = output_vectors[output_name] @ compartments
            return derivative

        return rhs, output_names

    def simulate(
        self,
        intake_bq: float,
        intake_time_d: float,
        query_times_d: Sequence[float],
        solver: str = "LSODA",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> Dict[float, np.ndarray]:
        """Solve the initial-value problem and return states at query times."""

        if solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"Unsupported ODE solver: {solver}")
        query_times = sorted({float(item) for item in query_times_d})
        if not query_times:
            return {}
        start_time = min(0.0, float(intake_time_d), query_times[0])
        end_time = max(query_times[-1], float(intake_time_d))
        rhs, output_names = self._rhs()
        state_size = len(self.compartments)
        total_size = state_size + len(output_names)
        state = np.zeros(total_size, dtype=float)
        intake_index = self.compartments.index(self.intake_compartment)

        def integrate(
            segment_start: float,
            segment_end: float,
            initial_state: np.ndarray,
            targets: Sequence[float],
        ) -> Dict[float, np.ndarray]:
            if segment_end <= segment_start:
                return {}
            targets = sorted({float(item) for item in targets if item > segment_start})
            if not targets:
                return {}
            solution = solve_ivp(
                rhs,
                (segment_start, segment_end),
                initial_state,
                method=solver,
                t_eval=targets,
                rtol=rtol,
                atol=atol,
            )
            if not solution.success:
                raise RuntimeError(
                    f"{self.model_id} failed with {solver}: {solution.message}"
                )
            return {
                float(time): solution.y[:, column].copy()
                for column, time in enumerate(solution.t)
            }

        pre_targets = [item for item in query_times if item <= intake_time_d]
        post_targets = [item for item in query_times if item >= intake_time_d]
        states: Dict[float, np.ndarray] = {}
        if intake_time_d > start_time:
            pre_states = integrate(start_time, intake_time_d, state, pre_targets)
            states.update(pre_states)
            if pre_states:
                state = pre_states[max(pre_states)]
        state[intake_index] += float(intake_bq)
        if intake_time_d in query_times:
            states[float(intake_time_d)] = state.copy()
        post_states = integrate(intake_time_d, end_time, state, post_targets)
        states.update(post_states)
        return states

    def predict(
        self,
        intake_bq: float,
        intake_time_d: float,
        observations: Sequence[Dict[str, Any]],
        solver: str = "LSODA",
        parameters: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        """Map ODE states and cumulative fluxes to bioassay observations."""

        if not observations:
            return []
        query_times = []
        for observation in observations:
            time_d = float(observation["time_d"])
            query_times.append(time_d)
            if observation["type"] in self.measurements:
                spec = self.measurements[observation["type"]]
                if spec.get("kind") == "interval_flux":
                    query_times.append(time_d - float(spec.get("window_d", 1.0)))
        parameters = parameters or {}
        protected = sorted(
            set(parameters) - {"log10_intake", "intake_time_d"}
        )
        if protected:
            raise ValueError(
                "ICRP compartment topology, transfer rates, and measurement "
                f"mapping are immutable; rejected parameters: {protected}"
            )
        states = self.simulate(
            intake_bq=float(intake_bq),
            intake_time_d=float(intake_time_d),
            query_times_d=query_times,
            solver=solver,
        )
        _, output_names = self._rhs()
        output_index = {
            name: len(self.compartments) + index
            for index, name in enumerate(output_names)
        }
        compartment_index = {
            name: index for index, name in enumerate(self.compartments)
        }

        predictions = []
        for observation in observations:
            time_d = float(observation["time_d"])
            observation_type = str(observation["type"])
            if observation_type not in self.measurements:
                raise KeyError(
                    f"{observation_type} is not supported by model {self.model_id}"
                )
            spec = self.measurements[observation_type]
            kind = spec.get("kind")
            if kind == "sum":
                value = sum(
                    states.get(time_d, np.zeros(len(self.compartments)))[
                        compartment_index[name]
                    ]
                    for name in spec["compartments"]
                )
            elif kind == "interval_flux":
                window_d = float(spec.get("window_d", 1.0))
                start_time = time_d - window_d
                end_state = states.get(time_d, np.zeros(len(self.compartments) + len(output_names)))
                start_state = states.get(
                    start_time,
                    np.zeros(len(self.compartments) + len(output_names)),
                )
                flux_name = str(spec["output"])
                value = end_state[output_index[flux_name]] - start_state[output_index[flux_name]]
            else:
                raise ValueError(f"Unknown measurement kind: {kind}")
            predictions.append(max(0.0, float(value)))
        return predictions


class ODEModelRegistry:
    def __init__(self, models: Optional[Iterable[CompartmentModel]] = None):
        self.models: Dict[Tuple[str, str], CompartmentModel] = {}
        for model in models or ():
            self.models[(model.nuclide, model.model_id)] = model

    @classmethod
    def from_csv_dir(
        cls,
        directory: Path,
        *,
        require_verified: bool = True,
    ) -> "ODEModelRegistry":
        """Load a versioned model registry from normalized CSV tables.

        The loader is intentionally small and deterministic: CSV files contain
        topology and measurement mappings only; no code is executed from them.
        Required files are ``models.csv``, ``transfers.csv`` and
        ``measurements.csv``.  This makes the algorithm benchmark reproducible
        while allowing domain experts to maintain parameter tables separately.
        """
        directory = Path(directory)
        paths = {name: directory / name for name in ("models.csv", "transfers.csv", "measurements.csv")}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing ODE registry CSV files: " + ", ".join(missing))

        def rows(path: Path) -> List[Dict[str, str]]:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]

        model_rows = rows(paths["models.csv"])
        if not model_rows:
            raise ValueError(f"ODE registry is empty: {directory}")
        transfer_rows = rows(paths["transfers.csv"])
        measurement_rows = rows(paths["measurements.csv"])
        models: List[CompartmentModel] = []
        for model_row in model_rows:
            model_id = str(model_row["model_id"]).strip()
            nuclide = str(model_row["nuclide"]).strip()
            compartments = [x.strip() for x in str(model_row["compartments"]).split(";") if x.strip()]
            transfers = []
            for item in transfer_rows:
                if str(item.get("model_id", "")).strip() != model_id:
                    continue
                target = str(item.get("target", "")).strip() or None
                output = str(item.get("output", "")).strip() or None
                transfers.append(Transfer(
                    source=str(item["source"]).strip(),
                    target=target,
                    rate_per_d=float(item["rate_per_d"]),
                    output=output,
                ))
            measurements: Dict[str, Dict[str, Any]] = {}
            for item in measurement_rows:
                if str(item.get("model_id", "")).strip() != model_id:
                    continue
                kind = str(item["kind"]).strip()
                spec: Dict[str, Any] = {"kind": kind}
                if str(item.get("compartments", "")).strip():
                    spec["compartments"] = [x.strip() for x in str(item["compartments"]).split(";") if x.strip()]
                if str(item.get("output", "")).strip():
                    spec["output"] = str(item["output"]).strip()
                if str(item.get("window_d", "")).strip():
                    spec["window_d"] = float(item["window_d"])
                measurements[str(item["measurement_type"]).strip()] = spec
            metadata = {
                key: value for key, value in model_row.items()
                if key.startswith("meta_") and str(value).strip()
            }
            if require_verified:
                required = {
                    "meta_verification_status": "verified_icrp_transcription",
                    "meta_registry_kind": "icrp_verified",
                }
                for key, expected in required.items():
                    if metadata.get(key) != expected:
                        raise ValueError(
                            f"Registry model {model_id} is not a verified ICRP model: "
                            f"{key}={metadata.get(key)!r}"
                        )
                for key in (
                    "meta_parameter_source",
                    "meta_source_pages",
                    "meta_source_table",
                    "meta_source_sha256",
                    "meta_route_scope",
                    "meta_measurement_adapter_status",
                ):
                    if not str(metadata.get(key, "")).strip():
                        raise ValueError(
                            f"Registry model {model_id} is missing required provenance field {key}."
                        )
                source_hash = str(metadata.get("meta_source_sha256", ""))
                if len(source_hash) != 64 or any(
                    character not in "0123456789abcdefABCDEF" for character in source_hash
                ):
                    raise ValueError(
                        f"Registry model {model_id} has an invalid source SHA-256."
                    )
                lowered = model_id.lower()
                if "demo" in lowered or "synthetic" in lowered:
                    raise ValueError(
                        f"Synthetic/demo model is forbidden in the formal registry: {model_id}"
                    )
                if (
                    metadata.get("meta_measurement_adapter_status")
                    != "fixed_engineering_projection_not_icrp_dose_coefficient"
                ):
                    raise ValueError(
                        f"Registry model {model_id} has an unaudited measurement adapter."
                    )
            models.append(CompartmentModel(
                model_id=model_id,
                nuclide=nuclide,
                compartments=compartments,
                intake_compartment=str(model_row["intake_compartment"]).strip(),
                physical_decay_constant_per_d=float(model_row["physical_decay_constant_per_d"]),
                transfers=transfers,
                measurements=measurements,
                metadata=metadata,
            ))
        registry = cls(models)
        if require_verified and not registry.models:
            raise ValueError(f"No verified ICRP models loaded from {directory}")
        return registry

    def get(self, nuclide: str, model_id: str) -> CompartmentModel:
        try:
            return self.models[(nuclide, model_id)]
        except KeyError as error:
            raise KeyError(f"No ODE model for {nuclide}/{model_id}") from error

    def compatible_model_ids(
        self,
        nuclide: str,
        observation_types: Iterable[str],
    ) -> List[str]:
        required = set(observation_types)
        result = []
        for (model_nuclide, model_id), model in self.models.items():
            if model_nuclide == nuclide and required.issubset(set(model.observation_types())):
                result.append(model_id)
        return sorted(result)

    def describe(self, nuclide: str, model_id: str) -> Dict[str, Any]:
        model = self.get(nuclide, model_id)
        graph_nodes = [
            {
                "node_id": name,
                "node_type": "compartment",
                "is_intake": name == model.intake_compartment,
            }
            for name in model.compartments
        ]
        graph_edges = [
            {
                "source": item.source,
                "target": item.target,
                "edge_type": item.output or "transfer",
                "rate_per_d": item.rate_per_d,
                "output": item.output,
            }
            for item in model.transfers
        ]
        return {
            "model_id": model.model_id,
            "nuclide": model.nuclide,
            "compartments": model.compartments,
            "intake_compartment": model.intake_compartment,
            "physical_decay_constant_per_d": model.physical_decay_constant_per_d,
            "transfers": [
                {
                    "source": item.source,
                    "target": item.target,
                    "rate_per_d": item.rate_per_d,
                    "output": item.output,
                }
                for item in model.transfers
            ],
            "transfer_count": len(model.transfers),
            "observation_types": model.observation_types(),
            "graph": {
                "nodes": graph_nodes,
                "edges": graph_edges,
                "global_features": {
                    "physical_decay_constant_per_d": model.physical_decay_constant_per_d,
                    "nuclide": model.nuclide,
                },
            },
            "metadata": model.metadata,
        }


class CompartmentODEForwardModel:
    """Forward-model adapter with the same interface as the CSV backend."""

    def __init__(self, registry: Optional[ODEModelRegistry] = None):
        if registry is None or not registry.models:
            raise ValueError("A non-empty verified ICRP ODE registry is required.")
        self.registry = registry

    def supported_observation_types(self, nuclide: str, model_id: str) -> List[str]:
        return self.registry.get(nuclide, model_id).observation_types()

    def predict(
        self,
        nuclide: str,
        model_id: str,
        intake_bq: float,
        intake_time_d: float,
        observations: Sequence[Any],
        solver: str = "LSODA",
        parameters: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        serialised = [
            {"time_d": item.time_d, "type": item.type}
            if hasattr(item, "time_d")
            else item
            for item in observations
        ]
        return self.registry.get(nuclide, model_id).predict(
            intake_bq=intake_bq,
            intake_time_d=intake_time_d,
            observations=serialised,
            solver=solver,
            parameters=parameters,
        )
