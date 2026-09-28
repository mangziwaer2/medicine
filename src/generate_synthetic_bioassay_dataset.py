"""Generate theory-validation cases from fixed verified ICRP models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from .compartment_ode import CompartmentODEForwardModel, ODEModelRegistry
    from .multi_nuclide_llm_optimizer import Observation
except ImportError:
    from compartment_ode import CompartmentODEForwardModel, ODEModelRegistry
    from multi_nuclide_llm_optimizer import Observation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "synthetic_bioassay_icrp_v1"
DEFAULT_REGISTRY_DIR = ROOT / "data" / "icrp_model_registry_v1"
TIMES = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 7.0, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0)
PRIORITY = ("Thyroid-Bioa", "T-body", "24-h Urine", "24-h Feces", "Skelet-Bioa")


def _tier(index: int, args: argparse.Namespace) -> Dict[str, Any]:
    if args.difficulty_mode == "uniform":
        return {"name": "uniform", "noise": args.noise_sigma, "censor": args.censor_probability, "missing": args.missing_probability, "obs_per_type": args.observations_per_type, "candidate_policy": "all"}
    name = ("easy", "medium", "hard")[index % 3]
    if name == "easy":
        return {"name": name, "noise": args.noise_sigma * 0.6, "censor": args.censor_probability * 0.4, "missing": args.missing_probability * 0.4, "obs_per_type": args.observations_per_type + 1, "candidate_policy": "single"}
    if name == "hard":
        return {"name": name, "noise": args.noise_sigma * 1.8, "censor": args.censor_probability * 2.5, "missing": args.missing_probability * 2.5, "obs_per_type": max(2, args.observations_per_type - 1), "candidate_policy": "all"}
    return {"name": name, "noise": args.noise_sigma, "censor": args.censor_probability, "missing": args.missing_probability, "obs_per_type": args.observations_per_type, "candidate_policy": "all"}


def _catalog(registry: ODEModelRegistry) -> Dict[str, List[Dict[str, Any]]]:
    catalog: Dict[str, List[Dict[str, Any]]] = {}
    for (nuclide, model_id), model in sorted(registry.models.items()):
        catalog.setdefault(nuclide, []).append({"model_id": model_id, "observation_types": model.observation_types()})
    return catalog


def _split(index: int, total: int, train_ratio: float, validation_ratio: float) -> str:
    if index < int(total * train_ratio):
        return "train"
    if index < int(total * (train_ratio + validation_ratio)):
        return "validation"
    return "test"


def _case(index: int, rng: np.random.Generator, catalog: Dict[str, List[Dict[str, Any]]], forward: CompartmentODEForwardModel, args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any]]:
    tier = _tier(index, args)
    names = sorted(catalog)
    count = int(rng.integers(args.min_nuclides, min(args.max_nuclides, len(names)) + 1))
    selected = [str(item) for item in rng.choice(names, size=count, replace=False)]
    case_nuclides: List[Dict[str, Any]] = []
    truth: Dict[str, Any] = {}
    for nuclide in selected:
        models = catalog[nuclide]
        candidate_count = 1 if tier["candidate_policy"] == "single" else len(models)
        candidate_rows = [models[int(i)] for i in rng.choice(len(models), size=candidate_count, replace=False)]
        true_model = candidate_rows[int(rng.integers(len(candidate_rows)))]["model_id"]
        common = set(candidate_rows[0]["observation_types"])
        for row in candidate_rows[1:]:
            common.intersection_update(row["observation_types"])
        selected_types: List[tuple[str, List[float]]] = []
        for observation_type in PRIORITY + tuple(sorted(common)):
            if observation_type not in common or any(item[0] == observation_type for item in selected_types):
                continue
            valid_times = []
            for time_d in TIMES:
                if time_d > args.max_observation_horizon_d:
                    continue
                value = forward.predict(nuclide, true_model, 1.0, 0.0, [Observation(time_d, observation_type, 1.0)])[0]
                if math.isfinite(value) and value > 1e-12:
                    valid_times.append(time_d)
            if valid_times:
                if len(valid_times) > tier["obs_per_type"]:
                    indices = np.linspace(0, len(valid_times) - 1, tier["obs_per_type"]).round().astype(int)
                    valid_times = [valid_times[int(i)] for i in np.unique(indices)]
                selected_types.append((observation_type, valid_times))
            if len(selected_types) >= args.observation_types:
                break
        intake = float(10 ** rng.uniform(math.log10(args.min_intake_bq), math.log10(args.max_intake_bq)))
        intake_time = float(rng.uniform(0, args.max_intake_time_d))
        observations: List[Dict[str, Any]] = []
        observation_index = 0
        for observation_type, times in selected_types:
            for elapsed in times:
                if rng.random() < min(0.8, tier["missing"]):
                    continue
                observation_time = intake_time + elapsed
                prediction = forward.predict(nuclide, true_model, intake, intake_time, [Observation(observation_time, observation_type, 1.0)])[0]
                noisy = max(1e-10, prediction * float(rng.lognormal(0, min(0.5, max(0.001, tier["noise"])))) )
                limit = float(10 ** rng.uniform(math.log10(args.detection_limit_min_bq), math.log10(args.detection_limit_max_bq)))
                censored = bool(noisy <= limit or rng.random() < min(0.8, tier["censor"]))
                if censored:
                    noisy = limit
                observation_index += 1
                observations.append({"measurement_id": f"{nuclide}-obs-{observation_index}", "time_d": observation_time, "type": observation_type, "value_bq": noisy, "sigma_bq": max(noisy * tier["noise"], limit * 0.05), "detection_limit_bq": limit, "is_censored": censored, "sample_window_d": 1.0 if observation_type.startswith("24-h") else 0.0})
        if not observations:
            raise RuntimeError(f"No observations generated for {nuclide}")
        case_nuclides.append({"nuclide": nuclide, "candidate_model_ids": [row["model_id"] for row in candidate_rows], "route_candidates": ["acute_systemic_input_to_blood"], "chemical_form": "reference_worker_systemic_model", "observations": observations})
        truth[nuclide] = {"model_id": true_model, "intake_bq": intake, "intake_time_d": intake_time, "intake_type": "acute_single_event", "route": "acute_systemic_input_to_blood"}
    case_id = f"ode-synthetic-{index:06d}"
    case = {"case_id": case_id, "subject": {"age_y": int(rng.integers(18, 76)), "sex": str(rng.choice(["female", "male", "unknown"])), "body_mass_kg": float(rng.uniform(50, 95))}, "exposure_context": {"route_candidates": ["acute_systemic_input_to_blood"], "intake_type_candidates": ["acute_single_event"], "intake_pattern": "acute", "chemical_form": "reference_worker_systemic_model", "time_reference": "exposure_window_origin", "observation_time_definition": "days_from_exposure_window_origin", "intake_time_definition": "days_from_same_origin_and_unknown_to_inverse_solver", "quantity_semantics": "activity entering the systemic blood compartment"}, "nuclides": case_nuclides}
    label = {"case_id": case_id, "hidden_truth": truth, "difficulty_tier": tier["name"], "generation": {"forward_model": "linear_compartment_ode_v1", "registry_policy": "verified_icrp_only", "noise_model": "multiplicative_lognormal_plus_detection_censoring", **tier}}
    return case, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ODE bioassay cases.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--min-nuclides", type=int, default=1)
    parser.add_argument("--max-nuclides", type=int, default=3)
    parser.add_argument("--observation-types", type=int, default=2)
    parser.add_argument("--observations-per-type", type=int, default=3)
    parser.add_argument("--max-observation-horizon-d", type=float, default=365)
    parser.add_argument("--min-intake-bq", type=float, default=1e3)
    parser.add_argument("--max-intake-bq", type=float, default=1e7)
    parser.add_argument("--max-intake-time-d", type=float, default=0.5)
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument("--detection-limit-min-bq", type=float, default=1e-3)
    parser.add_argument("--detection-limit-max-bq", type=float, default=100)
    parser.add_argument("--censor-probability", type=float, default=0.05)
    parser.add_argument("--missing-probability", type=float, default=0.03)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--difficulty-mode", choices=("uniform", "mixed"), default="mixed")
    args = parser.parse_args()
    output = (args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir).resolve()
    registry_dir = (args.registry_dir if args.registry_dir.is_absolute() else ROOT / args.registry_dir).resolve()
    registry = ODEModelRegistry.from_csv_dir(registry_dir, require_verified=True)
    forward = CompartmentODEForwardModel(registry)
    catalog = _catalog(registry)
    rng = np.random.default_rng(args.seed)
    inputs: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    difficulty_counts = {"train": {}, "validation": {}, "test": {}}
    split_case_ids = {"train": [], "validation": [], "test": []}
    for index in range(args.cases):
        case, label = _case(index, rng, catalog, forward, args)
        split = _split(index, args.cases, args.train_ratio, args.validation_ratio)
        label["split"] = split
        inputs.append(case)
        labels.append(label)
        split_counts[split] += 1
        tier_name = str(label["difficulty_tier"])
        difficulty_counts[split][tier_name] = difficulty_counts[split].get(tier_name, 0) + 1
        split_case_ids[split].append(case["case_id"])
        split_dir = output / "inputs" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / f"{case['case_id']}.json").write_text(json.dumps(case, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs.jsonl").write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in inputs), encoding="utf-8")
    (output / "labels.jsonl").write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in labels), encoding="utf-8")
    manifest = {"schema_version": "project_synthetic_bioassay_icrp_v1", "forward_model": "linear_compartment_ode_v1", "registry_policy": "verified_icrp_only", "quantity_semantics": "acute systemic activity input to the ICRP blood/plasma compartment", "cases": args.cases, "split_counts": split_counts, "difficulty_counts": difficulty_counts, "split_case_ids": split_case_ids, "nuclides": sorted(catalog), "usage": {"inversion_input": "inputs/<split>/*.json", "offline_labels": "labels.jsonl only"}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": args.cases, "splits": split_counts}, indent=2))


if __name__ == "__main__":
    main()
