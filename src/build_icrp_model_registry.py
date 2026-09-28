"""Build the executable ICRP registry from a reviewed JSON configuration.

The JSON file is the domain-review boundary. Medical values, compartment
names, transfer rates, source citations, and measurement mappings live there;
this module only validates the file shape and normalizes it for the numerical
engine. It does not certify that a domain expert entered correct values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "icrp_reviewed_models.json"


def _write(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ICRP configuration root must be a JSON object")
    if payload.get("schema_version") != "icrp_reviewed_model_config_v1":
        raise ValueError("Unsupported ICRP configuration schema_version")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("ICRP configuration must contain a non-empty models list")
    metadata = _required_mapping(payload.get("registry_metadata"), "registry_metadata")
    required_metadata = (
        "meta_verification_status",
        "meta_registry_kind",
        "meta_clinical_use",
        "meta_purpose",
        "meta_measurement_adapter_status",
        "meta_measurement_adapter",
    )
    missing_metadata = [
        key for key in required_metadata if not str(metadata.get(key, "")).strip()
    ]
    if missing_metadata:
        raise ValueError(f"registry_metadata is missing fields: {missing_metadata}")
    for index, model in enumerate(models):
        model = _required_mapping(model, f"models[{index}]")
        for key in (
            "model_id",
            "nuclide",
            "compartments",
            "intake_compartment",
            "physical_decay_constant_per_d",
            "physical_decay_source",
            "transfers",
            "measurements",
            "source",
            "route_scope",
        ):
            if key not in model:
                raise ValueError(f"models[{index}] is missing {key}")
        if not model["model_id"] or not model["nuclide"]:
            raise ValueError(f"models[{index}] requires model_id and nuclide")
        if not isinstance(model["compartments"], list) or not model["compartments"]:
            raise ValueError(f"models[{index}].compartments must be a non-empty list")
        source = _required_mapping(model["source"], f"models[{index}].source")
        for key in ("publication", "pages", "table", "official_url", "sha256"):
            if not str(source.get(key, "")).strip():
                raise ValueError(f"models[{index}].source is missing {key}")
        if not isinstance(model["transfers"], list):
            raise ValueError(f"models[{index}].transfers must be a list")
        if not isinstance(model["measurements"], dict):
            raise ValueError(f"models[{index}].measurements must be an object")
        decay_source = _required_mapping(
            model["physical_decay_source"],
            f"models[{index}].physical_decay_source",
        )
        for key in ("publication", "nuclide", "half_life_d", "official_url"):
            if not str(decay_source.get(key, "")).strip():
                raise ValueError(
                    f"models[{index}].physical_decay_source is missing {key}"
                )
    return payload


def _model_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    shared = dict(config["registry_metadata"])
    rows = []
    for model in config["models"]:
        source = model["source"]
        rows.append(
            {
                "model_id": model["model_id"],
                "nuclide": model["nuclide"],
                "compartments": ";".join(str(item) for item in model["compartments"]),
                "intake_compartment": model["intake_compartment"],
                "physical_decay_constant_per_d": model[
                    "physical_decay_constant_per_d"
                ],
                "meta_verification_status": shared["meta_verification_status"],
                "meta_registry_kind": shared["meta_registry_kind"],
                "meta_parameter_source": source["publication"],
                "meta_source_pages": source["pages"],
                "meta_source_table": source["table"],
                "meta_source_url": source["official_url"],
                "meta_source_sha256": source["sha256"],
                "meta_route_scope": model["route_scope"],
                "meta_physical_decay_source": model["physical_decay_source"]["publication"],
                "meta_physical_decay_nuclide": model["physical_decay_source"]["nuclide"],
                "meta_physical_half_life_d": model["physical_decay_source"]["half_life_d"],
                "meta_physical_decay_url": model["physical_decay_source"]["official_url"],
                "meta_clinical_use": shared["meta_clinical_use"],
                "meta_purpose": shared["meta_purpose"],
                "meta_measurement_adapter_status": shared[
                    "meta_measurement_adapter_status"
                ],
                "meta_measurement_adapter": shared["meta_measurement_adapter"],
            }
        )
    return rows


def _transfer_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in config["models"]:
        for transfer in model["transfers"]:
            rows.append(
                {
                    "model_id": model["model_id"],
                    "source": transfer["source"],
                    "target": transfer.get("target"),
                    "rate_per_d": transfer["rate_per_d"],
                    "output": transfer.get("output"),
                }
            )
    return rows


def _measurement_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in config["models"]:
        for measurement_type, measurement in model["measurements"].items():
            rows.append(
                {
                    "model_id": model["model_id"],
                    "measurement_type": measurement_type,
                    "kind": measurement["kind"],
                    "compartments": ";".join(measurement.get("compartments", [])),
                    "output": measurement.get("output", ""),
                    "window_d": measurement.get("window_d", ""),
                }
            )
    return rows


def build(output_dir: Path, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "models": _write(output_dir / "models.csv", _model_rows(config)),
        "transfers": _write(output_dir / "transfers.csv", _transfer_rows(config)),
        "measurements": _write(
            output_dir / "measurements.csv", _measurement_rows(config)
        ),
        "parameters": _write(
            output_dir / "parameters.csv", config.get("optimization_parameters", [])
        ),
    }
    (output_dir / "source_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.txt").write_text(
        "This registry is generated from config/icrp_reviewed_models.json.\n"
        "The JSON file is the domain-review boundary; the execution code only\n"
        "normalizes its values and does not certify medical correctness.\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "icrp_model_registry_v1",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    print(build(output_dir.resolve(), config_path.resolve()))


if __name__ == "__main__":
    main()
