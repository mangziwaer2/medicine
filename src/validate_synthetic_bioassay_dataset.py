"""Validate synthetic ODE datasets and summarize difficulty strata."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from .compartment_ode import ODEModelRegistry
    from .multi_nuclide_llm_optimizer import load_case, validate_case
except ImportError:
    from compartment_ode import ODEModelRegistry
    from multi_nuclide_llm_optimizer import load_case, validate_case


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a synthetic ODE dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--registry-dir", type=Path,
        default=ROOT / "data" / "icrp_model_registry_v1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = resolve(args.dataset_dir)
    registry = ODEModelRegistry.from_csv_dir(resolve(args.registry_dir), require_verified=True)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    labels = [
        json.loads(line)
        for line in (dataset / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels_by_case = {str(row["case_id"]): row for row in labels}
    if len(labels_by_case) != len(labels):
        raise SystemExit("Duplicate case_id found in labels.jsonl")

    files = sorted((dataset / "inputs").glob("*/*.json"))
    ids = [path.stem for path in files]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate case input file stem found across splits")
    missing_labels = sorted(set(ids) - set(labels_by_case))
    extra_labels = sorted(set(labels_by_case) - set(ids))
    if missing_labels or extra_labels:
        raise SystemExit(
            f"Input/label mismatch: missing_labels={missing_labels}, extra_labels={extra_labels}"
        )

    tier_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    split_counts = Counter()
    tier_counts = Counter()
    zero_uncensored_nuclides = 0
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "hidden_truth" in raw or "difficulty_tier" in raw:
            raise SystemExit(f"Hidden label leaked into input: {path}")
        case = load_case(path)
        validate_case(case, ode_registry=registry)
        split = path.parent.name
        label = labels_by_case[case.case_id]
        if str(label.get("split")) != split:
            raise SystemExit(f"Split mismatch for {case.case_id}")
        tier = str(label.get("difficulty_tier", "uniform"))
        split_counts[split] += 1
        tier_counts[tier] += 1
        for nuclide in case.nuclides:
            count = len(nuclide.observations)
            censored = sum(item.is_censored for item in nuclide.observations)
            if censored == count:
                zero_uncensored_nuclides += 1
            tier_stats[tier]["observation_count"].append(float(count))
            tier_stats[tier]["censored_fraction"].append(
                float(censored) / max(1, count)
            )
            tier_stats[tier]["model_candidate_count"].append(
                float(len(nuclide.candidate_model_ids))
            )
            truth = label.get("hidden_truth", {}).get(nuclide.nuclide, {})

    def describe(values: List[float]) -> Dict[str, float | None]:
        return {
            "count": len(values),
            "mean": statistics.mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    payload = {
        "schema_version": "synthetic_dataset_validation_v1",
        "dataset_dir": str(dataset),
        "valid": True,
        "manifest_schema": manifest.get("schema_version"),
        "case_count": len(files),
        "label_count": len(labels),
        "split_counts": dict(split_counts),
        "difficulty_counts": dict(tier_counts),
        "zero_uncensored_nuclide_count": zero_uncensored_nuclides,
        "difficulty_statistics": {
            tier: {
                name: describe(values) for name, values in metrics.items()
            }
            for tier, metrics in sorted(tier_stats.items())
        },
        "hidden_truth_leakage": False,
        "clinical_use": False,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
