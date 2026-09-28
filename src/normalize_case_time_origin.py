"""Normalize a case to a documented first-observation time origin.

This is for cases whose input contains numeric observation times but no
reliable exposure-window origin. The earliest observation endpoint across the
whole case becomes time zero; intake_time_d is then allowed to be negative.
The original numeric time is retained as ``source_time_d`` for auditability.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def normalize_case(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(payload)
    observations: List[Dict[str, Any]] = []
    for nuclide in result.get("nuclides", []):
        for observation in nuclide.get("observations", []):
            if "time_d" not in observation:
                raise ValueError("Every observation must contain numeric time_d.")
            try:
                time_d = float(observation["time_d"])
            except (TypeError, ValueError) as error:
                raise ValueError("Observation time_d must be numeric.") from error
            observations.append({"record": observation, "time_d": time_d})
    if not observations:
        raise ValueError("Case contains no observations.")
    origin = min(item["time_d"] for item in observations)
    for item in observations:
        record = item["record"]
        record["source_time_d"] = record["time_d"]
        record["time_d"] = float(item["time_d"] - origin)
    context = result.setdefault("exposure_context", {})
    context.update({
        "time_reference": "first_observation",
        "observation_time_definition": "days_from_case_first_observation",
        "intake_time_definition": "days_from_case_first_observation_and_unknown",
        "time_origin_shift_d": float(origin),
        "minimum_intake_time_d": -14.0,
        "source_time_note": (
            "Numeric source time was shifted by time_origin_shift_d; "
            "intake_time_d may be negative."
        ),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = json.loads(source.read_text(encoding="utf-8"))
    normalized = normalize_case(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "input": str(source.resolve()),
        "output": str(output.resolve()),
        "time_reference": normalized["exposure_context"]["time_reference"],
        "time_origin_shift_d": normalized["exposure_context"]["time_origin_shift_d"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
