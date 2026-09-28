"""Export leakage-safe binary LLM-gate labels from preference JSONL data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def current_gate_decision(row: Dict[str, Any]) -> str | None:
    ranking = row.get("llm", {}).get("gate_ranking")
    if not isinstance(ranking, list) or not ranking:
        return None
    valid = [item for item in ranking if isinstance(item, dict)]
    if not valid:
        return None
    return str(max(valid, key=lambda item: float(item.get("probability", 0.0)))["decision"])


def make_label(
    row: Dict[str, Any],
    margin: float,
    assumed_llm_wall_time_s: float | None,
) -> Dict[str, Any]:
    executable = [
        candidate for candidate in row.get("candidate_operators", [])
        if not str(candidate.get("executed_operator", "")).startswith("STOP_")
    ]
    gamma = float(row.get("reward_config", {}).get("gamma", 0.0))
    def adjusted_call_reward(candidate: Dict[str, Any]) -> float:
        stored = float(candidate.get("reward", float("-inf")))
        if assumed_llm_wall_time_s is None:
            return stored
        current_latency_penalty = float(
            candidate.get("reward_components", {}).get(
                "llm_latency_penalty", 0.0
            )
        )
        return (
            stored + current_latency_penalty
            - gamma * max(0.0, assumed_llm_wall_time_s)
        )

    best_call = max(executable, key=adjusted_call_reward, default=None)
    best_call_reward = (
        adjusted_call_reward(best_call)
        if best_call is not None else float("-inf")
    )
    # SKIP_LLM executes the deterministic fallback operator recorded by the
    # preference builder.  Its reward must be measured by the same fixed ODE
    # executor; using zero would teach the gate to stop rather than to skip
    # the LLM while continuing numerical optimization.
    skip_baseline = row.get("skip_baseline", {})
    skip_reward = float(skip_baseline.get("reward", 0.0))
    label = (
        "CALL_LLM"
        if best_call is not None and best_call_reward > skip_reward + margin
        else "SKIP_LLM"
    )
    current = current_gate_decision(row)
    return {
        "case_id": row.get("case_id"),
        "split": row.get("split"),
        "state_type": row.get("state_type"),
        "state": row.get("state"),
        # train_llm_gate.py intentionally derives the binary label from this
        # operator field, keeping compatibility with existing trajectory data.
        "operator": (
            str(best_call.get("executed_operator"))
            if label == "CALL_LLM" and best_call is not None
            else "STOP_UNIDENTIFIABLE"
        ),
        "gate_label": label,
        "skip_reward": skip_reward,
        "skip_baseline_operator": skip_baseline.get("operator"),
        "skip_baseline_forward_predict_calls": skip_baseline.get(
            "forward_predict_calls"
        ),
        "best_call_reward": (
            best_call_reward if best_call is not None else None
        ),
        "call_advantage": (
            best_call_reward - skip_reward if best_call is not None else None
        ),
        "best_call_operator": (
            best_call.get("executed_operator") if best_call is not None else None
        ),
        "best_call_forward_predict_calls": (
            best_call.get("forward_predict_calls") if best_call is not None else None
        ),
        "llm_wall_time_s": row.get("llm", {}).get("wall_time_s"),
        "assumed_llm_wall_time_s": assumed_llm_wall_time_s,
        "current_gate_decision": current,
        "current_gate_correct": current == label if current is not None else None,
        "label_source": "fixed_state_counterfactual_quality_cost_reward",
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert counterfactual preferences to binary gate labels."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-prefix", type=Path,
        default=ROOT / "temp" / "gate_counterfactual_v3",
    )
    parser.add_argument(
        "--assumed-llm-wall-time-s",
        type=float,
        help=(
            "Replace the stored LLM latency penalty with gamma times this "
            "value; useful for numerical-only oracle datasets."
        ),
    )
    parser.add_argument(
        "--minimum-call-advantage",
        type=float,
        default=0.0,
        help="Require best CALL reward to exceed SKIP reward by this margin.",
    )
    args = parser.parse_args()

    source = resolve(args.input)
    rows = [
        make_label(
            row,
            float(args.minimum_call_advantage),
            args.assumed_llm_wall_time_s,
        )
        for row in read_jsonl(source)
    ]
    prefix = resolve(args.output_prefix)
    outputs: Dict[str, str] = {}
    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row.get("split") == split]
        split_rows[split] = selected
        path = prefix.with_name(f"{prefix.name}_{split}.jsonl")
        write_jsonl(path, selected)
        outputs[split] = str(path)
    all_path = prefix.with_name(f"{prefix.name}_all.jsonl")
    write_jsonl(all_path, rows)
    outputs["all"] = str(all_path)

    labeled_predictions = [
        row for row in rows if row.get("current_gate_correct") is not None
    ]
    summary = {
        "schema_version": "llm_gate_counterfactual_labels_v1",
        "source": str(source),
        "minimum_call_advantage": float(args.minimum_call_advantage),
        "assumed_llm_wall_time_s": args.assumed_llm_wall_time_s,
        "row_count": len(rows),
        "split_counts": {
            split: len(selected) for split, selected in split_rows.items()
        },
        "label_counts": dict(Counter(row["gate_label"] for row in rows)),
        "best_call_operator_counts": dict(Counter(
            row["best_call_operator"] for row in rows
            if row["best_call_operator"] is not None
        )),
        "current_gate_evaluated_count": len(labeled_predictions),
        "current_gate_correct_count": sum(
            row["current_gate_correct"] is True for row in labeled_predictions
        ),
        "current_gate_accuracy": (
            sum(row["current_gate_correct"] is True for row in labeled_predictions)
            / len(labeled_predictions)
            if labeled_predictions else None
        ),
        "false_skip_count": sum(
            row["current_gate_decision"] == "SKIP_LLM"
            and row["gate_label"] == "CALL_LLM"
            for row in rows
        ),
        "false_call_count": sum(
            row["current_gate_decision"] == "CALL_LLM"
            and row["gate_label"] == "SKIP_LLM"
            for row in rows
        ),
        "outputs": outputs,
    }
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
