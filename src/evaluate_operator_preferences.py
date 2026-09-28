"""Evaluate offline counterfactual operator preference trajectories.

The input is JSONL produced by ``build_llm_preference_dataset.py``.  The
report is algorithm-development metadata only; hidden synthetic labels are
not used by this evaluator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    operator_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    chosen_counts: Counter[str] = Counter()
    stop_count = 0
    regrets: List[float] = []
    llm_regrets: List[float] = []
    top1_correct = 0
    llm_count = 0
    budget_violations = []

    for row in rows:
        candidates = [
            item for item in row.get("candidate_operators", [])
            if isinstance(item, dict)
        ]
        if not candidates:
            continue
        chosen = row.get("chosen", {})
        chosen_operator = str(chosen.get("executed_operator", ""))
        chosen_counts[chosen_operator] += 1
        if chosen_operator.startswith("STOP"):
            stop_count += 1
        best_reward = max(float(item.get("reward", 0.0)) for item in candidates)
        chosen_reward = float(chosen.get("reward", 0.0))
        regrets.append(max(0.0, best_reward - chosen_reward))
        budget = int(row.get("reward_config", {}).get(
            "counterfactual_forward_budget", 0
        ))
        for item in candidates:
            operator = str(item.get("executed_operator", ""))
            operator_rows[operator].append(item)
            calls = int(item.get("forward_predict_calls", 0))
            if budget and calls > budget:
                budget_violations.append({
                    "case_id": row.get("case_id"),
                    "operator": operator,
                    "calls": calls,
                    "budget": budget,
                })

        llm = row.get("llm", {})
        raw_operator = llm.get("raw_operator") or llm.get("raw_llm_operator")
        if raw_operator:
            llm_count += 1
            matching = [
                item for item in candidates
                if str(item.get("requested_operator")) == str(raw_operator)
            ]
            if matching:
                predicted_reward = float(matching[0].get("reward", 0.0))
                llm_regrets.append(max(0.0, best_reward - predicted_reward))
                if abs(predicted_reward - best_reward) <= 1e-12:
                    top1_correct += 1

    by_operator: Dict[str, Any] = {}
    for operator, items in sorted(operator_rows.items()):
        rewards = [float(item.get("reward", 0.0)) for item in items]
        calls = [float(item.get("forward_predict_calls", 0)) for item in items]
        by_operator[operator] = {
            "count": len(items),
            "mean_reward": _mean(rewards),
            "mean_forward_predict_calls": _mean(calls),
        }

    # Compute wins in a second pass to keep the per-operator aggregation clear.
    wins: Counter[str] = Counter()
    for row in rows:
        candidates = [
            item for item in row.get("candidate_operators", [])
            if isinstance(item, dict)
        ]
        if candidates:
            winner = max(candidates, key=lambda item: float(item.get("reward", 0.0)))
            wins[str(winner.get("executed_operator", ""))] += 1
    for operator, summary in by_operator.items():
        summary["win_count"] = int(wins.get(operator, 0))
        summary["win_rate"] = float(wins.get(operator, 0) / max(1, len(rows)))

    return {
        "schema_version": "llm_operator_preference_evaluation_v1",
        "record_count": len(rows),
        "records_with_candidates": sum(bool(row.get("candidate_operators")) for row in rows),
        "oracle_chosen_operator_counts": dict(chosen_counts),
        "oracle_stop_rate": float(stop_count / max(1, len(rows))),
        "mean_chosen_reward": _mean(
            float(row.get("chosen", {}).get("reward", 0.0))
            for row in rows
        ),
        "mean_oracle_regret": _mean(regrets),
        "mean_llm_regret": _mean(llm_regrets),
        "llm_records": llm_count,
        "llm_top1_oracle_rate": (
            float(top1_correct / llm_count) if llm_count else None
        ),
        "budget_violation_count": len(budget_violations),
        "budget_violations": budget_violations[:20],
        "by_operator": by_operator,
        "clinical_use": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    report = evaluate(_read_jsonl(input_path.resolve()))
    payload = json.dumps(report, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
