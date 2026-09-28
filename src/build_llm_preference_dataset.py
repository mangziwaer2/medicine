"""Build counterfactual operator preferences for the LLM optimization planner.

Each record fixes one numerical state, evaluates several safe operator branches
from that same state, and stores the best quality/cost trade-off as an oracle
target.  With ``--query-llm`` it also stores the raw Qwen proposal, the audited
executed proposal, and a chosen/rejected pair suitable for later SFT/DPO data
preparation.

The generated data are algorithm-development data, not clinical evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .compartment_ode import CompartmentODEForwardModel, ODEModelRegistry
    from .multi_nuclide_llm_optimizer import (
        DEFAULT_LLM_GATE_PATH,
        DEFAULT_LLM_GATE_THRESHOLD,
        DEFAULT_MODEL_PATH,
        AdaptiveOperatorPlanner,
        LLMGatedOperatorPlanner,
        OPERATOR_DESCRIPTIONS,
        PARAMETER_BLOCKS,
        QwenClient,
        StructuredKnowledgeBase,
        _allowed_operators_for_state,
        _best_round_summary,
        _operator_selection_prompt,
        _optimization_state,
        execute_numerical_plan,
        load_case,
        validate_case,
    )
    from .optimization_schema import PARAMETER_BY_NAME
except ImportError:
    from compartment_ode import CompartmentODEForwardModel, ODEModelRegistry
    from multi_nuclide_llm_optimizer import (
        DEFAULT_LLM_GATE_PATH,
        DEFAULT_LLM_GATE_THRESHOLD,
        DEFAULT_MODEL_PATH,
        AdaptiveOperatorPlanner,
        LLMGatedOperatorPlanner,
        OPERATOR_DESCRIPTIONS,
        PARAMETER_BLOCKS,
        QwenClient,
        StructuredKnowledgeBase,
        _allowed_operators_for_state,
        _best_round_summary,
        _operator_selection_prompt,
        _optimization_state,
        execute_numerical_plan,
        load_case,
        validate_case,
    )
    from optimization_schema import PARAMETER_BY_NAME


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPERATORS = (
    "EVALUATE_CANDIDATE_POINTS",
    "LOCAL_REFINE",
    "SHRINK_BOUNDS",
    "REFINE_CURRENT_MODEL",
    "CHECK_IDENTIFIABILITY",
    "RUN_DE_GLOBAL",
    "STOP_UNIDENTIFIABLE",
)


def _perturbed_plan(
    plan: Dict[str, Any],
    *,
    shift_log10_intake: float = 1.0,
    shift_time_fraction: float = 0.35,
) -> Dict[str, Any]:
    """Build a deliberately imperfect incumbent for operator training.

    Profile output is usually already near its optimum, which makes STOP win
    every counterfactual branch. This plan keeps the same models and bounds but
    evaluates a bounded, reproducible off-centre candidate instead.
    """
    perturbed = copy.deepcopy(plan)
    for action in perturbed.get("actions", []):
        action["optimizer"] = "candidate_points"
        action["maxiter"] = 1
        action["candidate_points"] = []
        bounds = action.get("bounds", {})
        intake_bounds = bounds.get("log10_intake", [0.0, 8.0])
        time_bounds = bounds.get("intake_time_d", [0.0, 1.0])
        intake_low, intake_high = float(intake_bounds[0]), float(intake_bounds[1])
        time_low, time_high = float(time_bounds[0]), float(time_bounds[1])
        intake_mid = 0.5 * (intake_low + intake_high)
        time_mid = time_low + shift_time_fraction * (time_high - time_low)
        point = {
            "nuclide": action.get("nuclide"),
            "log10_intake": max(intake_low, min(intake_high, intake_mid + shift_log10_intake)),
            "intake_time_d": max(time_low, min(time_high, time_mid)),
        }
        action["candidate_points"] = [point]
        action["count_as_candidate_points"] = False
    return perturbed


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _history_item(
    round_index: int,
    plan: Dict[str, Any],
    planner_info: Dict[str, Any],
    tool_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "round": round_index,
        "plan": plan,
        "planner": planner_info,
        "planner_wall_time_s": 0.0,
        "tool_result": tool_result,
        "best_by_nuclide": _best_round_summary(tool_result),
        "operator_trace": {
            "operator": plan.get("operator"),
            "objective_improvement": None,
            "gain_per_forward_predict": None,
        },
        "round_wall_time_s": float(tool_result.get("numerical_wall_time_s", 0.0)),
    }


def _incumbent_value(row: Dict[str, Any], name: str) -> float:
    estimates = row.get("parameter_estimates", {})
    if name in estimates:
        return float(estimates[name])
    if name == "log10_intake":
        return float(row.get("log10_intake", 4.0))
    if name == "intake_time_d":
        return float(row.get("intake_time_d", 0.0))
    return float(PARAMETER_BY_NAME[name].default)


def _candidate_proposal(case: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Create bounded, non-incumbent seeds for the candidate-point branch."""
    case_by_nuclide = {item.nuclide: item for item in case.nuclides}
    names: List[str] = list(PARAMETER_BLOCKS["BASIC"])
    for row in state.get("per_nuclide", []):
        for name in row.get("recommended_parameter_names", []):
            if name in PARAMETER_BY_NAME and name not in names:
                names.append(name)

    points: List[Dict[str, Any]] = []
    for row in state.get("per_nuclide", []):
        if row.get("status") != "ok":
            continue
        nuclide = str(row.get("nuclide"))
        nuclide_case = case_by_nuclide[nuclide]
        earliest = min(float(item.time_d) for item in nuclide_case.observations)
        for direction in (-1.0, 1.0):
            point: Dict[str, Any] = {"nuclide": nuclide}
            for name in names:
                value = _incumbent_value(row, name)
                spec = PARAMETER_BY_NAME[name]
                low, high = float(spec.lower), float(spec.upper)
                if name == "intake_time_d":
                    high = min(high, earliest)
                    delta = max(0.03, 0.12 * max(earliest, 0.25))
                elif name == "log10_intake":
                    delta = 0.30
                else:
                    delta = 0.12
                point[name] = max(low, min(high, value + direction * delta))
            points.append(point)
    return {
        "parameter_names": names,
        "candidate_points": points[:8],
    }


def _limit_plan_budget(
    plan: Dict[str, Any],
    total_budget: int,
) -> Dict[str, Any]:
    limited = copy.deepcopy(plan)
    actions = limited.get("actions", [])
    material_branch_count = sum(
        max(1, len(action.get("model_ids", []))) for action in actions
    )
    # invert_one_nuclide applies forward_predict_budget independently to each
    # material candidate, so divide by model branches rather than only by
    # nuclides. The inner solver accepts small budgets for cheap counterfactual
    # branches; keeping the minimum at one preserves the requested total cap
    # even when a case has many compatible material models.
    per_material = max(
        1,
        int(total_budget) // max(1, int(material_branch_count)),
    )
    for action in actions:
        current = int(action.get("forward_predict_budget", per_material))
        action["forward_predict_budget"] = max(1, min(current, per_material))
    return limited


def _reward(
    objective_before: float,
    objective_after: float,
    forward_calls: int,
    budget: int,
    llm_wall_time_s: float,
    beta: float,
    gamma: float,
) -> Tuple[float, Dict[str, float]]:
    quality_gain = math.log1p(max(0.0, objective_before)) - math.log1p(
        max(0.0, objective_after)
    )
    ode_penalty = float(beta) * float(forward_calls) / max(1, int(budget))
    llm_penalty = float(gamma) * max(0.0, float(llm_wall_time_s))
    return quality_gain - ode_penalty - llm_penalty, {
        "log_objective_gain": quality_gain,
        "ode_cost_penalty": ode_penalty,
        "llm_latency_penalty": llm_penalty,
    }


def _operator_response(
    operator: str,
    proposal: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "operator": operator,
        "parameter_names": proposal.get(
            "parameter_names", list(PARAMETER_BLOCKS["BASIC"])
        ),
        "parameter_bounds": proposal.get("parameter_bounds", {}),
        "candidate_points": proposal.get("candidate_points", []),
        "reason": reason,
    }


def _plan_response_payload(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Expose the actual validated action schema used by a deterministic branch."""
    actions = [
        action for action in plan.get("actions", [])
        if isinstance(action, dict)
    ]
    if not actions:
        return {}
    # A plan can contain one action per nuclide and the identifiability
    # diagnostic can recommend a different block for each action. The current
    # LLM response schema has one global field, so use the stable union.
    parameter_names: List[str] = []
    parameter_bounds: Dict[str, Any] = {}
    for action in actions:
        for name in action.get("parameter_names", []):
            if name in PARAMETER_BY_NAME and name not in parameter_names:
                parameter_names.append(name)
        for name, bounds in action.get("parameter_bounds", {}).items():
            if name in PARAMETER_BY_NAME and name not in parameter_bounds:
                parameter_bounds[name] = bounds
    candidate_points: List[Dict[str, Any]] = []
    for action in actions:
        nuclide = action.get("nuclide")
        for point in action.get("candidate_points", []):
            if isinstance(point, dict):
                candidate_points.append({
                    "nuclide": nuclide,
                    **point,
                })
    return {
        "operator": plan.get("operator"),
        "parameter_names": parameter_names,
        "parameter_bounds": parameter_bounds,
        "candidate_points": candidate_points[:8],
    }


def _canonical_training_response(
    row: Dict[str, Any],
    response: Any,
    operator: Optional[str] = None,
) -> Dict[str, Any]:
    """Rebuild a safe supervision response from the executed oracle action.

    The response is rebuilt from the executed, audited action.
    """
    source = response if isinstance(response, dict) else {}
    executed = str(operator or source.get("operator") or "")
    basic = list(PARAMETER_BLOCKS["BASIC"])
    parameter_names = []
    for raw_name in source.get("parameter_names", []):
        name = str(raw_name)
        spec = PARAMETER_BY_NAME.get(name)
        if spec is not None and spec.llm_allowed and name not in parameter_names:
            parameter_names.append(name)
    for name in reversed(basic):
        if name not in parameter_names:
            parameter_names.insert(0, name)

    bounds = (
        {
            str(name): value
            for name, value in source.get("parameter_bounds", {}).items()
            if str(name) in parameter_names
        }
        if isinstance(source.get("parameter_bounds"), dict)
        else {}
    )
    points = source.get("candidate_points", [])
    if not isinstance(points, list):
        points = []
    return {
        "operator": executed,
        "parameter_names": parameter_names,
        "parameter_bounds": bounds,
        "candidate_points": points,
        "reason": str(
            source.get("reason", "Oracle action from fixed-state evaluation.")
        ),
    }


def repair_preference_training_responses(
    row: Dict[str, Any],
) -> Tuple[Dict[str, Any], int]:
    """Repair chosen supervision fields in one persisted preference record."""
    repaired = copy.deepcopy(row)
    changes = 0
    for candidate in repaired.get("candidate_operators", []):
        operator = str(candidate.get("executed_operator", ""))
        old = candidate.get("response")
        new = _canonical_training_response(repaired, old, operator)
        if old != new:
            candidate["response"] = new
            changes += 1

    chosen = repaired.get("chosen", {})
    chosen_operator = str(chosen.get("executed_operator", ""))
    old_chosen = chosen.get("response")
    new_chosen = _canonical_training_response(repaired, old_chosen, chosen_operator)
    if old_chosen != new_chosen:
        chosen["response"] = new_chosen
        changes += 1

    views = repaired.setdefault("training_views", {})
    sft = views.get("sft")
    if isinstance(sft, dict) and sft.get("response") != new_chosen:
        sft["response"] = copy.deepcopy(new_chosen)
        changes += 1
    dpo = views.get("dpo")
    if isinstance(dpo, dict) and dpo.get("chosen") != new_chosen:
        # Keep rejected as the raw/corrected LLM proposal; only canonicalize
        # the oracle side of the preference pair.
        dpo["chosen"] = copy.deepcopy(new_chosen)
        changes += 1
    return repaired, changes


def _evaluate_state(
    *,
    case: Any,
    split: str,
    knowledge: StructuredKnowledgeBase,
    history: Sequence[Dict[str, Any]],
    state_type: str,
    evaluator: AdaptiveOperatorPlanner,
    forward_model: CompartmentODEForwardModel,
    requested_operators: Sequence[str],
    counterfactual_budget: int,
    seed: int,
    beta: float,
    gamma: float,
    llm_planner: Optional[LLMGatedOperatorPlanner],
) -> Dict[str, Any]:
    state = _optimization_state(history[-1]["tool_result"])
    if state is None:
        raise RuntimeError("Counterfactual evaluation requires a numerical state.")
    allowed = _allowed_operators_for_state(
        state,
        history,
        evaluator.model_gap_threshold,
        evaluator.convergence_loss,
        evaluator.max_forward_predict_calls,
        force_easy_stop=False,
    )
    operators = [name for name in requested_operators if name in allowed]
    if not operators:
        operators = sorted(allowed)

    llm_plan: Optional[Dict[str, Any]] = None
    llm_info: Dict[str, Any] = {
        "source": "not_queried",
        "llm_invoked": False,
    }
    llm_wall_time_s = 0.0
    if llm_planner is not None:
        started = time.perf_counter()
        llm_plan, llm_info = llm_planner.propose(case, history)
        llm_wall_time_s = float(time.perf_counter() - started)

    prompt_operators = [name for name in operators if name in OPERATOR_DESCRIPTIONS]
    prompt = llm_info.get("prompt") or _operator_selection_prompt(
        case, knowledge, state, prompt_operators, history
    )
    candidate_proposal = _candidate_proposal(case, state)
    objective_before = float(state.get("objective_score", float("inf")))

    # Evaluate the non-LLM fallback from the same incumbent state.  This is
    # the counterfactual baseline for the gate: SKIP_LLM means continue with
    # this deterministic operator, not stop and not a synthetic reward of 0.
    skip_plan, skip_info = evaluator.propose(case, history)
    skip_operator = str(skip_plan.get("operator", ""))
    if bool(skip_plan.get("stop", False)):
        skip_objective_after = objective_before
        skip_forward_calls = 0
        skip_wall_time_s = 0.0
        skip_status = "terminal"
    else:
        skip_plan = _limit_plan_budget(skip_plan, counterfactual_budget)
        skip_result = execute_numerical_plan(
            case,
            skip_plan,
            forward_model,
            seed=seed + 900000,
        )
        skip_objective_after = float(
            skip_result.get("objective_score", float("inf"))
        )
        skip_forward_calls = int(
            skip_result.get("total_forward_predict_calls", 0)
        )
        skip_wall_time_s = float(skip_result.get("numerical_wall_time_s", 0.0))
        skip_status = str(skip_result.get("status", "unknown"))
    skip_reward, skip_components = _reward(
        objective_before,
        skip_objective_after,
        skip_forward_calls,
        counterfactual_budget,
        0.0,
        beta,
        gamma,
    )
    skip_baseline = {
        "operator": skip_operator,
        "status": skip_status,
        "objective_before": objective_before,
        "objective_after": skip_objective_after,
        "objective_improvement": objective_before - skip_objective_after,
        "forward_predict_calls": skip_forward_calls,
        "numerical_wall_time_s": skip_wall_time_s,
        "reward": skip_reward,
        "reward_components": skip_components,
        "planner_info": skip_info,
    }
    candidates: List[Dict[str, Any]] = []
    llm_executed_operator = (
        str(llm_plan.get("operator")) if isinstance(llm_plan, dict) else None
    )

    for index, requested_operator in enumerate(operators):
        proposal: Dict[str, Any] = {
            "source": "counterfactual_oracle",
            "operator": requested_operator,
            "reason": "Counterfactual branch from a fixed incumbent state.",
        }
        if requested_operator == "EVALUATE_CANDIDATE_POINTS":
            proposal.update(candidate_proposal)

        if llm_plan is not None and llm_executed_operator == requested_operator:
            plan = copy.deepcopy(llm_plan)
            audit = copy.deepcopy(llm_info)
            branch_source = "audited_llm_plan"
        else:
            plan, audit = evaluator._build_operator_plan(
                case,
                history,
                requested_operator,
                proposal["reason"],
                copy.deepcopy(proposal),
            )
            branch_source = "deterministic_counterfactual"

        executed_operator = str(plan.get("operator", requested_operator))
        if bool(plan.get("stop", False)):
            objective_after = objective_before
            forward_calls = 0
            numerical_wall_time_s = 0.0
            status = "terminal"
        else:
            plan = _limit_plan_budget(
                plan, counterfactual_budget
            )
            allocated_forward_budget = sum(
                int(action.get("forward_predict_budget", 0))
                * max(1, len(action.get("model_ids", [])))
                for action in plan.get("actions", [])
            )
            result = execute_numerical_plan(
                case,
                plan,
                forward_model,
                seed=seed + index * 1000,
            )
            objective_after = float(result.get("objective_score", float("inf")))
            forward_calls = int(result.get("total_forward_predict_calls", 0))
            numerical_wall_time_s = float(result.get("numerical_wall_time_s", 0.0))
            status = str(result.get("status", "unknown"))
        if bool(plan.get("stop", False)):
            allocated_forward_budget = 0

        reward, components = _reward(
            objective_before,
            objective_after,
            forward_calls,
            counterfactual_budget,
            llm_wall_time_s if llm_planner is not None else 0.0,
            beta,
            gamma,
        )
        # Supervise the final audited plan. The raw answer remains available
        # under row["llm"] and, when useful, on the rejected side of DPO.
        response_proposal = _plan_response_payload(plan)
        candidates.append({
            "requested_operator": requested_operator,
            "executed_operator": executed_operator,
            "branch_source": branch_source,
            "status": status,
            "objective_before": objective_before,
            "objective_after": objective_after,
            "objective_improvement": objective_before - objective_after,
            "forward_predict_calls": forward_calls,
            "allocated_forward_budget": allocated_forward_budget,
            "realized_budget_fraction": (
                float(forward_calls) / max(1, allocated_forward_budget)
                if allocated_forward_budget else 0.0
            ),
            "numerical_wall_time_s": numerical_wall_time_s,
            "reward": reward,
            "reward_components": components,
            "audit": {
                "strategy_correction": audit.get("strategy_correction"),
                "candidate_rejection": audit.get("candidate_rejection"),
            },
            "response": _operator_response(
                executed_operator,
                response_proposal if isinstance(response_proposal, dict) else {},
                str(plan.get("operator_reason", proposal["reason"])),
            ),
        })

    candidates.sort(key=lambda item: float(item["reward"]), reverse=True)
    chosen = candidates[0]
    raw_operator = llm_info.get("raw_llm_operator")
    raw_candidate = next(
        (
            item for item in candidates
            if item["requested_operator"] == raw_operator
            and item["executed_operator"] == raw_operator
        ),
        None,
    )
    if raw_operator is None:
        rejected = None
    elif raw_candidate is not None:
        rejected = raw_candidate
    else:
        rejected = {
            "requested_operator": raw_operator,
            "executed_operator": llm_info.get("executed_operator"),
            "reward": None,
            "invalid_or_corrected": True,
            "correction_reason": (
                llm_info.get("strategy_correction")
                or llm_info.get("candidate_rejection")
            ),
            "response": llm_info.get("raw_llm_proposal"),
        }

    preference_usable = bool(
        rejected is not None
        and (
            rejected.get("invalid_or_corrected") is True
            or chosen.get("executed_operator") != rejected.get("executed_operator")
            or float(chosen.get("reward", 0.0))
            > float(rejected.get("reward", 0.0)) + 1e-9
        )
    )
    rejected_response = (
        rejected.get("response") if isinstance(rejected, dict) else None
    )
    chosen_is_stop = str(chosen.get("executed_operator", "")).startswith("STOP_")
    training_views = {
        "sft": (
            {
                "prompt": prompt,
                "response": chosen.get("response"),
            }
            if not chosen_is_stop else None
        ),
        "dpo": (
            {
                "prompt": prompt,
                "chosen": chosen.get("response"),
                "rejected": rejected_response,
            }
            if (
                not chosen_is_stop
                and preference_usable
                and rejected_response is not None
            ) else None
        ),
    }
    return {
        "schema_version": "llm_operator_preference_v1",
        "case_id": case.case_id,
        "split": split,
        "state_type": state_type,
        "state": state,
        "allowed_operators": sorted(allowed),
        "evaluated_operators": operators,
        "prompt": prompt,
        "llm": {
            "queried": llm_planner is not None,
            "invoked": bool(llm_info.get("llm_invoked")),
            "wall_time_s": llm_wall_time_s,
            "source": llm_info.get("source"),
            "raw_output": llm_info.get("raw_output"),
            "raw_operator": raw_operator,
            "raw_proposal": llm_info.get("raw_llm_proposal"),
            "executed_operator": llm_info.get(
                "executed_operator", llm_executed_operator
            ),
            "raw_executed_match": llm_info.get("raw_executed_match"),
            "strategy_correction": llm_info.get("strategy_correction"),
            "candidate_rejection": llm_info.get("candidate_rejection"),
            "gate_ranking": llm_info.get("gate_ranking"),
        },
        "reward_config": {
            "formula": (
                "log1p(loss_before)-log1p(loss_after)"
                "-beta*forward_calls/budget-gamma*llm_wall_time_s"
            ),
            "beta": beta,
            "gamma": gamma,
            "counterfactual_forward_budget": counterfactual_budget,
        },
        "candidate_operators": candidates,
        "skip_baseline": skip_baseline,
        "chosen": chosen,
        "rejected": rejected,
        "preference_pair_usable": preference_usable,
        "training_views": training_views,
    }


def _read_cases(
    dataset_dir: Path,
    split: str,
    limit: int,
    case_ids: Sequence[str],
) -> List[Path]:
    splits = ("train", "validation", "test") if split == "all" else (split,)
    # Preserve the explicit train -> validation -> test order.  A global path
    # sort would put ``test`` first on common filesystems and could make a
    # limited training export consume held-out cases before training cases.
    files = [
        path
        for split_name in splits
        for path in sorted((dataset_dir / "inputs" / split_name).glob("*.json"))
    ]
    if case_ids:
        by_id = {path.stem: path for path in files}
        files = [by_id[case_id] for case_id in case_ids if case_id in by_id]
    return files[:max(1, limit)]


def _manifest_case_ids(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    payload = json.loads(resolve(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get("case_ids")
        if values is None:
            values = [
                item.get("case_id") for item in payload.get("cases", [])
                if isinstance(item, dict)
            ]
    else:
        values = []
    if not isinstance(values, list):
        raise ValueError("Case manifest must provide a case_ids list.")
    return list(dict.fromkeys(str(value) for value in values if value))


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _export_training_views(
    output: Path,
    rows: Sequence[Dict[str, Any]],
) -> Tuple[Path, Path, int, int]:
    sft_path = output.with_name(f"{output.stem}.sft.jsonl")
    dpo_path = output.with_name(f"{output.stem}.dpo.jsonl")
    sft_rows = []
    dpo_rows = []
    for original_row in rows:
        row, _ = repair_preference_training_responses(original_row)
        if str(row["chosen"].get("executed_operator", "")).startswith("STOP_"):
            continue
        sft = row["training_views"]["sft"]
        if sft is None:
            continue
        response_text = json.dumps(sft["response"], ensure_ascii=False)
        sft_rows.append({
            "case_id": row["case_id"],
            "split": row["split"],
            "state_type": row["state_type"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific optimization planner. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": sft["prompt"]},
                {"role": "assistant", "content": response_text},
            ],
            "oracle_reward": row["chosen"]["reward"],
        })
        dpo = row["training_views"].get("dpo")
        if dpo is not None:
            dpo_rows.append({
                "case_id": row["case_id"],
                "split": row["split"],
                "state_type": row["state_type"],
                "prompt": dpo["prompt"],
                "chosen": json.dumps(dpo["chosen"], ensure_ascii=False),
                "rejected": json.dumps(dpo["rejected"], ensure_ascii=False),
                "chosen_reward": row["chosen"]["reward"],
                "rejection_reason": (
                    row["llm"].get("strategy_correction")
                    or row["llm"].get("candidate_rejection")
                ),
            })
    _write_jsonl(sft_path, sft_rows)
    _write_jsonl(dpo_path, dpo_rows)
    return sft_path, dpo_path, len(sft_rows), len(dpo_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed-state counterfactual preferences for LLM operators."
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=ROOT / "data" / "synthetic_bioassay_icrp_v1",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="test",
    )
    parser.add_argument(
        "--state-modes",
        default="post_profile,post_identifiability,perturbed",
        help="Comma-separated state modes. perturbed creates imperfect incumbents for operator learning.",
    )
    parser.add_argument(
        "--registry-dir", type=Path,
        default=ROOT / "data" / "icrp_model_registry_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "temp" / "operator_preferences_icrp_train.jsonl",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Optional exact case id; repeat to select multiple cases.",
    )
    parser.add_argument(
        "--case-manifest",
        type=Path,
        help="JSON manifest containing case_ids or cases[].case_id.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--counterfactual-budget", type=int, default=300)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument(
        "--operators",
        default=",".join(DEFAULT_OPERATORS),
        help="Comma-separated counterfactual operator subset.",
    )
    parser.add_argument("--query-llm", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Checkpoint after each case and skip complete cases in an existing output.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--llm-adapter-path",
        type=Path,
        help="Optional LoRA adapter checkpoint applied to the Qwen planner.",
    )
    parser.add_argument(
        "--llm-gate-model-path", type=Path,
        default=DEFAULT_LLM_GATE_PATH,
    )
    parser.add_argument(
        "--llm-gate-threshold",
        type=float,
        default=DEFAULT_LLM_GATE_THRESHOLD,
    )
    args = parser.parse_args()

    dataset_dir = resolve(args.dataset_dir)
    registry = ODEModelRegistry.from_csv_dir(resolve(args.registry_dir), require_verified=True)
    selected_case_ids = list(dict.fromkeys(
        [*args.case_id, *_manifest_case_ids(args.case_manifest)]
    ))
    case_paths = _read_cases(
        dataset_dir, args.split, args.limit, selected_case_ids
    )
    if not case_paths:
        raise SystemExit(f"No cases found under {dataset_dir / 'inputs' / args.split}")
    requested_operators = [
        item.strip() for item in args.operators.split(",") if item.strip()
    ]
    unknown = sorted(set(requested_operators) - set(OPERATOR_DESCRIPTIONS))
    if unknown:
        raise SystemExit(f"Unknown operators: {unknown}")

    output = resolve(args.output)
    rows: List[Dict[str, Any]] = []
    if args.resume and output.is_file():
        rows = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected_ids = {path.stem for path in case_paths}
        unexpected = sorted({str(row.get("case_id")) for row in rows} - selected_ids)
        if unexpected:
            raise SystemExit(
                f"Resume output contains cases outside the current selection: {unexpected}"
            )
        for row in rows:
            config = row.get("reward_config", {})
            if (
                int(config.get("counterfactual_forward_budget", -1))
                != int(args.counterfactual_budget)
                or not math.isclose(float(config.get("beta", -1.0)), float(args.beta))
                or not math.isclose(float(config.get("gamma", -1.0)), float(args.gamma))
                or bool(row.get("llm", {}).get("queried")) != bool(args.query_llm)
            ):
                raise SystemExit(
                    "Resume output does not match the current budget/reward/LLM configuration."
                )
    completed_case_ids = {
        case_id for case_id in {str(row.get("case_id")) for row in rows}
        if {
            str(row.get("state_type")) for row in rows
            if str(row.get("case_id")) == case_id
        } >= {"post_profile", "post_identifiability"}
    }
    remaining_case_paths = [
        path for path in case_paths if path.stem not in completed_case_ids
    ]
    adapter_path = resolve(args.llm_adapter_path) if args.llm_adapter_path else None
    qwen_client = (
        QwenClient(resolve(args.model_path), adapter_path=adapter_path)
        if args.query_llm and remaining_case_paths else None
    )
    for case_index, case_path in enumerate(case_paths):
        if case_path.stem in completed_case_ids:
            continue
        case = load_case(case_path)
        case_split = case_path.parent.name
        validate_case(case, ode_registry=registry)
        knowledge = StructuredKnowledgeBase(
            dataset_dir,
            ode_registry=registry,
            icrp_root=ROOT / "data" / "icrp_knowledge_base_v2",
        )
        evaluator = AdaptiveOperatorPlanner(knowledge)
        llm_planner = (
            LLMGatedOperatorPlanner(
                knowledge,
                qwen_client=qwen_client,
                gate_checkpoint_path=resolve(args.llm_gate_model_path),
                gate_threshold=args.llm_gate_threshold,
            )
            if qwen_client is not None else None
        )
        forward_model = CompartmentODEForwardModel(registry)

        profile_plan, profile_info = evaluator._build_operator_plan(
            case,
            [],
            "RUN_PROFILE",
            "Mandatory low-cost profile probe for preference-state construction.",
            {"source": "preference_builder", "operator": "RUN_PROFILE"},
        )
        profile_result = execute_numerical_plan(
            case, profile_plan, forward_model, seed=args.seed + case_index * 10000
        )
        profile_history = [
            _history_item(0, profile_plan, profile_info, profile_result)
        ]
        state_modes = {
            item.strip() for item in args.state_modes.split(",") if item.strip()
        }
        if "post_profile" in state_modes:
            rows.append(_evaluate_state(
                case=case,
                split=case_split,
                knowledge=knowledge,
                history=profile_history,
                state_type="post_profile",
                evaluator=evaluator,
                forward_model=forward_model,
                requested_operators=requested_operators,
                counterfactual_budget=args.counterfactual_budget,
                seed=args.seed + case_index * 10000 + 1000,
                beta=args.beta,
                gamma=args.gamma,
                llm_planner=llm_planner,
            ))

        ident_plan, ident_info = evaluator._build_operator_plan(
            case,
            profile_history,
            "CHECK_IDENTIFIABILITY",
            "Construct the post-identifiability preference state.",
            {"source": "preference_builder", "operator": "CHECK_IDENTIFIABILITY"},
        )
        ident_plan = _limit_plan_budget(
            ident_plan, args.counterfactual_budget
        )
        ident_result = execute_numerical_plan(
            case, ident_plan, forward_model,
            seed=args.seed + case_index * 10000 + 500,
        )
        ident_history = profile_history + [
            _history_item(1, ident_plan, ident_info, ident_result)
        ]
        if "post_identifiability" in state_modes:
            rows.append(_evaluate_state(
                case=case,
                split=case_split,
                knowledge=knowledge,
                history=ident_history,
                state_type="post_identifiability",
                evaluator=evaluator,
                forward_model=forward_model,
                requested_operators=requested_operators,
                counterfactual_budget=args.counterfactual_budget,
                seed=args.seed + case_index * 10000 + 6000,
                beta=args.beta,
                gamma=args.gamma,
                llm_planner=llm_planner,
            ))
        if "perturbed" in state_modes:
            perturbed_plan = _perturbed_plan(profile_plan)
            perturbed_result = execute_numerical_plan(
                case,
                perturbed_plan,
                forward_model,
                seed=args.seed + case_index * 10000 + 8000,
            )
            perturbed_history = [
                _history_item(0, perturbed_plan, profile_info, perturbed_result)
            ]
            rows.append(_evaluate_state(
                case=case,
                split=case_split,
                knowledge=knowledge,
                history=perturbed_history,
                state_type="perturbed",
                evaluator=evaluator,
                forward_model=forward_model,
                requested_operators=requested_operators,
                counterfactual_budget=args.counterfactual_budget,
                seed=args.seed + case_index * 10000 + 9000,
                beta=args.beta,
                gamma=args.gamma,
                llm_planner=llm_planner,
            ))
        if args.resume:
            _write_jsonl(output, rows)

    _write_jsonl(output, rows)
    sft_path, dpo_path, sft_count, dpo_count = _export_training_views(output, rows)
    correction_reasons = Counter(
        row["llm"].get("strategy_correction")
        or row["llm"].get("candidate_rejection")
        for row in rows
        if row["llm"].get("strategy_correction")
        or row["llm"].get("candidate_rejection")
    )
    operator_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"reward": [], "forward_calls": [], "wins": []}
    )
    for row in rows:
        chosen_operator = row["chosen"]["executed_operator"]
        for candidate in row["candidate_operators"]:
            operator = candidate["executed_operator"]
            operator_metrics[operator]["reward"].append(float(candidate["reward"]))
            operator_metrics[operator]["forward_calls"].append(
                float(candidate["forward_predict_calls"])
            )
            operator_metrics[operator]["wins"].append(
                float(operator == chosen_operator)
            )
    audited_count = sum(
        row["llm"]["raw_operator"] is not None for row in rows
    )
    mismatch_count = sum(
        row["llm"]["raw_executed_match"] is False for row in rows
    )
    summary = {
        "schema_version": "llm_operator_preference_summary_v1",
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "case_count": len(case_paths),
        "resumed_case_count": len(completed_case_ids),
        "split_counts": dict(Counter(
            split for _, split in {
                (str(row["case_id"]), str(row["split"])) for row in rows
            }
        )),
        "state_count": len(rows),
        "llm_queried_state_count": sum(row["llm"]["queried"] for row in rows),
        "llm_invoked_state_count": sum(row["llm"]["invoked"] for row in rows),
        "raw_executed_mismatch_count": sum(
            row["llm"]["raw_executed_match"] is False for row in rows
        ),
        "raw_executed_mismatch_rate": (
            float(mismatch_count) / audited_count if audited_count else 0.0
        ),
        "mean_llm_wall_time_s": (
            statistics.mean(float(row["llm"]["wall_time_s"]) for row in rows)
            if rows else 0.0
        ),
        "mean_chosen_reward": (
            statistics.mean(float(row["chosen"]["reward"]) for row in rows)
            if rows else None
        ),
        "mean_chosen_forward_predict_calls": (
            statistics.mean(
                float(row["chosen"]["forward_predict_calls"]) for row in rows
            ) if rows else None
        ),
        "preference_pair_usable_count": sum(
            row["preference_pair_usable"] for row in rows
        ),
        "sft_example_count": sft_count,
        "dpo_example_count": dpo_count,
        "chosen_operator_counts": {
            operator: sum(
                row["chosen"]["executed_operator"] == operator for row in rows
            )
            for operator in sorted({
                row["chosen"]["executed_operator"] for row in rows
            })
        },
        "raw_operator_counts": dict(Counter(
            row["llm"]["raw_operator"] for row in rows
            if row["llm"]["raw_operator"] is not None
        )),
        "executed_llm_operator_counts": dict(Counter(
            row["llm"]["executed_operator"] for row in rows
            if row["llm"]["executed_operator"] is not None
        )),
        "correction_reason_counts": dict(correction_reasons),
        "counterfactual_operator_metrics": {
            operator: {
                "count": len(values["reward"]),
                "mean_reward": statistics.mean(values["reward"]),
                "mean_forward_predict_calls": statistics.mean(
                    values["forward_calls"]
                ),
                "win_count": int(sum(values["wins"])),
            }
            for operator, values in sorted(operator_metrics.items())
        },
        "output": str(output),
        "sft_output": str(sft_path),
        "dpo_output": str(dpo_path),
        "clinical_use": False,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
