"""Online verifier-in-the-loop GRPO for the fixed ICRP operator planner.

Unlike the preference-data trainer, this entry point never reads a stored
operator reward.  Every rollout is sampled from the current policy, audited,
translated into a numerical plan, and evaluated by the frozen ICRP ODE before
the group-relative LoRA update is applied.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .compartment_ode import CompartmentODEForwardModel, ODEModelRegistry
    from .multi_nuclide_llm_optimizer import (
        AdaptiveOperatorPlanner,
        StructuredKnowledgeBase,
        _allowed_operators_for_state,
        _best_round_summary,
        _operator_selection_prompt,
        _optimization_state,
        _validate_llm_proposal,
        execute_numerical_plan,
        load_case,
        validate_case,
    )
    from .train_qwen_grpo import (
        _sequence_logprob,
        generation_inputs,
        proposal_from_text,
    )
    from .train_qwen_lora import _adapter_state, _replace_target_modules, load_lora_adapter
    from .operator_policy import GATE_ACTIONS, FEATURE_NAMES, OperatorMLP, state_to_features
except ImportError:
    from compartment_ode import CompartmentODEForwardModel, ODEModelRegistry
    from multi_nuclide_llm_optimizer import (
        AdaptiveOperatorPlanner,
        StructuredKnowledgeBase,
        _allowed_operators_for_state,
        _best_round_summary,
        _operator_selection_prompt,
        _optimization_state,
        _validate_llm_proposal,
        execute_numerical_plan,
        load_case,
        validate_case,
    )
    from train_qwen_grpo import _sequence_logprob, generation_inputs, proposal_from_text
    from train_qwen_lora import _adapter_state, _replace_target_modules, load_lora_adapter
    from operator_policy import GATE_ACTIONS, FEATURE_NAMES, OperatorMLP, state_to_features


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _case_paths(dataset_dir: Path, split: str, limit: int) -> List[Path]:
    paths = sorted((_resolve(dataset_dir) / "inputs" / split).glob("*.json"))
    return paths[: max(1, int(limit))]


def _limit_plan_budget(plan: Dict[str, Any], budget: int) -> Dict[str, Any]:
    """Cap the executor budget without changing any medical model fields."""
    limited = copy.deepcopy(plan)
    actions = [item for item in limited.get("actions", []) if isinstance(item, dict)]
    branches = sum(max(1, len(item.get("model_ids", []))) for item in actions)
    per_model = max(1, int(budget) // max(1, branches))
    for action in actions:
        current = int(action.get("forward_predict_budget", per_model))
        action["forward_predict_budget"] = max(1, min(current, per_model))
    return limited


def _reward(
    objective_before: float,
    objective_after: float,
    forward_calls: int,
    budget: int,
    *,
    beta: float,
    invalid: bool = False,
    stopped: bool = False,
) -> Tuple[float, Dict[str, float]]:
    if invalid:
        return -1.0, {"invalid_penalty": 1.0}
    if not math.isfinite(objective_after):
        return -1.0, {"failed_execution_penalty": 1.0}
    quality_gain = math.log1p(max(0.0, objective_before)) - math.log1p(
        max(0.0, objective_after)
    )
    cost = float(beta) * float(forward_calls) / max(1, int(budget))
    # A premature stop has no ODE gain and should not become an attractive
    # zero-cost action. The auditor still records it as a valid proposal.
    stop_penalty = 0.25 if stopped and objective_before > 1e-2 else 0.0
    return quality_gain - cost - stop_penalty, {
        "log_objective_gain": quality_gain,
        "ode_cost_penalty": cost,
        "premature_stop_penalty": stop_penalty,
    }


def _initial_state(
    case: Any,
    knowledge: StructuredKnowledgeBase,
    forward: CompartmentODEForwardModel,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], AdaptiveOperatorPlanner]:
    """Run the mandatory deterministic profile probe for a fresh case."""
    planner = AdaptiveOperatorPlanner(knowledge, max_forward_predict_calls=500)
    plan, planner_info = planner.propose(case, [])
    tool_result = execute_numerical_plan(case, plan, forward, seed=seed)
    history = [{
        "round": 0,
        "plan": plan,
        "planner": planner_info,
        "tool_result": tool_result,
        "best_by_nuclide": _best_round_summary(tool_result),
        "operator_trace": {"operator": plan.get("operator")},
    }]
    state = _optimization_state(tool_result)
    if state is None:
        raise RuntimeError(f"Profile probe did not produce a numerical state for {case.case_id}.")
    return state, history, planner


def _load_gate(path: Path) -> Tuple[OperatorMLP, Dict[str, Any]]:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES or tuple(payload.get("operators", ())) != GATE_ACTIONS:
        raise ValueError("Gate checkpoint schema does not match this code version.")
    model = OperatorMLP(int(payload["input_size"]), int(payload["hidden_size"]), len(GATE_ACTIONS))
    model.load_state_dict(payload["model_state_dict"])
    return model, {"feature_mean": payload["feature_mean"], "feature_std": payload["feature_std"], "input_size": payload["input_size"], "hidden_size": payload["hidden_size"], "feature_names": list(FEATURE_NAMES), "operators": list(GATE_ACTIONS)}


def _skip_rollout(*, case: Any, forward: CompartmentODEForwardModel, planner: AdaptiveOperatorPlanner, history: Sequence[Dict[str, Any]], objective_before: float, budget: int, beta: float, seed: int) -> Tuple[float, Dict[str, Any]]:
    plan, info = planner.propose(case, history)
    if bool(plan.get("stop")):
        reward, components = _reward(objective_before, objective_before, 0, budget, beta=beta, stopped=True)
        return reward, {"operator": plan.get("operator"), "objective_after": objective_before, "forward_predict_calls": 0, "status": "terminal", "reward_components": components, "planner_info": info}
    result = execute_numerical_plan(case, _limit_plan_budget(plan, budget), forward, seed=seed)
    objective_after = float(result.get("objective_score", float("inf")))
    calls = int(result.get("total_forward_predict_calls", 0))
    reward, components = _reward(objective_before, objective_after, calls, budget, beta=beta)
    return reward, {"operator": plan.get("operator"), "objective_after": objective_after, "forward_predict_calls": calls, "status": result.get("status"), "reward_components": components}


def _online_rollout(
    *,
    case: Any,
    knowledge: StructuredKnowledgeBase,
    forward: CompartmentODEForwardModel,
    planner: AdaptiveOperatorPlanner,
    state: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    prompt: str,
    raw_text: str,
    seed: int,
    budget: int,
    beta: float,
) -> Tuple[float, Dict[str, Any]]:
    """Audit and execute one current-policy proposal against the live ODE."""
    objective_before = float(state.get("objective_score", float("inf")))
    allowed = _allowed_operators_for_state(
        state,
        history,
        planner.model_gap_threshold,
        planner.convergence_loss,
        planner.max_forward_predict_calls,
        force_easy_stop=False,
    )
    proposal = proposal_from_text(raw_text)
    audit: Dict[str, Any] = {
        "raw_output": raw_text,
        "allowed_operators": sorted(allowed),
        "proposal": proposal,
        "valid": False,
        "executed_operator": None,
        "objective_before": objective_before,
        "objective_after": None,
        "forward_predict_calls": 0,
        "status": "invalid",
    }
    try:
        if proposal is None:
            raise ValueError("No JSON proposal found.")
        checked = _validate_llm_proposal(case, knowledge, proposal, sorted(allowed))
        operator = str(checked["operator"])
        planner_info = {
            "source": "online_qwen_rollout",
            "operator": operator,
            "raw_llm_operator": operator,
            "raw_llm_proposal": checked,
            "parameter_names": checked.get("parameter_names"),
            "parameter_bounds": checked.get("parameter_bounds", {}),
            "candidate_points": checked.get("candidate_points", []),
            "reason": checked.get("reason", "Online policy proposal."),
        }
        plan, plan_info = planner._build_operator_plan(
            case,
            history,
            operator,
            str(checked.get("reason", "Online policy proposal.")),
            planner_info,
        )
        audit["valid"] = True
        audit["executed_operator"] = str(plan.get("operator", operator))
        audit["plan"] = plan
        audit["planner_info"] = plan_info
        if bool(plan.get("stop")):
            objective_after = objective_before
            forward_calls = 0
            result = {"status": "terminal", "objective_score": objective_after}
        else:
            plan = _limit_plan_budget(plan, budget)
            audit["plan"] = plan
            result = execute_numerical_plan(case, plan, forward, seed=seed)
            objective_after = float(result.get("objective_score", float("inf")))
            forward_calls = int(result.get("total_forward_predict_calls", 0))
        reward, components = _reward(
            objective_before,
            objective_after,
            forward_calls,
            budget,
            beta=beta,
            stopped=bool(plan.get("stop")),
        )
        audit.update({
            "status": result.get("status", "unknown"),
            "objective_after": objective_after,
            "forward_predict_calls": forward_calls,
            "result": result,
            "reward_components": components,
        })
        return reward, audit
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        audit["error"] = str(error)
        reward, components = _reward(
            objective_before, float("inf"), 0, budget, beta=beta, invalid=True
        )
        audit["reward_components"] = components
        return reward, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "synthetic_bioassay_icrp_v1")
    parser.add_argument("--registry-dir", type=Path, default=ROOT / "data" / "icrp_model_registry_v1")
    parser.add_argument("--icrp-knowledge-dir", type=Path, default=ROOT / "data" / "icrp_knowledge_base_v2")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--cases", type=int, default=2, help="Number of fresh cases to cycle through.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "Qwen3-4b")
    parser.add_argument("--llm-adapter-path", type=Path)
    parser.add_argument("--gate-path", type=Path, help="Optional pretrained gate checkpoint; otherwise initialize a fresh binary gate.")
    parser.add_argument("--gate-hidden-size", type=int, default=32)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gate-temperature", type=float, default=0.05)
    parser.add_argument("--gate-exploration", type=float, default=0.20, help="Minimum CALL probability used for the sampled active gate action.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "online_grpo_icrp_v1")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--forward-budget", type=int, default=48)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = _resolve(args.dataset_dir)
    registry = ODEModelRegistry.from_csv_dir(_resolve(args.registry_dir), require_verified=True)
    paths = _case_paths(dataset_dir, args.split, args.cases)
    if not paths:
        raise SystemExit(f"No cases found under {dataset_dir / 'inputs' / args.split}")
    cases = [load_case(path) for path in paths]
    for case in cases:
        validate_case(case, ode_registry=registry)
    forward = CompartmentODEForwardModel(registry)
    knowledge = StructuredKnowledgeBase(
        dataset_root=dataset_dir,
        ode_registry=registry,
        icrp_root=_resolve(args.icrp_knowledge_dir),
    )

    model_path = _resolve(args.model_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "samples.jsonl"
    readable_path = output_dir / "samples.txt"
    adapter_metadata: Dict[str, Any] = {}
    gate_metadata: Dict[str, Any]
    adapter_path = _resolve(args.llm_adapter_path) if args.llm_adapter_path else None
    if args.gate_path:
        gate, gate_metadata = _load_gate(_resolve(args.gate_path))
    else:
        gate = OperatorMLP(len(FEATURE_NAMES), args.gate_hidden_size, len(GATE_ACTIONS))
        gate_metadata = {"feature_mean": [0.0] * len(FEATURE_NAMES), "feature_std": [1.0] * len(FEATURE_NAMES), "input_size": len(FEATURE_NAMES), "hidden_size": args.gate_hidden_size, "feature_names": list(FEATURE_NAMES), "operators": list(GATE_ACTIONS)}
    if adapter_path:
        payload = torch.load(str(adapter_path), map_location="cpu", weights_only=False)
        adapter_metadata = dict(payload.get("metadata", {}))
    rank = int(adapter_metadata.get("rank", args.rank))
    alpha = float(adapter_metadata.get("alpha", args.alpha))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    gate.to(device)
    gate_optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_learning_rate)
    feature_mean = torch.tensor(gate_metadata["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.tensor(gate_metadata["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replaced = _replace_target_modules(model, rank, alpha, 0.05, ("q_proj", "v_proj"))
    if not replaced:
        raise RuntimeError("No q_proj/v_proj modules were replaced for LoRA.")
    if adapter_path:
        load_lora_adapter(model, adapter_path)
    model.config.use_cache = False
    model.to(device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    history: List[Dict[str, Any]] = []
    case_cache: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]], AdaptiveOperatorPlanner]] = {}
    with sample_path.open("w", encoding="utf-8") as sample_log, readable_path.open("w", encoding="utf-8") as readable_log:
        for step in range(max(1, args.steps)):
            case = cases[step % len(cases)]
            if case.case_id not in case_cache:
                case_cache[case.case_id] = _initial_state(case, knowledge, forward, args.seed + step * 1000)
            state, history_before, planner = case_cache[case.case_id]
            allowed = _allowed_operators_for_state(
                state, history_before, planner.model_gap_threshold,
                planner.convergence_loss, planner.max_forward_predict_calls,
                force_easy_stop=False,
            )
            objective_before = float(state.get("objective_score", float("inf")))
            features = (torch.from_numpy(state_to_features(state)).to(device) - feature_mean) / feature_std
            gate_logits = gate(features.unsqueeze(0))
            call_probability = torch.softmax(gate_logits.detach(), dim=-1)[0, GATE_ACTIONS.index("CALL_LLM")]
            active_call_probability = max(float(args.gate_exploration), float(call_probability.cpu()))
            active_call = random.random() < active_call_probability
            skip_reward, skip_record = _skip_rollout(case=case, forward=forward, planner=planner, history=history_before, objective_before=objective_before, budget=args.forward_budget, beta=args.beta, seed=args.seed + step * 1000 + 777)
            # The call-vs-skip target is finalized after the live Qwen rollouts.
            target_call = 0.5
            prompt = _operator_selection_prompt(case, knowledge, state, sorted(allowed), history_before)
            encoded = generation_inputs(tokenizer, prompt, args.max_length)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=max(2, args.group_size),
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                )
            prompt_length = int(input_ids.shape[-1])
            texts = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
            rollout_rewards: List[float] = []
            rollout_records: List[Dict[str, Any]] = []
            for index, text in enumerate(texts):
                reward, audit = _online_rollout(
                    case=case, knowledge=knowledge, forward=forward,
                    planner=planner, state=state, history=history_before,
                    prompt=prompt, raw_text=text, seed=args.seed + step * 10000 + index,
                    budget=args.forward_budget, beta=args.beta,
                )
                rollout_rewards.append(float(reward))
                audit["reward"] = float(reward)
                rollout_records.append(audit)
            rewards = torch.tensor(rollout_rewards, dtype=torch.float32, device=device)
            call_reward = float(rewards.mean().detach().cpu())
            active_reward = call_reward if active_call else float(skip_reward)
            target_call = max(-30.0, min(30.0, (call_reward - skip_reward) / max(args.gate_temperature, 1e-6)))
            target_call = float(torch.sigmoid(torch.tensor(target_call)).item())
            target = torch.tensor([[1.0 - target_call, target_call]], dtype=torch.float32, device=device)
            gate_loss = -(target * F.log_softmax(gate_logits, dim=-1)).sum(dim=-1).mean()
            gate_optimizer.zero_grad(set_to_none=True)
            gate_loss.backward()
            clip_grad_norm_(gate.parameters(), 1.0)
            gate_optimizer.step()
            advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-3)
            if active_call:
                optimizer.zero_grad(set_to_none=True)
                logprob = _sequence_logprob(model, generated, prompt_length, int(tokenizer.pad_token_id))
                loss = -(advantages.detach() * logprob).mean()
                loss.backward()
                clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                loss_value = float(loss.detach().cpu())
            else:
                # The call branch was evaluated as a counterfactual for the gate,
                # but SKIP_LLM means no Qwen policy update on this state.
                loss_value = 0.0
            record = {
                "step": step + 1,
                "case_id": case.case_id,
                "split": args.split,
                "algorithm": "strict_online_joint_grpo",
                "objective_before": float(state.get("objective_score", float("inf"))),
                "gate_call_probability": float(call_probability.cpu()),
                "gate_active_call_probability": active_call_probability,
                "gate_action": "CALL_LLM" if active_call else "SKIP_LLM",
                "active_reward": active_reward,
                "gate_target_call_probability": target_call,
                "gate_loss": float(gate_loss.detach().cpu()),
                "skip_reward": float(skip_reward),
                "skip_rollout": skip_record,
                "allowed_operators": sorted(allowed),
                "loss": loss_value,
                "mean_reward": float(rewards.mean().detach().cpu()),
                "max_reward": float(rewards.max().detach().cpu()),
                "min_reward": float(rewards.min().detach().cpu()),
                "valid_output_fraction": float(sum(item["valid"] for item in rollout_records) / max(1, len(rollout_records))),
                "forward_predict_calls": int(sum(item["forward_predict_calls"] for item in rollout_records)),
                "prompt": prompt,
                "rollouts": rollout_records,
            }
            history.append(record)
            sample_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            sample_log.flush()
            readable_log.write(json.dumps({
                key: record[key] for key in (
                    "step", "case_id", "loss", "mean_reward", "max_reward",
                    "min_reward", "gate_call_probability", "gate_action", "active_reward", "gate_loss", "skip_reward",
                    "valid_output_fraction", "forward_predict_calls",
                )
            }, ensure_ascii=False) + "\n")
            readable_log.flush()
            print(json.dumps({
                key: record[key] for key in (
                    "step", "case_id", "loss", "mean_reward", "max_reward",
                    "gate_call_probability", "gate_action", "active_reward",
                    "valid_output_fraction", "forward_predict_calls",
                )
            }, ensure_ascii=False), flush=True)

    metadata = {
        "algorithm": "strict_online_joint_grpo",
        "base_model": str(model_path),
        "split": args.split,
        "case_count": len(cases),
        "steps": args.steps,
        "group_size": args.group_size,
        "forward_budget": args.forward_budget,
        "beta": args.beta,
        "rank": rank,
        "alpha": alpha,
        "replaced_modules": replaced,
        "history": history,
        "sample_log": str(sample_path),
        "clinical_use": False,
    }
    metadata["gate_algorithm"] = "online_counterfactual_skip_vs_call"
    metadata["gate_exploration"] = args.gate_exploration
    metadata["gate_history"] = [{key: row[key] for key in ("step", "gate_call_probability", "gate_active_call_probability", "gate_action", "active_reward", "gate_target_call_probability", "gate_loss", "skip_reward") if key in row} for row in history]
    torch.save({"adapter_state": _adapter_state(model), "metadata": metadata}, output_dir / "adapter.pt")
    torch.save({"model_state_dict": {key: value.detach().cpu() for key, value in gate.state_dict().items()}, **gate_metadata, "algorithm": "online_counterfactual_skip_vs_call", "history": metadata["gate_history"]}, output_dir / "llm_gate.pt")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "adapter": str(output_dir / "adapter.pt"), "metadata": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
