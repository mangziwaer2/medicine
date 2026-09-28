"""Minimal GRPO trainer for the fixed ICRP operator-planning task.

This trainer deliberately optimizes only the Qwen policy over a finite
operator/model-ID action space.  Every sampled answer is scored by the stored
external ODE-verifier rewards; invalid JSON or an unregistered operator gets a
negative reward.  It never changes the ICRP ODE registry.

The implementation is intentionally dependency-light so it can run on a cloud
GPU with the existing custom LoRA wrapper.  It is a research baseline rather
than a distributed production trainer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .multi_nuclide_llm_optimizer import _extract_json_object
    from .train_qwen_lora import _adapter_state, _replace_target_modules, load_lora_adapter
except ImportError:
    from multi_nuclide_llm_optimizer import _extract_json_object
    from train_qwen_lora import _adapter_state, _replace_target_modules, load_lora_adapter


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = "You are a scientific optimization planner. Return only valid JSON."


def read_rows(path: Path, split: str | None = None) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split is None or str(row.get("split", "")) == split:
            rows.append(row)
    return rows


def prompt_for(row: Dict[str, Any]) -> str:
    value = row.get("prompt") or row.get("llm", {}).get("prompt")
    if not value:
        raise ValueError("GRPO row has no prompt field.")
    return str(value)


def generation_inputs(tokenizer: Any, prompt: str, max_length: int) -> Dict[str, torch.Tensor]:
    """Encode a planner prompt exactly as it was formatted for SFT/QwenClient."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(prompt)},
    ]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
    if hasattr(encoded, "items"):
        result = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    else:
        result = {"input_ids": encoded if isinstance(encoded, torch.Tensor) else torch.tensor(encoded)}
    if result["input_ids"].ndim == 1:
        result = {key: value.unsqueeze(0) for key, value in result.items()}
    if "attention_mask" not in result:
        result["attention_mask"] = result["input_ids"].ne(tokenizer.pad_token_id).long()
    # Keep the end of the numerical state/context, which is more useful than
    # truncating the current state away from the right side of the prompt.
    if result["input_ids"].shape[-1] > max_length:
        head = max(1, int(max_length * 0.68))
        tail = max(1, max_length - head)
        result = {
            key: torch.cat((value[..., :head], value[..., -tail:]), dim=-1)
            for key, value in result.items()
        }
    return result


PROPOSAL_KEYS = {
    "operator",
    "parameter_names",
    "parameter_bounds",
    "model_ids_by_nuclide",
    "candidate_points",
    "reason",
}
POINT_KEYS = {"nuclide", "model_id", "log10_intake", "intake_time_d"}
PROTECTED_KEYS = {
    "compartment",
    "compartments",
    "transfer",
    "transfers",
    "rate",
    "rate_per_d",
    "prior",
    "physical_decay",
    "physical_decay_constant_per_d",
    "measurement_mapping",
    "ode",
    "ode_code",
    "python",
    "code",
}


def proposal_from_text(text: str) -> Dict[str, Any] | None:
    try:
        payload = _extract_json_object(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload


def operator_from_text(text: str) -> str | None:
    payload = proposal_from_text(text)
    if payload is None:
        return None
    operator = payload.get("operator")
    return str(operator) if operator is not None else None


def _has_protected_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in PROTECTED_KEYS or _has_protected_key(item):
                return True
    elif isinstance(value, list):
        return any(_has_protected_key(item) for item in value)
    return False


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _proposal_is_valid(row: Dict[str, Any], proposal: Dict[str, Any]) -> bool:
    if set(proposal) - PROPOSAL_KEYS or _has_protected_key(proposal):
        return False
    operator = str(proposal.get("operator", ""))
    if not operator or operator not in {str(item) for item in row.get("allowed_operators", [])}:
        return False

    names = proposal.get("parameter_names", ["log10_intake", "intake_time_d"])
    if names != ["log10_intake", "intake_time_d"]:
        return False
    bounds = proposal.get("parameter_bounds", {})
    if not isinstance(bounds, dict) or set(bounds) - set(names):
        return False
    global_bounds = {
        "log10_intake": (0.0, 12.0),
        "intake_time_d": (-3650.0, 3650.0),
    }
    for name, value in bounds.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        low = _finite_number(value[0])
        high = _finite_number(value[1])
        floor, ceiling = global_bounds[name]
        if low is None or high is None or not (floor <= low < high <= ceiling):
            return False

    state_rows = [
        item for item in row.get("state", {}).get("per_nuclide", [])
        if isinstance(item, dict) and item.get("nuclide")
    ]
    registered = {
        str(item["nuclide"]): str(item.get("best_model_id", ""))
        for item in state_rows
    }
    selected = proposal.get("model_ids_by_nuclide", {})
    if not isinstance(selected, dict):
        return False
    for nuclide, model_ids in selected.items():
        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if (
            str(nuclide) not in registered
            or not isinstance(model_ids, list)
            or not model_ids
            or any(str(model_id) != registered[str(nuclide)] for model_id in model_ids)
        ):
            return False

    points = proposal.get("candidate_points", [])
    if not isinstance(points, list) or len(points) > 32:
        return False
    if operator == "EVALUATE_CANDIDATE_POINTS" and not points:
        return False
    for point in points:
        if not isinstance(point, dict) or set(point) - POINT_KEYS:
            return False
        nuclide = str(point.get("nuclide", ""))
        if nuclide not in registered:
            return False
        if point.get("model_id") and str(point["model_id"]) != registered[nuclide]:
            return False
        log_intake = _finite_number(point.get("log10_intake"))
        intake_time = _finite_number(point.get("intake_time_d"))
        if (
            log_intake is None
            or intake_time is None
            or not 0.0 <= log_intake <= 12.0
            or not -3650.0 <= intake_time <= 3650.0
        ):
            return False
    return True


def reward_for(row: Dict[str, Any], proposal: Dict[str, Any] | None) -> float:
    if proposal is None or not _proposal_is_valid(row, proposal):
        return -1.0
    operator = str(proposal["operator"])
    for candidate in row.get("candidate_operators", []):
        if not isinstance(candidate, dict):
            continue
        if operator in {
            str(candidate.get("requested_operator", "")),
            str(candidate.get("executed_operator", "")),
        }:
            return float(candidate.get("reward", -1.0))
    return -1.0


def dry_run(rows: Sequence[Dict[str, Any]], group_size: int) -> Dict[str, Any]:
    rewards = []
    candidate_counts = []
    for row in rows:
        candidate_counts.append(len(row.get("candidate_operators", [])))
        rewards.append(max(
            [float(item.get("reward", -1.0)) for item in row.get("candidate_operators", []) if isinstance(item, dict)]
            or [-1.0]
        ))
    return {
        "mode": "dry_run",
        "rows": len(rows),
        "group_size": int(group_size),
        "mean_oracle_reward": sum(rewards) / max(1, len(rewards)),
        "mean_candidate_count": sum(candidate_counts) / max(1, len(candidate_counts)),
        "reward_rule": "stored external-ODE operator reward; invalid/unsafe/unregistered output=-1",
    }


def _token_ids(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _sequence_logprob(
    model: Any,
    sequences: torch.Tensor,
    prompt_length: int,
    pad_token_id: int,
) -> torch.Tensor:
    # Calling a causal LM normally materializes logits for every prompt token
    # and the full vocabulary.  That is unnecessarily large for Qwen (and can
    # exceed a local 6-8 GB GPU).  Compute hidden states once, then project
    # only the generated response positions through lm_head.
    attention = sequences.ne(pad_token_id).long()
    start = max(0, int(prompt_length) - 1)
    targets = sequences[:, start + 1:]
    target_mask = attention[:, start + 1:].bool() & targets.ne(pad_token_id)
    if hasattr(model, "model") and hasattr(model, "lm_head"):
        hidden_output = model.model(
            input_ids=sequences,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
        hidden = hidden_output.last_hidden_state[:, start:-1, :]
        logits = model.lm_head(hidden)
    else:
        output = model(input_ids=sequences, attention_mask=attention, use_cache=False)
        logits = output.logits[:, start:-1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    denominator = target_mask.sum(dim=1).clamp_min(1)
    return (token_log_probs * target_mask).sum(dim=1) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "Qwen3-1.7B")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "qwen3_operator_grpo_icrp_cloud_v1",
    )
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm-adapter-path", type=Path)
    parser.add_argument("--sample-log", type=Path)
    parser.add_argument("--readable-log", type=Path)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    source = args.input if args.input.is_absolute() else ROOT / args.input
    rows = read_rows(source.resolve(), args.split)
    if not rows:
        raise SystemExit(f"No GRPO rows found in {source} for split={args.split!r}.")
    if args.dry_run:
        print(json.dumps(dry_run(rows, args.group_size), ensure_ascii=False, indent=2))
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    adapter_path = None
    adapter_metadata: Dict[str, Any] = {}
    if args.llm_adapter_path:
        adapter_path = args.llm_adapter_path if args.llm_adapter_path.is_absolute() else ROOT / args.llm_adapter_path
        payload = torch.load(str(adapter_path.resolve()), map_location="cpu", weights_only=True)
        adapter_metadata = dict(payload.get("metadata", {}))
    rank = int(args.rank if args.rank is not None else adapter_metadata.get("rank", 8))
    alpha = float(args.alpha if args.alpha is not None else adapter_metadata.get("alpha", 16.0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True,
        dtype=dtype, low_cpu_mem_usage=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replaced = _replace_target_modules(
        model, rank, alpha, 0.05, ("q_proj", "v_proj")
    )
    if not replaced:
        raise RuntimeError("No Qwen q_proj/v_proj modules were replaced for LoRA.")
    if args.llm_adapter_path:
        load_lora_adapter(model, adapter_path.resolve())
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    history = []
    for step in range(max(1, args.steps)):
        row = rows[step % len(rows)]
        prompt = prompt_for(row)
        encoded = generation_inputs(tokenizer, prompt, args.max_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=max(2, args.group_size),
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
            )
        prompt_length = int(input_ids.shape[-1])
        texts = tokenizer.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
        rewards = torch.tensor(
            [reward_for(row, proposal_from_text(text)) for text in texts],
            dtype=torch.float32,
            device=device,
        )
        advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        logprob = _sequence_logprob(
            model, generated, prompt_length, int(tokenizer.pad_token_id)
        )
        loss = -(advantages.detach() * logprob).mean()
        loss.backward()
        clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        record = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "mean_reward": float(rewards.mean().detach().cpu()),
            "max_reward": float(rewards.max().detach().cpu()),
            "valid_output_fraction": float(sum(operator_from_text(text) is not None for text in texts) / len(texts)),
        }
        history.append(record)
        if step == 0 or (step + 1) % max(1, args.log_every) == 0:
            print(json.dumps(record, ensure_ascii=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": str(model_path),
        "algorithm": "minimal_on_policy_group_relative_policy_gradient",
        "group_size": args.group_size,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "rank": rank,
        "alpha": alpha,
        "target_modules": ["q_proj", "v_proj"],
        "replaced_modules": replaced,
        "split": args.split,
        "registry_policy": "verified_icrp_only",
        "history_tail": history[-10:],
        "clinical_use": False,
    }
    torch.save({"adapter_state": _adapter_state(model), "metadata": metadata}, output_dir / "adapter.pt")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "adapter": str(output_dir / "adapter.pt"), "metadata": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
