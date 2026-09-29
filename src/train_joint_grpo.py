"""Joint gate/LoRA policy-gradient smoke trainer.

This is the first joint-training baseline for the project.  The gate is
updated on every logged state from the measured counterfactual value of CALL
versus the deterministic fallback.  The Qwen LoRA policy is updated only when
a sampled response passes the proposal auditor.  The ICRP registry and ODE
executor remain frozen; this trainer does not generate or modify medical
models.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .multi_nuclide_llm_optimizer import _extract_json_object
    from .operator_policy import FEATURE_NAMES, GATE_ACTIONS, OperatorMLP, state_to_features
    from .train_qwen_grpo import (
        _proposal_is_valid,
        _sequence_logprob,
        generation_inputs,
        prompt_for,
        proposal_from_text,
        reward_for,
    )
    from .train_qwen_lora import _adapter_state, _replace_target_modules, load_lora_adapter
except ImportError:
    from multi_nuclide_llm_optimizer import _extract_json_object
    from operator_policy import FEATURE_NAMES, GATE_ACTIONS, OperatorMLP, state_to_features
    from train_qwen_grpo import (
        _proposal_is_valid,
        _sequence_logprob,
        generation_inputs,
        prompt_for,
        proposal_from_text,
        reward_for,
    )
    from train_qwen_lora import _adapter_state, _replace_target_modules, load_lora_adapter


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path, split: str) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            if split == "all" or str(row.get("split")) == split:
                rows.append(row)
    return rows


def counterfactual_gate_target(row: Dict[str, Any], temperature: float) -> float:
    candidates = [
        item for item in row.get("candidate_operators", [])
        if isinstance(item, dict)
        and not str(item.get("executed_operator", "")).startswith("STOP_")
    ]
    call_reward = max(
        (float(item.get("reward", -1.0)) for item in candidates),
        default=-1.0,
    )
    skip_reward = float(row.get("skip_baseline", {}).get("reward", 0.0))
    advantage = (call_reward - skip_reward) / max(float(temperature), 1e-6)
    advantage = max(-30.0, min(30.0, advantage))
    return float(torch.sigmoid(torch.tensor(advantage)).item())


def load_gate(path: Path) -> tuple[OperatorMLP, Dict[str, Any]]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Gate checkpoint feature schema does not match this code version.")
    if tuple(checkpoint.get("operators", ())) != GATE_ACTIONS:
        raise ValueError("Gate checkpoint must use SKIP_LLM/CALL_LLM.")
    model = OperatorMLP(
        int(checkpoint["input_size"]),
        int(checkpoint["hidden_size"]),
        len(GATE_ACTIONS),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = {
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": checkpoint["feature_mean"],
        "feature_std": checkpoint["feature_std"],
        "input_size": checkpoint["input_size"],
        "hidden_size": checkpoint["hidden_size"],
        "operators": list(GATE_ACTIONS),
    }
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "Qwen3-1.7B")
    parser.add_argument("--gate-path", type=Path, default=ROOT / "models" / "llm_gate_mlp_icrp.pt")
    parser.add_argument("--llm-adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "joint_policy_icrp_cloud_v1")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-4)
    parser.add_argument("--llm-learning-rate", type=float, default=1e-6)
    parser.add_argument("--gate-temperature", type=float, default=0.05)
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="LoRA rank. Defaults to the loaded adapter metadata, otherwise 8.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="LoRA alpha. Defaults to the loaded adapter metadata, otherwise 16.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--llm-exploration",
        type=float,
        default=0.10,
        help="Minimum probability of sampling a Qwen rollout, even when gate prefers skip.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for Qwen rollout generation.",
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--validation-input", type=Path,
        help="Optional preference JSONL containing a validation split for checkpoint selection.",
    )
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--validation-limit", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--sample-log", type=Path)
    parser.add_argument("--readable-log", type=Path)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    gate_path = args.gate_path if args.gate_path.is_absolute() else ROOT / args.gate_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows = read_rows(input_path.resolve(), args.split)
    if not rows:
        raise SystemExit(f"No rows found for split={args.split!r}.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sample_log_path = args.sample_log or (output_dir / "samples.jsonl")
    readable_log_path = args.readable_log or (output_dir / "samples.txt")
    if not sample_log_path.is_absolute():
        sample_log_path = ROOT / sample_log_path
    if not readable_log_path.is_absolute():
        readable_log_path = ROOT / readable_log_path
    sample_log_path.parent.mkdir(parents=True, exist_ok=True)
    readable_log_path.parent.mkdir(parents=True, exist_ok=True)
    sample_log = sample_log_path.open("w", encoding="utf-8")
    readable_log = readable_log_path.open("w", encoding="utf-8")
    validation_rows: List[Dict[str, Any]] = []
    if args.validation_input:
        validation_path = args.validation_input if args.validation_input.is_absolute() else ROOT / args.validation_input
        validation_rows = read_rows(validation_path.resolve(), args.validation_split)
        validation_rows = validation_rows[: max(0, args.validation_limit)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    gate, gate_metadata = load_gate(gate_path.resolve())
    gate.to(device)
    gate_optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_learning_rate)
    feature_mean = torch.tensor(gate_metadata["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.tensor(gate_metadata["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter_metadata: Dict[str, Any] = {}
    if args.llm_adapter_path:
        adapter_path = args.llm_adapter_path if args.llm_adapter_path.is_absolute() else ROOT / args.llm_adapter_path
        adapter_checkpoint = torch.load(str(adapter_path.resolve()), map_location="cpu", weights_only=True)
        adapter_metadata = dict(adapter_checkpoint.get("metadata", {}))
    rank = int(args.rank if args.rank is not None else adapter_metadata.get("rank", 8))
    alpha = float(args.alpha if args.alpha is not None else adapter_metadata.get("alpha", 16.0))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
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
    llm_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    llm_optimizer = torch.optim.AdamW(llm_parameters, lr=args.llm_learning_rate)

    history = []
    best_validation_reward = float("-inf")
    best_validation_step = None
    best_adapter_state = None
    best_gate_state = None
    bad_validation_rounds = 0
    for step in range(max(1, args.steps)):
        row = rows[step % len(rows)]
        features = torch.from_numpy(state_to_features(row["state"])).to(device)
        features = (features - feature_mean) / feature_std
        gate_logits = gate(features.unsqueeze(0))
        gate_call_probability = float(
            torch.softmax(gate_logits.detach(), dim=-1)[
                :, GATE_ACTIONS.index("CALL_LLM")
            ].item()
        )
        target_call_probability = counterfactual_gate_target(row, args.gate_temperature)
        target = torch.tensor(
            [[1.0 - target_call_probability, target_call_probability]],
            dtype=torch.float32,
            device=device,
        )
        gate_loss = -(target * F.log_softmax(gate_logits, dim=-1)).sum(dim=-1).mean()
        gate_optimizer.zero_grad(set_to_none=True)
        gate_loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        gate_optimizer.step()

        prompt = prompt_for(row)
        should_call_llm = random.random() < max(
            float(args.llm_exploration), gate_call_probability
        )
        generated = None
        texts: List[str] = []
        proposals: List[Dict[str, Any] | None] = []
        valid: List[bool] = []
        rewards = torch.empty(0, dtype=torch.float32, device=device)
        prompt_length = 0
        if should_call_llm:
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
            texts = tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )
            proposals = [proposal_from_text(text) for text in texts]
            valid = [
                proposal is not None and _proposal_is_valid(row, proposal)
                for proposal in proposals
            ]
            rewards = torch.tensor(
                [reward_for(row, proposal) for proposal in proposals],
                dtype=torch.float32,
                device=device,
            )
        valid_count = int(sum(valid))
        reported_rewards = rewards if should_call_llm else torch.tensor(
            [float(row.get("skip_baseline", {}).get("reward", 0.0))],
            dtype=torch.float32,
            device=device,
        )
        llm_loss_value = 0.0
        if should_call_llm and len(texts) >= 2:
            # Invalid outputs are never sent to the ODE executor, but they do
            # receive a negative policy reward so the model can learn JSON and
            # whitelist compliance. This is essential during early training.
            advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-3)
            llm_optimizer.zero_grad(set_to_none=True)
            logprob = _sequence_logprob(
                model,
                generated,
                prompt_length,
                int(tokenizer.pad_token_id),
            )
            llm_loss = -(advantages.detach() * logprob).mean()
            llm_loss.backward()
            torch.nn.utils.clip_grad_norm_(llm_parameters, 1.0)
            llm_optimizer.step()
            llm_loss_value = float(llm_loss.detach().cpu())
        record = {
            "step": step + 1,
            "case_id": row.get("case_id"),
            "split": row.get("split"),
            "state_type": row.get("state_type"),
            "gate_call_probability": gate_call_probability,
            "gate_loss": float(gate_loss.detach().cpu()),
            "gate_target_call_probability": float(target_call_probability),
            "llm_called": bool(should_call_llm),
            "llm_loss": llm_loss_value,
            "mean_reward": float(reported_rewards.mean().detach().cpu()),
            "max_reward": float(reported_rewards.max().detach().cpu()),
            "valid_output_fraction": float(valid_count / max(1, len(valid))),
            "prompt": prompt,
            "samples": [
                {
                    "raw_output": text,
                    "proposal": proposal,
                    "valid": bool(is_valid),
                    "reward": float(reward),
                }
                for text, proposal, is_valid, reward in zip(
                    texts, proposals, valid, rewards.detach().cpu().tolist()
                )
            ],
        }
        history.append(record)
        sample_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        sample_log.flush()
        if (step + 1) % max(1, args.log_every) == 0:
            readable_log.write("=" * 88 + "\n")
            readable_log.write(
                f"step={record['step']} case={record['case_id']} "
                f"split={record['split']} state={record['state_type']}\n"
            )
            readable_log.write(
                f"gate_call_probability={record['gate_call_probability']:.6f} "
                f"target={record['gate_target_call_probability']:.6f} "
                f"gate_loss={record['gate_loss']:.6f} "
                f"llm_loss={record['llm_loss']:.6f}\n"
            )
            readable_log.write("PROMPT:\n" + prompt + "\n")
            for index, sample in enumerate(record["samples"]):
                readable_log.write(
                    f"\nSAMPLE {index + 1}: valid={sample['valid']} "
                    f"reward={sample['reward']:.6f}\n"
                )
                readable_log.write(str(sample["raw_output"]) + "\n")
                readable_log.write(
                    "PARSED:\n"
                    + json.dumps(sample["proposal"], ensure_ascii=False, indent=2)
                    + "\n"
                )
            readable_log.flush()
        print(json.dumps({
            key: record[key]
            for key in (
                "step",
                "case_id",
                "state_type",
                "llm_called",
                "gate_call_probability",
                "gate_loss",
                "llm_loss",
                "mean_reward",
                "valid_output_fraction",
            )
            if key in record
        }, ensure_ascii=False))

        if validation_rows and (step + 1) % max(1, args.eval_every) == 0:
            # Validation is intentionally sampled without gradient updates.
            model.eval()
            validation_records = []
            for validation_row in validation_rows:
                validation_prompt = prompt_for(validation_row)
                validation_encoded = generation_inputs(
                    tokenizer, validation_prompt, args.max_length
                )
                validation_input_ids = validation_encoded["input_ids"].to(device)
                validation_attention = validation_encoded["attention_mask"].to(device)
                with torch.no_grad():
                    validation_generated = model.generate(
                        input_ids=validation_input_ids,
                        attention_mask=validation_attention,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=False,
                    )
                validation_prompt_length = int(validation_input_ids.shape[-1])
                validation_text = tokenizer.decode(
                    validation_generated[0, validation_prompt_length:],
                    skip_special_tokens=True,
                )
                validation_proposal = proposal_from_text(validation_text)
                validation_records.append({
                    "valid": bool(
                        validation_proposal is not None
                        and _proposal_is_valid(validation_row, validation_proposal)
                    ),
                    "reward": reward_for(validation_row, validation_proposal),
                })
            validation_reward = sum(item["reward"] for item in validation_records) / max(1, len(validation_records))
            validation_valid = sum(item["valid"] for item in validation_records) / max(1, len(validation_records))
            validation_record = {
                "step": step + 1,
                "validation_mean_reward": validation_reward,
                "validation_valid_fraction": validation_valid,
                "validation_count": len(validation_records),
            }
            history.append(validation_record)
            sample_log.write(json.dumps(validation_record, ensure_ascii=False) + "\n")
            sample_log.flush()
            if validation_reward > best_validation_reward:
                best_validation_reward = validation_reward
                best_validation_step = step + 1
                best_adapter_state = _adapter_state(model)
                best_gate_state = {
                    key: value.detach().cpu().clone()
                    for key, value in gate.state_dict().items()
                }
                bad_validation_rounds = 0
            else:
                bad_validation_rounds += 1
            readable_log.write(
                f"VALIDATION step={step + 1} mean_reward={validation_reward:.6f} "
                f"valid_fraction={validation_valid:.6f}\n"
            )
            readable_log.flush()
            model.train()
            if bad_validation_rounds >= max(1, args.patience):
                break

    sample_log.close()
    readable_log.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_adapter_state is not None:
        model_state = best_adapter_state
    else:
        model_state = _adapter_state(model)
    if best_gate_state is not None:
        gate_state = best_gate_state
    else:
        gate_state = {key: value.detach().cpu() for key, value in gate.state_dict().items()}
    gate_checkpoint = {
        "model_state_dict": gate_state,
        **gate_metadata,
        "training_rows": len(rows),
        "algorithm": "joint_counterfactual_gate_plus_group_relative_lora",
        "history": history,
        "sample_log": str(sample_log_path),
        "readable_log": str(readable_log_path),
        "clinical_use": False,
    }
    torch.save(gate_checkpoint, output_dir / "llm_gate.pt")
    torch.save(
        {
            "adapter_state": model_state,
            "metadata": {
                "base_model": str(model_path),
                "rank": rank,
                "alpha": alpha,
                "target_modules": ["q_proj", "v_proj"],
                "replaced_modules": replaced,
                "algorithm": "joint_counterfactual_gate_plus_group_relative_lora",
                "best_validation_reward": best_validation_reward,
                "best_validation_step": best_validation_step,
                "history": history,
                "clinical_use": False,
            },
        },
        output_dir / "adapter.pt",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "history": history,
                "sample_log": str(sample_log_path),
                "readable_log": str(readable_log_path),
                "clinical_use": False,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "gate_checkpoint": str(output_dir / "llm_gate.pt"),
        "adapter": str(output_dir / "adapter.pt"),
        "sample_log": str(sample_log_path),
        "readable_log": str(readable_log_path),
        "history": history,
        "device": device,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
