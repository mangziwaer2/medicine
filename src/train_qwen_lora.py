"""Train a small LoRA adapter for the Qwen operator planner.

This intentionally has no PEFT dependency.  The base Qwen checkpoint stays
frozen and only low-rank A/B matrices on attention projections are optimized.
The input is the exported operator SFT JSONL; validation rows are never used
for gradient updates.  It trains optimization-policy output only and cannot
modify the verified ICRP forward-model registry.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.normal_(self.lora_A, std=0.02)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        lora_a = self.lora_A.to(dtype=x.dtype)
        lora_b = self.lora_B.to(dtype=x.dtype)
        update = self.dropout(x) @ lora_a.t() @ lora_b.t()
        return result + update.to(result.dtype) * self.scaling


def _replace_target_modules(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_suffixes: Sequence[str],
    prefix: str = "",
) -> List[str]:
    replaced: List[str] = []
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and any(
            full_name.endswith(suffix) for suffix in target_suffixes
        ):
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            replaced.append(full_name)
        else:
            replaced.extend(
                _replace_target_modules(
                    child, rank, alpha, dropout, target_suffixes, full_name
                )
            )
    return replaced


def _adapter_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def load_lora_adapter(model: nn.Module, path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("adapter_state", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_adapter = [
        name for name in missing if "lora_A" in name or "lora_B" in name
    ]
    if missing_adapter:
        raise RuntimeError(
            f"Missing LoRA tensors while loading {path}: {missing_adapter[:5]}"
        )
    return {
        "metadata": payload.get("metadata", {}),
        "missing_base_parameters": len(missing) - len(missing_adapter),
        "unexpected_keys": unexpected,
    }


class SFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any, max_length: int, split: str):
        self.rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        self.rows = [row for row in self.rows if str(row.get("split")) == split]
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        messages = self.rows[index]["messages"]
        prompt_messages = messages[:-1]
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        try:
            full = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            full = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        def ids(value: Any) -> List[int]:
            if hasattr(value, "get") and value.get("input_ids") is not None:
                value = value["input_ids"]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            # Some tokenizer versions wrap a single sequence in a batch.
            if value and isinstance(value[0], list):
                value = value[0]
            return [int(item) for item in value]

        prompt_ids = ids(prompt)
        full_ids = ids(full)
        prompt_len = min(len(prompt_ids), len(full_ids))
        response_ids = full_ids[prompt_len:]
        if len(response_ids) >= self.max_length:
            response_ids = response_ids[: self.max_length]
            prompt_ids = []
        else:
            available_prompt = self.max_length - len(response_ids)
            if len(prompt_ids) > available_prompt:
                head = max(1, int(available_prompt * 0.68))
                tail = max(1, available_prompt - head)
                prompt_ids = prompt_ids[:head] + prompt_ids[-tail:]
        full_ids = prompt_ids + response_ids
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + response_ids
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _collate(batch: Sequence[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    width = max(item["input_ids"].numel() for item in batch)
    inputs, labels, masks = [], [], []
    for item in batch:
        size = item["input_ids"].numel()
        pad = width - size
        inputs.append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=pad_id))
        labels.append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
        masks.append(torch.tensor([1] * size + [0] * pad, dtype=torch.long))
    return {
        "input_ids": torch.stack(inputs),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(masks),
    }


def _loss(model: nn.Module, batch: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
    output = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    return output.loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "Qwen3-1.7B")
    parser.add_argument("--sft-path", type=Path, default=ROOT / "temp" / "operator_oracle_v3_train_validation_48.sft.jsonl")
    parser.add_argument(
        "--validation-sft-path",
        type=Path,
        help="Optional separate JSONL source for validation examples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "qwen3_operator_lora_icrp_cloud_v1",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke", action="store_true", help="One train and one validation step only.")
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    sft_path = args.sft_path if args.sft_path.is_absolute() else ROOT / args.sft_path
    validation_sft_path = None
    if args.validation_sft_path:
        validation_sft_path = args.validation_sft_path if args.validation_sft_path.is_absolute() else ROOT / args.validation_sft_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True,
        dtype=dtype, low_cpu_mem_usage=True,
    )
    # Freeze the complete base model before inserting adapters. Only LoRA A/B
    # tensors are allowed into the optimizer state.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    targets = ("q_proj", "v_proj")
    replaced = _replace_target_modules(model, args.rank, args.alpha, args.dropout, targets)
    if not replaced:
        raise RuntimeError("No target linear modules were replaced for LoRA.")
    # Replacement creates fresh A/B tensors, so move only after insertion.
    model.to(device)
    train_data = SFTDataset(sft_path, tokenizer, args.max_length, "train")
    valid_data = SFTDataset(validation_sft_path or sft_path, tokenizer, args.max_length, "validation")
    if not train_data or not valid_data:
        raise RuntimeError(f"Need non-empty train/validation splits, got {len(train_data)}/{len(valid_data)}")
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: _collate(x, tokenizer.pad_token_id))
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, collate_fn=lambda x: _collate(x, tokenizer.pad_token_id))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history: List[Dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(max(1, args.epochs)):
        model.train()
        train_losses: List[float] = []
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if args.smoke:
                break
        model.eval()
        valid_losses: List[float] = []
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                valid_losses.append(float(_loss(model, batch, device).detach().cpu()))
                if args.smoke:
                    break
        row = {"epoch": epoch + 1, "train_loss": sum(train_losses) / len(train_losses), "validation_loss": sum(valid_losses) / len(valid_losses)}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if row["validation_loss"] < best_validation_loss:
            best_validation_loss = row["validation_loss"]
            best_state = _adapter_state(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if args.smoke:
            break
        if bad_epochs >= max(1, args.patience):
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = _adapter_state(model)
    metadata = {
        "base_model": str(model_path), "rank": args.rank, "alpha": args.alpha,
        "dropout": args.dropout, "target_modules": list(targets),
        "replaced_modules": replaced, "train_count": len(train_data),
        "validation_count": len(valid_data), "history": history,
        "best_validation_loss": best_validation_loss,
        "clinical_use": False,
    }
    torch.save({"adapter_state": best_state, "metadata": metadata}, output_dir / "adapter.pt")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "adapter": str(output_dir / "adapter.pt"), "trainable_parameters": sum(p.numel() for p in trainable), "device": device, "history": history}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
