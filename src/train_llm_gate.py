"""Train the binary MLP that decides whether to invoke the LLM planner.

Counterfactual labels are converted as follows:

- STOP_CONVERGED / STOP_UNIDENTIFIABLE -> SKIP_LLM
- any executable recovery/search operator -> CALL_LLM

The gate never selects the optimization operator. That decision belongs to
the LLM when the gate outputs CALL_LLM.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .operator_policy import GATE_ACTIONS, FEATURE_NAMES, OperatorMLP, state_to_features
except ImportError:
    from operator_policy import GATE_ACTIONS, FEATURE_NAMES, OperatorMLP, state_to_features


ROOT = Path(__file__).resolve().parents[1]


def gate_label(operator: str) -> str:
    return "SKIP_LLM" if str(operator).startswith("STOP_") else "CALL_LLM"


def read_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("state") and row.get("operator"):
                    rows.append(row)
    return rows


def make_arrays(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    action_to_index = {name: index for index, name in enumerate(GATE_ACTIONS)}
    features = np.stack([state_to_features(row["state"]) for row in rows])
    labels = np.asarray(
        [action_to_index[gate_label(str(row["operator"]))] for row in rows],
        dtype=np.int64,
    )
    return features, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the binary LLM-call gate.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--validation-input", type=Path, action="append")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "llm_gate_mlp_icrp.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = read_rows([path.resolve() for path in args.input])
    if len(train_rows) < 2:
        raise SystemExit("At least two valid trajectory rows are required.")
    if args.validation_input:
        validation_rows = read_rows([path.resolve() for path in args.validation_input])
    else:
        random.shuffle(train_rows)
        validation_count = max(1, int(len(train_rows) * args.validation_ratio))
        validation_rows = train_rows[:validation_count]
        train_rows = train_rows[validation_count:]
    if not train_rows or not validation_rows:
        raise SystemExit("Both training and validation rows are required.")

    x_train, y_train = make_arrays(train_rows)
    x_validation, y_validation = make_arrays(validation_rows)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_validation = (x_validation - mean) / std

    model = OperatorMLP(len(FEATURE_NAMES), args.hidden_size, len(GATE_ACTIONS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=min(args.batch_size, len(x_train)),
        shuffle=True,
    )
    best_accuracy = -1.0
    best_state = None
    for _epoch in range(max(1, args.epochs)):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad()
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predictions = model(torch.from_numpy(x_validation)).argmax(dim=1).numpy()
        accuracy = float(np.mean(predictions == y_validation))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    output = args.output if args.output.is_absolute() else (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    train_counts = {
        action: sum(gate_label(str(row["operator"])) == action for row in train_rows)
        for action in GATE_ACTIONS
    }
    torch.save({
        "model_state_dict": best_state,
        "input_size": len(FEATURE_NAMES),
        "hidden_size": args.hidden_size,
        "operators": list(GATE_ACTIONS),
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "validation_accuracy": best_accuracy,
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training_label_counts": train_counts,
        "seed": args.seed,
    }, output)
    print(json.dumps({
        "output": str(output),
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training_label_counts": train_counts,
        "validation_accuracy": best_accuracy,
        "warning": "The current dataset is small; report downstream cost and quality, not accuracy alone.",
    }, indent=2))


if __name__ == "__main__":
    main()
