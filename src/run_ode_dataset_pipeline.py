"""Run the canonical ODE-only multi-nuclide inversion pipeline.

This is the user-facing entry point for the fixed-model project direction:

    observation case -> registered ODE model candidates
    -> MLP gate -> optional Qwen operator/model selection
    -> audited numerical inversion -> result

The runner intentionally has no public ``mode`` or ``forward-model`` switch.
The active dataset and the verified ICRP ODE registry are one fixed
configuration. Hidden labels are never exposed to the planner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

try:
    from .compartment_ode import ODEModelRegistry
    from .multi_nuclide_llm_optimizer import (
        AdaptiveOperatorPlanner,
        DEFAULT_LLM_GATE_PATH,
        LLMGatedOperatorPlanner,
        DEFAULT_MODEL_PATH,
        QwenClient,
        StructuredKnowledgeBase,
        load_case,
        render_human_log,
        run_outer_loop,
        evaluate_intake_estimates,
        validate_case,
    )
except ImportError:
    from compartment_ode import ODEModelRegistry
    from multi_nuclide_llm_optimizer import (
        AdaptiveOperatorPlanner,
        DEFAULT_LLM_GATE_PATH,
        LLMGatedOperatorPlanner,
        DEFAULT_MODEL_PATH,
        QwenClient,
        StructuredKnowledgeBase,
        load_case,
        render_human_log,
        run_outer_loop,
        evaluate_intake_estimates,
        validate_case,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "data" / "synthetic_bioassay_icrp_v1"
DEFAULT_REGISTRY_DIR = ROOT / "data" / "icrp_model_registry_v1"
DEFAULT_ICRP_DIR = ROOT / "data" / "icrp_knowledge_base_v2"


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _find_case_file(
    dataset_dir: Path,
    case_file: Optional[Path],
    case_id: Optional[str],
    split: str,
) -> Path:
    if case_file is not None:
        resolved = _resolve_path(case_file)
        if not resolved.is_file():
            raise SystemExit(f"Case file not found: {resolved}")
        return resolved

    split_dir = dataset_dir / "inputs" / split
    if case_id:
        resolved = split_dir / f"{case_id}.json"
        if not resolved.is_file():
            raise SystemExit(f"Case id not found in {split_dir}: {case_id}")
        return resolved

    candidates = sorted(split_dir.glob("*.json"))
    if not candidates:
        raise SystemExit(f"No case files found in {split_dir}")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed ODE-only Qwen inverse bioassay pipeline."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=DEFAULT_REGISTRY_DIR,
        help="CSV ODE registry directory (models.csv, transfers.csv, measurements.csv).",
    )
    parser.add_argument("--case-file", type=Path)
    parser.add_argument("--case-id", type=str)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Used when --case-file and --case-id are omitted.",
    )
    parser.add_argument(
        "--planner",
        choices=("heuristic", "mlp_llm_operator"),
        default="mlp_llm_operator",
        help="heuristic is the no-LLM baseline; mlp_llm_operator is the mainline.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--llm-adapter-path",
        type=Path,
        help="Optional LoRA adapter applied to the Qwen operator planner.",
    )
    parser.add_argument("--llm-gate-path", type=Path, default=DEFAULT_LLM_GATE_PATH)
    parser.add_argument(
        "--llm-gate-threshold",
        type=float,
        default=0.22,
        help=(
            "CALL_LLM probability threshold for the MLP gate. Lower values "
            "are useful for controlled planner smoke tests."
        ),
    )
    parser.add_argument(
        "--icrp-knowledge-dir",
        type=Path,
        default=DEFAULT_ICRP_DIR,
        help="Generated, page-traceable ICRP JSONL index used by the planner.",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--human-log", type=Path)
    parser.add_argument("--input-output", type=Path)
    parser.add_argument(
        "--labels-file",
        type=Path,
        help="Optional hidden-label JSONL for offline evaluation only.",
    )
    parser.add_argument(
        "--disable-planner-fallback",
        action="store_true",
        help="Fail instead of using the deterministic planner if Qwen output is invalid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive.")

    dataset_dir = _resolve_path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"ODE dataset directory not found: {dataset_dir}")
    case_path = _find_case_file(
        dataset_dir=dataset_dir,
        case_file=args.case_file,
        case_id=args.case_id,
        split=args.split,
    )
    case = load_case(case_path)

    registry_dir = _resolve_path(args.registry_dir)
    if not registry_dir.is_dir():
        raise SystemExit(f"Verified ICRP registry directory not found: {registry_dir}")
    registry = ODEModelRegistry.from_csv_dir(registry_dir, require_verified=True)
    validate_case(
        case,
        forward_model_kind="ode",
        ode_registry=registry,
    )
    icrp_dir = _resolve_path(args.icrp_knowledge_dir)
    if not icrp_dir.is_dir():
        raise SystemExit(f"ICRP knowledge-base directory not found: {icrp_dir}")
    knowledge_base = StructuredKnowledgeBase(
        dataset_root=dataset_dir,
        forward_model_kind="ode",
        ode_registry=registry,
        icrp_root=icrp_dir,
    )

    if args.planner == "heuristic":
        planner = AdaptiveOperatorPlanner(knowledge_base)
        model_path = None
        gate_path = None
        adapter_path = None
    else:
        model_path = _resolve_path(args.model_path)
        if not model_path.is_dir():
            raise SystemExit(f"Qwen model directory not found: {model_path}")
        gate_path = _resolve_path(args.llm_gate_path)
        if not gate_path.is_file():
            raise SystemExit(f"MLP gate checkpoint not found: {gate_path}")
        adapter_path = (
            _resolve_path(args.llm_adapter_path)
            if args.llm_adapter_path is not None else None
        )
        if adapter_path is not None and not adapter_path.is_file():
            raise SystemExit(f"Qwen adapter checkpoint not found: {adapter_path}")
        qwen_client = QwenClient(model_path, adapter_path=adapter_path)
        planner = LLMGatedOperatorPlanner(
            knowledge_base,
            qwen_client=qwen_client,
            gate_checkpoint_path=gate_path,
            gate_threshold=args.llm_gate_threshold,
            allow_fallback=not args.disable_planner_fallback,
        )
    import time

    start = time.perf_counter()

    result = run_outer_loop(
        case=case,
        knowledge_base=knowledge_base,
        planner=planner,
        rounds=args.rounds,
        seed=args.seed,
        dataset_root=dataset_dir,
        forward_model_kind="ode",
    )
    end = time.perf_counter()

    print(f"elapsed_seconds: {end - start:.6f}")
    result["dataset"] = {
        "path": str(dataset_dir),
        "case_file": str(case_path),
        "labels_loaded": False,
        "purpose": "synthetic observations generated only from fixed ICRP registry models",
        "registry_dir": str(registry_dir),
        "registry_source": "verified_icrp_csv",
        "planner": args.planner,
        "model_path": str(model_path) if model_path else None,
        "llm_gate_path": str(gate_path) if gate_path else None,
        "llm_gate_threshold": (
            float(args.llm_gate_threshold) if args.planner == "mlp_llm_operator" else None
        ),
        "llm_adapter_path": str(adapter_path) if adapter_path else None,
        "planner_fallback_enabled": not args.disable_planner_fallback,
        "forward_model_policy": "registered_model_ids_only",
        "icrp_context_policy": "retrieved_for_planner_context",
    }
    if args.labels_file is not None:
        labels_path = _resolve_path(args.labels_file)
    else:
        labels_path = dataset_dir / "labels.jsonl"
    if labels_path.is_file():
        labels_by_case = {}
        with labels_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    labels_by_case[str(item.get("case_id"))] = item
        label = labels_by_case.get(case.case_id)
        if label and isinstance(label.get("hidden_truth"), dict):
            result["offline_evaluation"] = evaluate_intake_estimates(
                result, label["hidden_truth"]
            )
            result["dataset"]["labels_loaded"] = True
            result["dataset"]["labels_file"] = str(labels_path)

    payload = json.dumps(result, ensure_ascii=True, indent=2)
    if args.output:
        output_path = _resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if args.input_output:
        input_path = _resolve_path(args.input_output)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(
            json.dumps(case.to_dict(), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.human_log:
        human_log_path = _resolve_path(args.human_log)
        human_log_path.parent.mkdir(parents=True, exist_ok=True)
        human_log_path.write_text(render_human_log(result), encoding="utf-8")

    print(payload)


if __name__ == "__main__":
    main()
