"""Small auditable smoke runner for the canonical fixed-model pipeline.

It intentionally uses the deterministic planner by default so a user can
inspect registry provenance, ICRP retrieval, ODE predictions, and numerical
optimization without loading Qwen.  Pass ``--planner mlp_llm_operator`` only
after the strict registry smoke test succeeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .compartment_ode import ODEModelRegistry
    from .multi_nuclide_llm_optimizer import (
        AdaptiveOperatorPlanner,
        DEFAULT_LLM_GATE_PATH,
        DEFAULT_MODEL_PATH,
        LLMGatedOperatorPlanner,
        QwenClient,
        StructuredKnowledgeBase,
        load_case,
        run_outer_loop,
        validate_case,
    )
except ImportError:
    from compartment_ode import ODEModelRegistry
    from multi_nuclide_llm_optimizer import (
        AdaptiveOperatorPlanner,
        DEFAULT_LLM_GATE_PATH,
        DEFAULT_MODEL_PATH,
        LLMGatedOperatorPlanner,
        QwenClient,
        StructuredKnowledgeBase,
        load_case,
        run_outer_loop,
        validate_case,
    )


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, default=r"E:\project\medicine\data\synthetic_bioassay_icrp_v1\inputs\test\ode-synthetic-000255.json")
    parser.add_argument("--registry-dir", type=Path, default=ROOT / "data" / "icrp_model_registry_v1")
    parser.add_argument("--icrp-dir", type=Path, default=ROOT / "data" / "icrp_knowledge_base_v2")
    parser.add_argument("--planner", choices=("heuristic", "mlp_llm_operator"), default="mlp_llm_operator")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--llm-gate-path", type=Path, default=DEFAULT_LLM_GATE_PATH)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "temp" / "debug_pipeline_result.json")
    args = parser.parse_args()

    registry_dir = resolve(args.registry_dir)
    icrp_dir = resolve(args.icrp_dir)
    if not registry_dir.is_dir():
        raise SystemExit(f"Verified ICRP registry not found: {registry_dir}")
    if not icrp_dir.is_dir():
        raise SystemExit(f"ICRP knowledge base not found: {icrp_dir}")
    registry = ODEModelRegistry.from_csv_dir(registry_dir, require_verified=True)
    case = load_case(resolve(args.case_file))
    validate_case(case, ode_registry=registry)
    knowledge = StructuredKnowledgeBase(
        dataset_root=resolve(Path("data/synthetic_bioassay_icrp_v1")),
        ode_registry=registry,
        icrp_root=icrp_dir,
    )
    if args.planner == "heuristic":
        planner = AdaptiveOperatorPlanner(knowledge)
    else:
        model_path = resolve(args.model_path)
        gate_path = resolve(args.llm_gate_path)
        if not model_path.is_dir():
            raise SystemExit(f"Qwen model directory not found: {model_path}")
        if not gate_path.is_file():
            raise SystemExit(f"MLP gate checkpoint not found: {gate_path}")
        planner = LLMGatedOperatorPlanner(
            knowledge,
            qwen_client=QwenClient(model_path),
            gate_checkpoint_path=gate_path,
        )
    result = run_outer_loop(
        case=case,
        knowledge_base=knowledge,
        planner=planner,
        rounds=max(1, args.rounds),
        seed=17,
        dataset_root=knowledge.dataset_root,
        forward_model_kind="ode",
    )
    result["debug_provenance"] = {
        "registry_dir": str(registry_dir),
        "registry_models": [
            {
                "nuclide": nuclide,
                "model_id": model_id,
                "metadata": model.metadata,
            }
            for (nuclide, model_id), model in sorted(registry.models.items())
        ],
        "icrp_dir": str(icrp_dir),
        "planner": args.planner,
        "qwen_can_modify_ode": False,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result.get("final_numerical_result", {}).get("status"),
        "output": str(output),
        "models": len(registry.models),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
