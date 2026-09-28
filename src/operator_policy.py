"""Shared feature extraction and neural policy for optimization operators."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "log1p_objective",
    "log1p_fit",
    "failed_nuclides",
    "log1p_function_evaluations",
    "log1p_optimizer_iterations",
    "log1p_forward_predict_calls",
    "log1p_numerical_wall_time",
    "nuclide_count",
    "mean_log1p_best_loss",
    "max_log1p_best_loss",
    "mean_log1p_weighted_loss",
    "mean_log1p_relative_model_gap",
    "ambiguous_model_fraction",
    "intake_boundary_fraction",
    "time_boundary_fraction",
    "mean_log10_intake",
    "mean_intake_time_d",
    "mean_censored_fraction",
    "mean_log1p_observation_count",
    "mean_log1p_uncensored_count",
    "identifiability_available_fraction",
    "mean_log1p_identifiability_condition",
    "mean_identifiability_rank_fraction",
    "recommended_extension_fraction",
    "candidate_evaluated_fraction",
    "candidate_improved_fraction",
)

GATE_ACTIONS = ("SKIP_LLM", "CALL_LLM")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def state_to_features(state: Dict[str, Any]) -> np.ndarray:
    """Convert the fixed-schema numerical state to the MLP input vector."""
    rows = [row for row in state.get("per_nuclide", []) if row.get("status") == "ok"]
    best_losses = [max(0.0, _finite(row.get("best_loss"))) for row in rows]
    weighted_losses = [max(0.0, _finite(row.get("weighted_loss"))) for row in rows]
    relative_gaps = [
        max(0.0, _finite(row.get("relative_model_gap")))
        for row in rows
        if row.get("relative_model_gap") is not None
    ]
    diagnostics = [
        row.get("identifiability") for row in rows
        if isinstance(row.get("identifiability"), dict)
        and row.get("identifiability", {}).get("status") == "ok"
    ]
    features = [
        math.log1p(max(0.0, _finite(state.get("objective_score")))),
        math.log1p(max(0.0, _finite(state.get("fit_score")))),
        _finite(state.get("failed_nuclide_count")),
        math.log1p(max(0.0, _finite(state.get("function_evaluations")))),
        math.log1p(max(0.0, _finite(state.get("optimizer_iterations")))),
        math.log1p(max(0.0, _finite(state.get("forward_predict_calls")))),
        math.log1p(max(0.0, _finite(state.get("numerical_wall_time_s")))),
        float(len(rows)),
        _mean([math.log1p(value) for value in best_losses]),
        max([math.log1p(value) for value in best_losses], default=0.0),
        _mean([math.log1p(value) for value in weighted_losses]),
        _mean([math.log1p(value) for value in relative_gaps]),
        _mean([1.0 if value <= 0.10 else 0.0 for value in relative_gaps]),
        _mean([1.0 if row.get("intake_boundary_hit") else 0.0 for row in rows]),
        _mean([1.0 if row.get("time_boundary_hit") else 0.0 for row in rows]),
        _mean([_finite(row.get("log10_intake")) for row in rows]),
        _mean([_finite(row.get("intake_time_d")) for row in rows]),
        _mean([_finite(row.get("censored_fraction"), 1.0) for row in rows]),
        _mean([math.log1p(max(0.0, _finite(row.get("observation_count")))) for row in rows]),
        _mean([
            math.log1p(max(0.0, _finite(row.get("uncensored_observation_count"))))
            for row in rows
        ]),
        float(len(diagnostics)) / max(1, len(rows)),
        _mean([
            math.log1p(min(1e6, max(0.0, _finite(item.get("condition_number"), 1e6))))
            for item in diagnostics
        ]),
        _mean([
            _finite(item.get("rank")) / max(1, len(item.get("parameter_names", [])))
            for item in diagnostics
        ]),
        _mean([
            1.0 if len(item.get("recommended_parameter_names", [])) > 2 else 0.0
            for item in diagnostics
        ]),
        _mean([1.0 if int(row.get("candidate_point_count", 0)) > 0 else 0.0 for row in rows]),
        _mean([1.0 if row.get("candidate_improved_incumbent") is True else 0.0 for row in rows]),
    ]
    return np.asarray(features, dtype=np.float32)


class OperatorMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class OperatorMLPPredictor:
    """Load a trained policy and rank operators for one numerical state."""

    def __init__(self, checkpoint_path: Path):
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        feature_names = tuple(checkpoint.get("feature_names", ()))
        if feature_names != FEATURE_NAMES:
            raise ValueError("MLP checkpoint feature schema does not match this code version.")
        self.operators = tuple(str(item) for item in checkpoint["operators"])
        self.mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        self.model = OperatorMLP(
            int(checkpoint["input_size"]),
            int(checkpoint["hidden_size"]),
            len(self.operators),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def rank(self, state: Dict[str, Any]) -> List[Tuple[str, float]]:
        features = (state_to_features(state) - self.mean) / self.std
        tensor = torch.from_numpy(features).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].numpy()
        ranked = sorted(
            zip(self.operators, (float(value) for value in probabilities)),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked


class LLMGatePredictor(OperatorMLPPredictor):
    """Binary policy deciding whether an expensive LLM decision is needed."""

    def __init__(self, checkpoint_path: Path):
        super().__init__(checkpoint_path)
        if self.operators != GATE_ACTIONS:
            raise ValueError(
                "LLM gate checkpoint must use SKIP_LLM/CALL_LLM outputs."
            )

    def call_probability(self, state: Dict[str, Any]) -> float:
        return dict(self.rank(state))["CALL_LLM"]
