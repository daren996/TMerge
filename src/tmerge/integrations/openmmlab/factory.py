"""Registry hooks for optional OpenMMLab integrations."""

from __future__ import annotations

from typing import Any

from tmerge.integrations.openmmlab.native_ops import (
    MMDetDetector,
    MMTrackingMOTOperator,
    MMTrackingSortOperator,
    TorchReIdExtractor,
)


def register_openmmlab(registry: Any) -> None:
    registry.operators.update(
        {
            "integration.openmmlab.mmdet_detector": MMDetDetector.from_config,
            "integration.openmmlab.mmdet_detector_with_features": _build_mmdet_detector_with_features,
            "integration.openmmlab.mmtracking_mot": MMTrackingMOTOperator.from_config,
            "integration.openmmlab.mmtracking_sort": _build_mmtracking_sort,
            "integration.openmmlab.mmtracking_deepsort": _build_mmtracking_deepsort,
            "integration.openmmlab.mmtracking_tracktor": _build_mmtracking_tracktor,
            "integration.reid.torchreid_extractor": TorchReIdExtractor.from_config,
        }
    )


def _build_mmdet_detector_with_features(config: dict[str, Any]) -> MMDetDetector:
    enriched = dict(config)
    enriched["with_features"] = True
    return MMDetDetector.from_config(enriched)


def _build_mmtracking_sort(config: dict[str, Any]) -> MMTrackingSortOperator:
    enriched = dict(config)
    enriched["tracker_kind"] = "sort"
    return MMTrackingSortOperator.from_config(enriched)


def _build_mmtracking_deepsort(config: dict[str, Any]) -> MMTrackingSortOperator:
    enriched = dict(config)
    enriched["tracker_kind"] = "deepsort"
    return MMTrackingSortOperator.from_config(enriched)


def _build_mmtracking_tracktor(config: dict[str, Any]) -> MMTrackingSortOperator:
    enriched = dict(config)
    enriched["tracker_kind"] = "tracktor"
    return MMTrackingSortOperator.from_config(enriched)
