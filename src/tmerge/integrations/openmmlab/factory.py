"""Registry hooks for optional OpenMMLab and legacy research integrations."""

from __future__ import annotations

from typing import Any

from tmerge.integrations.openmmlab.legacy_bridge import LegacyOperatorAdapter


def register_openmmlab(registry: Any) -> None:
    registry.operators.update(
        {
            "integration.openmmlab.mmdet_detector": build_mmdet_detector,
            "integration.openmmlab.mmdet_detector_with_features": build_mmdet_detector_with_features,
            "integration.openmmlab.mmtracking_mot": build_mmtracking_mot,
            "integration.openmmlab.mmtracking_deepsort": build_mmtracking_deepsort,
            "integration.openmmlab.mmtracking_tracktor": build_mmtracking_tracktor,
            "integration.reid.torchreid_extractor": build_torchreid_extractor,
        }
    )


def build_mmdet_detector(config: dict[str, Any]) -> LegacyOperatorAdapter:
    from videosys.ingestion.objectdetection.mmdet import MMDetDetectorPipeline

    return LegacyOperatorAdapter(
        operator_factory=MMDetDetectorPipeline,
        operator_kwargs={
            "config_file": config["config_file"],
            "checkpoint_file": config["checkpoint_file"],
            "device": config.get("device", "cuda:0"),
            "detect_classes": config.get("detect_classes"),
        },
    )


def build_mmdet_detector_with_features(config: dict[str, Any]) -> LegacyOperatorAdapter:
    from videosys.ingestion.compat.mmlib import MMLibCompatable, MMLibMoveData, MMMultiScaleFlipAug
    from videosys.ingestion.objectdetection.mmdet import MMDetDetectorWithFeatures

    transforms = config.get(
        "transforms",
        [
            {"type": "Resize", "keep_ratio": True},
            {"type": "RandomFlip"},
            {
                "type": "Normalize",
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
                "to_rgb": True,
            },
            {"type": "Pad", "size_divisor": 32},
            {"type": "ImageToTensor", "keys": ["img"]},
            {"type": "VideoCollect", "keys": ["img"]},
        ],
    )
    img_scale = tuple(config.get("img_scale", [1088, 1088]))
    adapter = _CompoundLegacyOperator(
        [
            (MMLibCompatable, {}),
            (MMMultiScaleFlipAug, {"transforms": transforms, "img_scale": img_scale, "flip": False}),
            (MMLibMoveData, {}),
            (
                MMDetDetectorWithFeatures,
                {
                    "config_file": config["config_file"],
                    "checkpoint_file": config["checkpoint_file"],
                    "device": config.get("device", "cuda:0"),
                    "detect_classes": config.get("detect_classes"),
                },
            ),
        ]
    )
    return adapter


def build_mmtracking_mot(config: dict[str, Any]) -> LegacyOperatorAdapter:
    from videosys.ingestion.tracking.mmtracking import MMTrackingMOT

    return LegacyOperatorAdapter(
        operator_factory=MMTrackingMOT,
        operator_kwargs={"config_file": config["config_file"], "device": config.get("device", "cuda:0")},
    )


def build_mmtracking_deepsort(config: dict[str, Any]) -> LegacyOperatorAdapter:
    from videosys.ingestion.tracking.mmtracking import MMTrackingDeepSORT

    kwargs = {key: config[key] for key in ("pretrains", "motion", "reid", "tracker") if key in config}
    return LegacyOperatorAdapter(operator_factory=MMTrackingDeepSORT, operator_kwargs=kwargs)


def build_mmtracking_tracktor(config: dict[str, Any]) -> LegacyOperatorAdapter:
    from videosys.ingestion.tracking.mmtracking import MMTrackingTracktor

    kwargs = {key: config[key] for key in ("pretrains", "motion", "reid", "tracker") if key in config}
    return LegacyOperatorAdapter(operator_factory=MMTrackingTracktor, operator_kwargs=kwargs)


def build_torchreid_extractor(config: dict[str, Any]) -> LegacyOperatorAdapter:
    from videosys.ingestion.reid.torchreid import TorchReIdFeatureExtracor

    return LegacyOperatorAdapter(
        operator_factory=TorchReIdFeatureExtracor,
        operator_kwargs={
            "model_name": config["model_name"],
            "model_path": config["model_path"],
            "device": config.get("device", "cuda"),
        },
    )


class _CompoundLegacyOperator(LegacyOperatorAdapter):
    def __init__(self, operator_specs: list[tuple[Any, dict[str, Any]]]) -> None:
        self._operator_specs = operator_specs
        super().__init__(operator_factory=lambda: None)

    def prepare(self, context: Any) -> None:
        from tmerge.integrations.openmmlab.legacy_bridge import _LegacyContextProxy, _NoopCollector

        self._legacy_context = _LegacyContextProxy(context.state)
        self._legacy_operators = []
        for operator_factory, operator_kwargs in self._operator_specs:
            operator = operator_factory(**operator_kwargs)
            operator.setup(context=self._legacy_context, collector=_NoopCollector())
            operator.prepare()
            self._legacy_operators.append(operator)

    def process(self, packet: Any, context: Any) -> Any:
        from videosys.ingestion import fields as legacy_fields

        for operator in self._legacy_operators:
            tables = {
                legacy_fields.DATA_FRAME_ID: packet.frame_id,
                legacy_fields.DATA_FRAME: packet.frame,
                legacy_fields.DATA_FRAME_META: packet.frame_meta,
                legacy_fields.DATA_OBJECT_DETECTION: packet.detections,
                legacy_fields.DATA_OBJECT_TRACK: packet.tracks,
                legacy_fields.DATA_OBJECT_TRACK_GT: packet.ground_truth_tracks,
                legacy_fields.DATA_TRACK_FEAT: packet.track_features,
            }
            tables.update(packet.extras)
            operator.process(tables)
            packet.detections = list(tables.get(legacy_fields.DATA_OBJECT_DETECTION, packet.detections))
            packet.tracks = list(tables.get(legacy_fields.DATA_OBJECT_TRACK, packet.tracks))
            packet.ground_truth_tracks = list(
                tables.get(legacy_fields.DATA_OBJECT_TRACK_GT, packet.ground_truth_tracks)
            )
            packet.track_features = list(
                tables.get(legacy_fields.DATA_TRACK_FEAT, packet.track_features)
            )
            packet.extras = {
                key: value
                for key, value in tables.items()
                if key
                not in {
                    legacy_fields.DATA_FRAME_ID,
                    legacy_fields.DATA_FRAME,
                    legacy_fields.DATA_FRAME_META,
                    legacy_fields.DATA_OBJECT_DETECTION,
                    legacy_fields.DATA_OBJECT_TRACK,
                    legacy_fields.DATA_OBJECT_TRACK_GT,
                    legacy_fields.DATA_TRACK_FEAT,
                }
            }
        return packet

    def cleanup(self, context: Any) -> None:
        for operator in reversed(self._legacy_operators):
            operator.cleanup()
