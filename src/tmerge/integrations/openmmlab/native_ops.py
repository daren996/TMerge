"""Native OpenMMLab operators for the new runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tmerge.core.runtime import Operator, RuntimeContext
from tmerge.models.types import DetectionResult, FramePacket, TrackFeature, TrackingResult


def _require_module(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name, fromlist=["_sentinel"])
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Optional dependency '{module_name}' is required. Install with: {install_hint}"
        ) from exc


def _build_mmlib_sample(frame: np.ndarray, transforms: list[dict[str, Any]], img_scale: tuple[int, int], device: str) -> dict[str, Any]:
    mmcv_parallel = _require_module("mmcv.parallel", 'python -m pip install -e ".[openmmlab]"')
    multi_scale_module = _require_module(
        "mmdet.datasets.pipelines.test_time_aug",
        'python -m pip install -e ".[openmmlab]"',
    )
    aug = multi_scale_module.MultiScaleFlipAug(transforms, img_scale=img_scale, flip=False)
    results = {
        "img": frame,
        "img_shape": frame.shape,
        "ori_shape": frame.shape,
        "img_fields": ["img"],
        "img_info": {"width": frame.shape[1], "height": frame.shape[0]},
        "filename": None,
        "ori_filename": None,
    }
    data = aug(results)
    data = mmcv_parallel.collate([data], samples_per_gpu=1)
    if device.startswith("cuda"):
        data = mmcv_parallel.scatter(data, [device])[0]
    return data


def _filter_detect_classes(classes: tuple[str, ...], detect_classes: list[str] | None, logger: Any) -> list[int] | None:
    if detect_classes is None:
        return None
    ignored = [item for item in detect_classes if item not in classes]
    if ignored:
        logger.warning("Ignoring unsupported detect classes: %s", ignored)
    return [classes.index(item) for item in detect_classes if item in classes]


def _collect_detection_results(
    bbox_result: list[np.ndarray],
    detect_class_ids: list[int] | None,
) -> list[DetectionResult]:
    if not bbox_result:
        return []
    non_empty = [bbox for bbox in bbox_result if len(bbox) > 0]
    if not non_empty:
        return []
    bboxes = np.vstack(non_empty)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
        if len(bbox) > 0
    ]
    if len(labels) == 0:
        return []
    concatenated_labels = np.concatenate(labels)
    return [
        DetectionResult(bbox=row[:-1], label=int(label), confidence=float(row[-1]))
        for row, label in zip(bboxes, concatenated_labels)
        if detect_class_ids is None or int(label) in detect_class_ids
    ]


def _tracks_from_restore_result(restored: tuple[np.ndarray, np.ndarray, np.ndarray]) -> list[TrackingResult]:
    bboxes, labels, ids = restored
    return [
        TrackingResult(uid=int(uid), label=int(label), bbox=bbox[:4], confidence=float(bbox[4]))
        for bbox, label, uid in zip(bboxes, labels, ids)
    ]


@dataclass(slots=True)
class MMDetDetector(Operator):
    config_file: str
    checkpoint_file: str
    device: str = "cuda:0"
    detect_classes: list[str] | None = None
    with_features: bool = False
    img_scale: tuple[int, int] = (1088, 1088)
    transforms: list[dict[str, Any]] = field(default_factory=list)
    _bbox2result: Any = field(init=False, default=None)
    _torch: Any = field(init=False, default=None)
    _model: Any = field(init=False, default=None)
    _detect_class_ids: list[int] | None = field(init=False, default=None)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MMDetDetector":
        return cls(
            config_file=str(config["config_file"]),
            checkpoint_file=str(config["checkpoint_file"]),
            device=str(config.get("device", "cuda:0")),
            detect_classes=list(config["detect_classes"]) if config.get("detect_classes") else None,
            with_features=bool(config.get("with_features", False)),
            img_scale=tuple(config.get("img_scale", [1088, 1088])),
            transforms=list(config.get("transforms", _default_transforms())),
        )

    def prepare(self, context: RuntimeContext) -> None:
        mmdet_apis = _require_module("mmdet.apis", 'python -m pip install -e ".[openmmlab]"')
        self._bbox2result = _require_module("mmdet.core", 'python -m pip install -e ".[openmmlab]"').bbox2result
        self._torch = _require_module("torch", 'python -m pip install -e ".[openmmlab]"')
        self._model = mmdet_apis.init_detector(self.config_file, self.checkpoint_file, device=self.device)
        class_names = tuple(getattr(self._model, "CLASSES", ()))
        context.put("openmmlab.detector", self._model)
        context.put("openmmlab.detector_classes", class_names)
        self._detect_class_ids = _filter_detect_classes(class_names, self.detect_classes, context.logger)

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        if not self.with_features:
            mmdet_apis = _require_module("mmdet.apis", 'python -m pip install -e ".[openmmlab]"')
            result = mmdet_apis.inference_detector(self._model, packet.frame)
            bbox_result = result[0] if isinstance(result, tuple) else result
            packet.detections = _collect_detection_results(bbox_result, self._detect_class_ids)
            return packet

        sample = _build_mmlib_sample(packet.frame, self.transforms, self.img_scale, self.device)
        image = sample["img"][0]
        img_metas = sample["img_metas"][0]
        with self._torch.no_grad():
            features = self._model.extract_feat(image)
            if hasattr(self._model, "roi_head"):
                proposals = self._model.rpn_head.simple_test_rpn(features, img_metas)
                det_bboxes, det_labels = self._model.roi_head.simple_test_bboxes(
                    features,
                    img_metas,
                    proposals,
                    self._model.roi_head.test_cfg,
                    rescale=True,
                )
                det_bboxes = det_bboxes[0]
                det_labels = det_labels[0]
                num_classes = self._model.roi_head.bbox_head.num_classes
            elif hasattr(self._model, "bbox_head"):
                outs = self._model.bbox_head(features)
                result_list = self._model.bbox_head.get_bboxes(
                    *outs,
                    img_metas=img_metas,
                    rescale=True,
                )
                det_bboxes = result_list[0][0]
                det_labels = result_list[0][1]
                num_classes = self._model.bbox_head.num_classes
            else:
                raise TypeError("MMDet model must expose either roi_head or bbox_head")
            bbox_result = self._bbox2result(det_bboxes, det_labels, num_classes)
        packet.detections = _collect_detection_results(bbox_result, self._detect_class_ids)
        packet.extras["openmmlab.sample"] = sample
        packet.extras["openmmlab.features"] = features
        return packet


@dataclass(slots=True)
class MMTrackingMOTOperator(Operator):
    config_file: str
    device: str = "cuda:0"
    _restore_result: Any = field(init=False, default=None)
    _inference_mot: Any = field(init=False, default=None)
    _model: Any = field(init=False, default=None)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MMTrackingMOTOperator":
        return cls(config_file=str(config["config_file"]), device=str(config.get("device", "cuda:0")))

    def prepare(self, context: RuntimeContext) -> None:
        apis = _require_module("mmtrack.apis", 'python -m pip install -e ".[openmmlab]"')
        self._restore_result = _require_module("mmtrack.core", 'python -m pip install -e ".[openmmlab]"').restore_result
        self._inference_mot = apis.inference_mot
        self._model = apis.init_model(self.config_file, None, self.device)

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        result = self._inference_mot(self._model, packet.frame, packet.frame_id)
        restored = self._restore_result(result["track_results"], return_ids=True)
        packet.tracks = _tracks_from_restore_result(restored)
        return packet


@dataclass(slots=True)
class MMTrackingSortOperator(Operator):
    motion: dict[str, Any] | None = None
    tracker: dict[str, Any] | None = None
    pretrains: dict[str, Any] | None = None
    reid: dict[str, Any] | None = None
    tracker_kind: str = "sort"
    device: str = "cuda:0"
    _torch: Any = field(init=False, default=None)
    _track2result: Any = field(init=False, default=None)
    _restore_result: Any = field(init=False, default=None)
    _model: Any = field(init=False, default=None)
    _tracker: Any = field(init=False, default=None)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MMTrackingSortOperator":
        return cls(
            motion=dict(config["motion"]) if config.get("motion") else None,
            tracker=dict(config["tracker"]) if config.get("tracker") else None,
            pretrains=dict(config["pretrains"]) if config.get("pretrains") else None,
            reid=dict(config["reid"]) if config.get("reid") else None,
            tracker_kind=str(config.get("tracker_kind", "sort")),
            device=str(config.get("device", "cuda:0")),
        )

    def prepare(self, context: RuntimeContext) -> None:
        mmtrack_models = _require_module("mmtrack.models.mot", 'python -m pip install -e ".[openmmlab]"')
        mmtrack_core = _require_module("mmtrack.core", 'python -m pip install -e ".[openmmlab]"')
        self._torch = _require_module("torch", 'python -m pip install -e ".[openmmlab]"')
        self._track2result = mmtrack_core.track2result
        self._restore_result = mmtrack_core.restore_result
        if self.tracker_kind == "tracktor":
            self._model = mmtrack_models.Tracktor(
                pretrains=self.pretrains,
                motion=self.motion,
                reid=self.reid,
                tracker=self.tracker,
            )
        elif self.tracker_kind == "deepsort":
            self._model = mmtrack_models.DeepSORT(
                pretrains=self.pretrains,
                motion=self.motion,
                reid=self.reid,
                tracker=self.tracker,
            )
        else:
            self._model = mmtrack_models.DeepSORT(
                motion=self.motion,
                tracker=self.tracker,
            )
        self._tracker = self._model.tracker
        self._model.to(self.device)
        if self.tracker_kind == "tracktor":
            detector = context.get("openmmlab.detector")
            if detector is None:
                raise RuntimeError("Tracktor requires a detector to run before the tracker operator")
            self._model.detector = detector
        self._model.eval()

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        sample = packet.extras.get("openmmlab.sample")
        if sample is None:
            raise RuntimeError(
                "mmtracking SORT/DeepSORT/Tracktor operators require detector features. "
                "Run integration.openmmlab.mmdet_detector_with_features first."
            )
        class_names = context.get("openmmlab.detector_classes", ())
        num_classes = len(class_names)
        image = sample["img"][0]
        img_meta = sample["img_metas"][0]
        bboxes_tensor = self._torch.tensor([[*det.bbox, det.confidence] for det in packet.detections])
        labels_tensor = self._torch.tensor([det.label for det in packet.detections])
        if self.device.startswith("cuda"):
            bboxes_tensor = bboxes_tensor.cuda()
            labels_tensor = labels_tensor.cuda()
        if self.tracker_kind == "tracktor":
            features = packet.extras.get("openmmlab.features")
            result_tracks: list[TrackingResult] = []
            if len(bboxes_tensor) > 0:
                with self._torch.no_grad():
                    bboxes, labels, ids = self._tracker.track(
                        image,
                        img_meta,
                        self._model,
                        features,
                        bboxes_tensor,
                        labels_tensor,
                        packet.frame_id,
                        rescale=True,
                    )
                restored = self._restore_result(
                    self._track2result(bboxes, labels, ids, num_classes),
                    return_ids=True,
                )
                result_tracks = _tracks_from_restore_result(restored)
            packet.tracks = result_tracks
            return packet

        with self._torch.no_grad():
            bboxes, labels, ids = self._tracker.track(
                image,
                img_meta,
                self._model,
                bboxes_tensor,
                labels_tensor,
                packet.frame_id,
                rescale=True,
            )
        restored = self._restore_result(
            self._track2result(bboxes, labels, ids, num_classes),
            return_ids=True,
        )
        packet.tracks = _tracks_from_restore_result(restored)
        return packet


@dataclass(slots=True)
class TorchReIdExtractor(Operator):
    model_name: str
    model_path: str
    device: str = "cuda"
    _torch: Any = field(init=False, default=None)
    _cv2: Any = field(init=False, default=None)
    _extractor: Any = field(init=False, default=None)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TorchReIdExtractor":
        return cls(
            model_name=str(config["model_name"]),
            model_path=str(config["model_path"]),
            device=str(config.get("device", "cuda")),
        )

    def prepare(self, context: RuntimeContext) -> None:
        torchreid_utils = _require_module("torchreid.utils", 'python -m pip install -e ".[reid]"')
        self._torch = _require_module("torch", 'python -m pip install -e ".[reid]"')
        self._cv2 = _require_module("cv2", 'python -m pip install -e .')
        self._extractor = torchreid_utils.FeatureExtractor(
            model_name=self.model_name,
            model_path=self.model_path,
            device=self.device,
        )

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        if not packet.tracks:
            packet.track_features = []
            return packet
        bboxes = self._torch.tensor([track.bbox for track in packet.tracks])
        bboxes[:, 0::2] = self._torch.clamp(bboxes[:, 0::2], min=0, max=len(packet.frame[0]))
        bboxes[:, 1::2] = self._torch.clamp(bboxes[:, 1::2], min=0, max=len(packet.frame))
        cropped_images = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            if x2 == x1:
                x2 += 1
            if y2 == y1:
                y2 += 1
            cropped = packet.frame[y1:y2, x1:x2]
            cropped_images.append(self._cv2.cvtColor(cropped, self._cv2.COLOR_BGR2RGB))
        features = self._extractor(cropped_images)
        packet.track_features = [
            TrackFeature(uid=track.uid, bbox=track.bbox, feature=feature.cpu())
            for track, feature in zip(packet.tracks, features)
        ]
        return packet


def _default_transforms() -> list[dict[str, Any]]:
    return [
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
    ]
