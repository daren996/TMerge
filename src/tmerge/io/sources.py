"""Source operators for images and video."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2

from tmerge.core.runtime import RuntimeContext, Source
from tmerge.models.types import FramePacket, ImageFolderMeta, VideoMeta
from tmerge.utils.common import natural_frame_key as _natural_frame_key


@dataclass(slots=True)
class ImageFolderSource(Source):
    path: Path
    _images: list[Path] = field(init=False, default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ImageFolderSource":
        return cls(path=Path(str(config["path"])).expanduser())

    def prepare(self, context: RuntimeContext) -> None:
        if not self.path.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.path}")
        self._images = sorted(
            [item for item in self.path.iterdir() if item.is_file()],
            key=_natural_frame_key,
        )
        context.put("input_meta", ImageFolderMeta(total=len(self._images), path=self.path))

    def frames(self, context: RuntimeContext) -> Iterable[FramePacket]:
        for frame_id, path in enumerate(self._images, start=1):
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {path}")
            yield FramePacket(
                frame_id=frame_id,
                frame=frame,
                frame_meta={"img_shape": frame.shape, "ori_shape": frame.shape, "path": str(path)},
            )


@dataclass(slots=True)
class VideoSource(Source):
    path: Path
    _capture: object = field(init=False, default=None)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "VideoSource":
        return cls(path=Path(str(config["path"])).expanduser())

    def prepare(self, context: RuntimeContext) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        meta = VideoMeta(
            fps=float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0),
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            path=self.path,
        )
        context.put("input_meta", meta)

    def frames(self, context: RuntimeContext) -> Iterable[FramePacket]:
        frame_id = 0
        while True:
            ret, frame = self._capture.read()
            if not ret:
                break
            frame_id += 1
            yield FramePacket(
                frame_id=frame_id,
                frame=frame,
                frame_meta={"img_shape": frame.shape, "ori_shape": frame.shape},
            )

    def cleanup(self, context: RuntimeContext) -> None:
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()



