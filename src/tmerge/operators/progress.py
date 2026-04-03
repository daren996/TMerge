"""Progress reporting operator using structured logging."""

from __future__ import annotations

from dataclasses import dataclass
import time

from tmerge.core.runtime import Operator, RuntimeContext
from tmerge.models.types import FramePacket


@dataclass(slots=True)
class ProgressReporter(Operator):
    report_interval: int = 100
    log_prefix: str = "pipeline"

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ProgressReporter":
        return cls(
            report_interval=int(config.get("report_interval", 100)),
            log_prefix=str(config.get("log_prefix", "pipeline")),
        )

    def prepare(self, context: RuntimeContext) -> None:
        self._started_at = time.time()
        meta = context.get("input_meta")
        context.logger.info("%s started | input=%s", self.log_prefix, meta)

    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        if packet.frame_id % self.report_interval == 0:
            elapsed = time.time() - self._started_at
            fps = packet.frame_id / elapsed if elapsed > 0 else 0.0
            context.logger.info(
                "%s progress | frame=%s | elapsed=%.2fs | fps=%.2f",
                self.log_prefix,
                packet.frame_id,
                elapsed,
                fps,
            )
        return packet

    def cleanup(self, context: RuntimeContext) -> None:
        elapsed = time.time() - self._started_at
        context.logger.info("%s finished | elapsed=%.2fs", self.log_prefix, elapsed)

