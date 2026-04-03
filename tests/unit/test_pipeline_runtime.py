from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from tmerge.core.runtime import Operator, Pipeline, RuntimeContext, Source
from tmerge.models.types import FramePacket


@dataclass(slots=True)
class DummySource(Source):
    def frames(self, context: RuntimeContext):
        for frame_id in range(1, 3):
            yield FramePacket(frame_id=frame_id, frame=np.zeros((2, 2, 3), dtype=np.uint8))


@dataclass(slots=True)
class CounterOperator(Operator):
    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket:
        context.put("count", context.get("count", 0) + 1)
        return packet


def test_pipeline_runs_all_packets() -> None:
    context = RuntimeContext(logger=logging.getLogger("test"))
    pipeline = Pipeline(source=DummySource(), operators=[CounterOperator()], context=context)

    pipeline.run()

    assert context.get("count") == 2

