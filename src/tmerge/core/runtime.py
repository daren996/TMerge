"""Modern sequential runtime for TMerge pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, Iterable

from tmerge.models.types import FramePacket


@dataclass(slots=True)
class RuntimeContext:
    logger: logging.Logger
    state: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.state:
            raise KeyError(f"Missing required runtime context key: {key}")
        return self.state[key]

    def put(self, key: str, value: Any) -> None:
        self.state[key] = value

    def has(self, key: str) -> bool:
        return key in self.state


class PipelineComponent(ABC):
    def prepare(self, context: RuntimeContext) -> None:
        """Prepare the component before processing starts."""

    def cleanup(self, context: RuntimeContext) -> None:
        """Release resources after processing completes."""


class Source(PipelineComponent, ABC):
    @abstractmethod
    def frames(self, context: RuntimeContext) -> Iterable[FramePacket]:
        """Yield frame packets for downstream operators."""


class Operator(PipelineComponent, ABC):
    @abstractmethod
    def process(self, packet: FramePacket, context: RuntimeContext) -> FramePacket | None:
        """Process one packet and return the next packet."""


@dataclass(slots=True)
class Pipeline:
    source: Source
    operators: list[Operator]
    context: RuntimeContext

    def run(self) -> None:
        components: list[PipelineComponent] = [self.source, *self.operators]
        try:
            for component in components:
                component.prepare(self.context)
            for packet in self.source.frames(self.context):
                current: FramePacket | None = packet
                for operator in self.operators:
                    if current is None:
                        break
                    current = operator.process(current, self.context)
        finally:
            for component in reversed(components):
                component.cleanup(self.context)

