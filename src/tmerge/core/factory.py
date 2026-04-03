"""Operator registry and pipeline construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tmerge.core.runtime import Operator, Pipeline, RuntimeContext, Source
from tmerge.io.loaders import MotResultLoader
from tmerge.io.sinks import ImageFolderSink, MotResultSink, TrackFeatureSink, VideoSink
from tmerge.io.sources import ImageFolderSource, VideoSource
from tmerge.operators.progress import ProgressReporter
from tmerge.operators.transforms import DetectionsToTracks


Factory = Callable[[dict[str, Any]], object]


@dataclass(slots=True)
class Registry:
    sources: dict[str, Factory]
    operators: dict[str, Factory]


def default_registry() -> Registry:
    from tmerge.integrations.openmmlab.factory import register_openmmlab

    registry = Registry(
        sources={
            "source.image_folder": ImageFolderSource.from_config,
            "source.video": VideoSource.from_config,
        },
        operators={
            "io.mot_result_loader": MotResultLoader.from_config,
            "reporter.progress": ProgressReporter.from_config,
            "sink.image_folder": ImageFolderSink.from_config,
            "sink.video": VideoSink.from_config,
            "sink.mot_result": MotResultSink.from_config,
            "sink.track_features": TrackFeatureSink.from_config,
            "transform.detections_to_tracks": DetectionsToTracks.from_config,
        },
    )
    register_openmmlab(registry)
    return registry


def build_pipeline(config: dict[str, Any], context: RuntimeContext) -> Pipeline:
    pipeline_config = config.get("pipeline")
    if not isinstance(pipeline_config, dict):
        raise ValueError("Config must contain a 'pipeline' section")
    operator_specs = pipeline_config.get("operators", [])
    if not operator_specs:
        raise ValueError("Pipeline must define at least one operator")

    registry = default_registry()
    source_spec = operator_specs[0]
    source = _build_source(source_spec, registry)
    operators = [_build_operator(spec, registry) for spec in operator_specs[1:]]
    context.put("pipeline_name", pipeline_config.get("name", "unnamed"))
    return Pipeline(source=source, operators=operators, context=context)


def _build_source(spec: dict[str, Any], registry: Registry) -> Source:
    component_type = spec["type"]
    factory = registry.sources.get(component_type)
    if factory is None:
        raise ValueError(f"Unknown source type: {component_type}")
    source = factory(spec)
    if not isinstance(source, Source):
        raise TypeError(f"{component_type} did not create a Source instance")
    return source


def _build_operator(spec: dict[str, Any], registry: Registry) -> Operator:
    component_type = spec["type"]
    factory = registry.operators.get(component_type)
    if factory is None:
        raise ValueError(f"Unknown operator type: {component_type}")
    operator = factory(spec)
    if not isinstance(operator, Operator):
        raise TypeError(f"{component_type} did not create an Operator instance")
    return operator

