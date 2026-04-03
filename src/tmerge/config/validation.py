"""Config validation with actionable error messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ValidationIssue:
    path: str
    message: str


class ConfigValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        lines = ["Invalid TMerge config:"]
        lines.extend(f"- {issue.path}: {issue.message}" for issue in issues)
        super().__init__("\n".join(lines))


PIPELINE_OPERATOR_RULES: dict[str, tuple[str, ...]] = {
    "source.image_folder": ("path",),
    "source.video": ("path",),
    "io.mot_result_loader": ("path",),
    "reporter.progress": (),
    "sink.image_folder": ("path",),
    "sink.video": ("path",),
    "sink.mot_result": ("path",),
    "sink.track_features": ("path",),
    "transform.detections_to_tracks": (),
    "integration.openmmlab.mmdet_detector": ("config_file", "checkpoint_file"),
    "integration.openmmlab.mmdet_detector_with_features": ("config_file", "checkpoint_file"),
    "integration.openmmlab.mmtracking_mot": ("config_file",),
    "integration.openmmlab.mmtracking_sort": (),
    "integration.openmmlab.mmtracking_deepsort": (),
    "integration.openmmlab.mmtracking_tracktor": (),
    "integration.reid.torchreid_extractor": ("model_name", "model_path"),
}

TRAINING_RULES: dict[str, tuple[str, ...]] = {
    "tmerge_torchreid": ("config_file",),
    "legacy_torchreid": ("config_file",),
}


def validate_config(config: dict[str, Any]) -> None:
    issues: list[ValidationIssue] = []
    if not isinstance(config, dict):
        raise ConfigValidationError([ValidationIssue(path="$", message="config must be a mapping")])

    if "pipeline" in config:
        _validate_pipeline(config["pipeline"], issues)
    elif "training" in config:
        _validate_training(config["training"], issues)
    else:
        issues.append(
            ValidationIssue(
                path="$",
                message="config must contain either a 'pipeline' section or a 'training' section",
            )
        )

    if issues:
        raise ConfigValidationError(issues)


def _validate_pipeline(pipeline: Any, issues: list[ValidationIssue]) -> None:
    if not isinstance(pipeline, dict):
        issues.append(ValidationIssue(path="pipeline", message="must be a mapping"))
        return
    operators = pipeline.get("operators")
    if not isinstance(operators, list) or not operators:
        issues.append(ValidationIssue(path="pipeline.operators", message="must be a non-empty list"))
        return

    first = operators[0]
    if not isinstance(first, dict):
        issues.append(ValidationIssue(path="pipeline.operators.0", message="must be a mapping"))
    elif not str(first.get("type", "")).startswith("source."):
        issues.append(
            ValidationIssue(
                path="pipeline.operators.0.type",
                message="the first operator must be a source.* type",
            )
        )

    for index, operator in enumerate(operators):
        path = f"pipeline.operators.{index}"
        if not isinstance(operator, dict):
            issues.append(ValidationIssue(path=path, message="must be a mapping"))
            continue
        operator_type = operator.get("type")
        if not isinstance(operator_type, str) or not operator_type:
            issues.append(ValidationIssue(path=f"{path}.type", message="must be a non-empty string"))
            continue
        if operator_type not in PIPELINE_OPERATOR_RULES:
            issues.append(
                ValidationIssue(
                    path=f"{path}.type",
                    message=(
                        "unknown operator type; register it in the factory or fix the config value"
                    ),
                )
            )
            continue
        required_fields = PIPELINE_OPERATOR_RULES[operator_type]
        for field_name in required_fields:
            value = operator.get(field_name)
            if value is None or value == "":
                issues.append(
                    ValidationIssue(
                        path=f"{path}.{field_name}",
                        message="is required",
                    )
                )


def _validate_training(training: Any, issues: list[ValidationIssue]) -> None:
    if not isinstance(training, dict):
        issues.append(ValidationIssue(path="training", message="must be a mapping"))
        return
    backend = training.get("backend", "legacy_torchreid")
    if backend not in TRAINING_RULES:
        issues.append(
            ValidationIssue(
                path="training.backend",
                message=f"unsupported backend '{backend}'",
            )
        )
        return
    for field_name in TRAINING_RULES[backend]:
        value = training.get(field_name)
        if value is None or value == "":
            issues.append(ValidationIssue(path=f"training.{field_name}", message="is required"))
