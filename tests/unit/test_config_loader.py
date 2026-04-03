from __future__ import annotations

from pathlib import Path

from tmerge.config.loader import load_yaml_config
from tmerge.config.validation import ConfigValidationError, validate_config


def test_load_yaml_config_supports_extends_and_overrides(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "pipeline:\n"
        "  name: base\n"
        "  operators:\n"
        "    - type: source.image_folder\n"
        "      path: ${DATA_ROOT}/images\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: base.yaml\n"
        "pipeline:\n"
        "  name: child\n",
        encoding="utf-8",
    )

    config = load_yaml_config(child, ["pipeline.name=overridden"])

    assert config["pipeline"]["name"] == "overridden"
    assert config["pipeline"]["operators"][0]["type"] == "source.image_folder"


def test_validate_config_rejects_non_source_first_operator() -> None:
    config = {
        "pipeline": {
            "name": "broken",
            "operators": [
                {"type": "reporter.progress", "report_interval": 10},
            ],
        }
    }

    try:
        validate_config(config)
    except ConfigValidationError as exc:
        assert "first operator must be a source.* type" in str(exc)
    else:
        raise AssertionError("Expected validation to fail")


def test_validate_config_accepts_mot_sort_shape(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "img1"
    input_dir.mkdir()
    monkeypatch.setenv("TMERGE_MOT_INPUT", str(input_dir))
    monkeypatch.setenv("TMERGE_MMDET_CONFIG", "/tmp/config.py")
    monkeypatch.setenv("TMERGE_MMDET_CHECKPOINT", "/tmp/checkpoint.pth")
    monkeypatch.setenv("TMERGE_DEVICE", "cpu")
    monkeypatch.setenv("TMERGE_OUTPUT", "/tmp/output.txt")

    config = load_yaml_config(Path("configs/mot/mmdet_sort.yaml"))
    validate_config(config)
