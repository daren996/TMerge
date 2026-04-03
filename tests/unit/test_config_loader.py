from __future__ import annotations

from pathlib import Path

from tmerge.config.loader import load_yaml_config


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

