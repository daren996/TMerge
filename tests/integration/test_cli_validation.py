from __future__ import annotations

from pathlib import Path

from tmerge.cli.main import main


def test_cli_returns_nonzero_for_invalid_config(tmp_path: Path) -> None:
    config = tmp_path / "broken.yaml"
    config.write_text(
        "pipeline:\n"
        "  name: broken\n"
        "  operators:\n"
        "    - type: reporter.progress\n",
        encoding="utf-8",
    )

    exit_code = main(["run", "pipeline", str(config)])

    assert exit_code == 2
