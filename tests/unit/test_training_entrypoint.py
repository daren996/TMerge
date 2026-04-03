from __future__ import annotations

from pathlib import Path
import types

from tmerge.config.loader import load_yaml_config
from tmerge.config.validation import validate_config
from tmerge.training import reid as reid_entry


def test_training_config_in_modern_directory_is_valid() -> None:
    config = load_yaml_config(Path("examples/train.reid.yaml"))
    validate_config(config)
    assert config["training"]["backend"] == "tmerge_torchreid"
    assert str(config["training"]["config_file"]).startswith("configs/training/reid/")


def test_run_reid_training_uses_native_backend(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_main(argv):
        called["argv"] = argv
        return 0

    monkeypatch.setattr("tmerge.training.reid_runner.main", fake_main)

    result = reid_entry.run_reid_training(
        {
            "training": {
                "backend": "tmerge_torchreid",
                "config_file": "configs/training/reid/mot_osnet_x1_0_softmax_256x128_amsgrad.yaml",
                "extra_args": ["train.max_epoch", "2"],
            }
        }
    )

    assert result == 0
    assert called["argv"] == [
        "--config-file",
        "configs/training/reid/mot_osnet_x1_0_softmax_256x128_amsgrad.yaml",
        "train.max_epoch",
        "2",
    ]
