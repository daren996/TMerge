# Legacy E2E Tool Status

The following `e2e/configs/tools` entries now have first-class replacements and
should be treated as legacy-only references:

- `gen_mot_result_video.py`
  Replacement: `tmerge mot export-video`
- `visualize_mot_result.py`
  Replacement: `tmerge mot visualize`
- `gen_track_features_torchreid.py`
  Replacement: `tmerge export features examples/export.track-features.yaml`
- `gen_track_features.py`
  Replacement: `tmerge export features examples/export.track-features-openmmlab.yaml`
- `gen_video_from_images.py`
  Replacement: `tmerge export video --data ... --output ... --fps ...`

The following tools remain transitional and are not fully replaced yet:

- None in `e2e/configs/tools/` for the common workflows covered by the modern CLI.
- Keep the files as historical references until you decide to freeze or remove
  the legacy tool directory.

Practical guideline:

- Keep legacy tool files for reference and historical reproducibility.
- Prefer adding no new behavior to them.
- When a command has a `tmerge ...` equivalent, document and use the modern
  command instead.
