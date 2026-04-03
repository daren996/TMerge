

```
python ./videosys/train/reid/train_reid.py --config-file config/train/reid/pathtrack_osnet_x1_0_triplet_256x128_amsgrad.yaml
```

```
python ./scripts/pathtrack/run_mot_methods.py
python ./scripts/pathtrack/filter_track_results.py
python ./scripts/pathtrack/extract_reid_features_w_torch_reid.py
python ./scripts/pathtrack/test_mot_person_reid_features_pairwise_average.py
```
