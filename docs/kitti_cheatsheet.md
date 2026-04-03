

```
python ./videosys/train/reid/train_reid.py --config-file config/train/reid/kitti_osnet_x1_0_triplet_256x128_amsgrad.yaml
```

```
python ./scripts/kitti/run_mot_methods.py
python ./scripts/kitti/filter_track_results.py
python ./scripts/kitti/extract_reid_features_w_torch_reid.py
python ./scripts/kitti/test_mot_person_reid_features_pairwise_average.py
```
