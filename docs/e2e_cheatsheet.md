# simple

```
python e2e/ingestion_runner.py e2e/configs/simple/play_video.py
```

```
python e2e/ingestion_runner.py e2e/configs/simple/play_images.py
```

# tracking

```
python e2e/ingestion_runner.py e2e/configs/tracking/mmt_sort_private.py
```

```
python e2e/ingestion_runner.py e2e/configs/tracking/mmt_deepsort_private.py --video VIDEO_PATH --output OUTPUT_PATH
```

e.g. SORT with faster_rcnn (one class)

sort
```
python e2e/ingestion_runner.py e2e/configs/tracking/mmt_sort_private.py \
    --config ./e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py \
    --checkpoint https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth \
    --output ../storage/results/mot16/MOT16-03-faster_rcnn-sort-person.txt
```
tracktor
```
python e2e/ingestion_runner.py e2e/configs/tracking/mmt_tracktor_private.py \
    --config ./e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py \
    --checkpoint https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth \
    --output ../storage/results/mot16/MOT16-03-faster_rcnn-tracktor-person.txt
```

uma
```
python e2e/ingestion_runner.py e2e/configs/tracking/uma_private.py \
    --config ./e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py \
    --checkpoint https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth \
    --output ../storage/results/mot16/MOT16-03-faster_rcnn-uma-person.txt
```
# sot
```
python e2e/ingestion_runner.py e2e/configs/sot/mmt_pipeline_siamese_rpn.py
```