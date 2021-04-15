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

deep sort
```
python e2e/ingestion_runner.py e2e/configs/tracking/mmt_deepsort_private_person.py --video VIDEO_PATH --output OUTPUT_PATH
```

e.g. SORT with faster_rcnn (one class)

deep sort
```
python e2e/ingestion_runner.py e2e/configs/tracking/mmt_deepsort_private.py 
```

sort
```
python e2e/ingestion_runner.py e2e/configs/tracking/mmt_sort_private.py \
    --config ./e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py \
    --checkpoint https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth \
    --output ../storage/results/mot17/MOT17-11-DPM-faster_rcnn-sort-person.txt
```
tracktor
```
python e2e/ingestion_runner.py e2e/configs/tracking/mmt_tracktor_private.py \
    --config ./e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py \
    --checkpoint https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth \
    --output ../storage/results/mot17/MOT17-11-DPM-faster_rcnn-tracktor-person.txt
```

uma
```
python e2e/ingestion_runner.py e2e/configs/tracking/uma_private.py \
    --config ./e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py \
    --checkpoint https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth \
    --output ../storage/results/mot17/MOT17-11-DPM-faster_rcnn-uma-person.txt
```
# sot
```
python e2e/ingestion_runner.py e2e/configs/sot/mmt_pipeline_siamese_rpn.py
```

# tools

generate MOT result videos.

```
python e2e/ingestion_runner.py e2e/configs/tools/gen_mot_result_video.py \
    --result_path ../storage/results/mot17/MOT17-11-DPM-faster_rcnn-sort-person.txt \
    --output_folder ../storage/results/mot17/videos/ \
    --output_name MOT17-11-faster_rcnn-sort-person --fps 30
    

python e2e/ingestion_runner.py e2e/configs/tools/gen_mot_result_video.py \
    --result_path ../storage/results/mot17/MOT17-11-DPM-faster_rcnn-deepsort-person.txt \
    --output_folder ../storage/results/mot17/videos/ \
    --output_name MOT17-11-faster_rcnn-deepsort-person --fps 30

python e2e/ingestion_runner.py e2e/configs/tools/gen_mot_result_video.py \
    --result_path ../storage/results/mot17/MOT17-11-DPM-faster_rcnn-tracktor-person.txt \
    --output_folder ../storage/results/mot17/videos/ \
    --output_name MOT17-11-faster_rcnn-tracktor-person --fps 30

python e2e/ingestion_runner.py e2e/configs/tools/gen_mot_result_video.py \
    --result_path ../storage/results/mot17/MOT17-11-DPM-faster_rcnn-uma-person.txt \
    --output_folder ../storage/results/mot17/videos/ \
    --output_name MOT17-11-faster_rcnn-uma-person --fps 30
```