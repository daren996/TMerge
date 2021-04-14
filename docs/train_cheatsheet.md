# mmtracking

Setup data folder first: create a symbolic link to the dataset.
`ln -s /path/to/dataset/ data`

## for MOT dataset

convert annotations to coco style.

```
python tools/convert_datasets/mot2coco.py -i data/MOT17/ -o data/MOT17/annotations/ --convert-det --split-train
```

train, e.g.:

```
python tools/train.py configs/det/faster-rcnn_r50_fpn_4e_mot17-half.py
```