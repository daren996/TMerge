# mmtracking

Setup data folder first: create a symbolic link to the dataset.
`ln -s /path/to/dataset/ data`

```
ln -s ../storage/dataset/ ../mmtracking/data
```

## for MOT dataset

```
cd ../mmtracking
```

convert annotations to coco style.

```
python tools/convert_datasets/mot2coco.py -i data/MOT17/ -o data/MOT17/annotations/ --convert-det --split-train
```

e.g., train using mmtracking's own `train.py`

```
python tools/train.py configs/det/faster-rcnn_r50_fpn_4e_mot17-half.py
```

## Train reid model

train reid model using `deep-person-reid`

```
cd ../VideoSystem
python ./videosys/train/reid/train_reid.py --config-file config/train/reid/mot_osnet_x1_0_softmax_256x128_amsgrad.yaml
```
