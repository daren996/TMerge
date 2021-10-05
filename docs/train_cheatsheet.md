# mmtracking

Setup data folder first: create a symbolic link to the dataset.
`ln -s /path/to/dataset/ data`

```
ln -s ../storage/dataset/ ../mmtracking/data
```

## Train detection model

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

Download weights [osnet_x1_0_imagenet.pth](https://drive.google.com/file/d/1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY/view?usp=sharing) from [MODEL_ZOOM](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html)

Put the model into path `../storage/models/reid/`

train our own reid model using [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)

```
cd ../VideoSystem
python ./videosys/train/reid/train_reid.py --config-file config/train/reid/mot_osnet_x1_0_softmax_256x128_amsgrad.yaml
```

Configuration is set in file `mot_osnet_x1_0_softmax_256x128_amsgrad.yaml`.

Newly trained reid model is located in `log/osnet_x1_0_mot17det_softmax_rx/model`

### Example: test new reid model

first, run all (or some of) the mot methods
```
python ./scripts/mot/run_mot_methods.py
```

filter track results
```
python ./scripts/mot/reid/filter_track_results.py
```

extract reid features
```
python ./scripts/mot/reid/extract_reid_features_w_torch_reid.py
```

test mot features pairwise
```
python ./scripts/mot/reid/test_mot_person_reid_features_pairwise_average.py
python ./scripts/mot/reid/plot_rank_and_sampling.py  # plot rank
```
