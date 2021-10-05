"""
Background Subtraction.
Generate MOT17BGS folder.
python tests/bgs.py --dataset 02,04,05,09,10,11,13 --show 0
"""

# import numpy as np
import sys
import os
import shutil
import argparse
import cv2

parser = argparse.ArgumentParser(
    description='This program shows how to use background subtraction methods provided by \
    OpenCV. You can process both videos and images.')
parser.add_argument('--dataset', type=str, help='MOT dataset id.')
parser.add_argument('--algo', type=str, 
    help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--show', type=int, help='Show results.', default=0)
args = parser.parse_args()

datasets = []
if ',' in args.dataset:
    for ds in args.dataset.split(','):
        datasets.append(ds)
else:
    datasets.append(args.dataset)
print(args.dataset)
print(datasets)

for dataset in datasets:
    print("Processing MOT17-{}".format(dataset))
    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    source_path = '../storage/dataset/MOT17Det/train/MOT17-{}/img1'.format(dataset)
    images = []
    for root, _, files in os.walk(source_path):
        for file in files:
            images.append((os.path.join(root, file), int(file[:file.index('.')])))
    images = [file[0] for file in sorted(images, key=lambda x:x[1])]
    images = [images[0]] + images

    if not os.path.exists('../storage/dataset/MOT17BGS/train/MOT17-{}'.format(dataset)):
        shutil.copytree('../storage/dataset/MOT17Det/train/MOT17-{}'.format(dataset), 
                        '../storage/dataset/MOT17BGS/train/MOT17-{}'.format(dataset))
    output_path = '../storage/dataset/MOT17BGS/train/MOT17-{}/img1'.format(dataset)

    for idx, file in enumerate(images):
        if idx > 99 and idx % 100 == 0:
            print("\t%d/%d" % (idx, len(images)))
        frame = cv2.imread(file)
        if frame is None:
            print("NONE FILE")
            sys.exit(-2)
        fgmask = backSub.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  # denoising
        frame_masked = cv2.bitwise_and(frame, frame, mask=fgmask)
        if args.show:
            cv2.imshow('frame', frame_masked)
            cv2.waitKey(1)
        else:
            cv2.imwrite(os.path.join(output_path, os.path.split(file)[1]), frame_masked)

    cv2.destroyAllWindows()
