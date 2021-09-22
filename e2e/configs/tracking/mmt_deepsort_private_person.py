from .mmt_deepsort_private import operators as private_operators
from .mmt_deepsort_private import default_args as private_args

_default_args = dict(
    path='../storage/dataset/MOT17/train/MOT17-11-DPM/img1',
    # path='../storage/dataset/videos/MOT16-03.mp4',
    config='./e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py',
    # pylint: disable=line-too-long
    # checkpoint='https://download.openmmlab.com/mmtracking/mot/faster_rcnn/'
    #            'faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth',
    checkpoint='../mmtracking/checkpoints/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth',
    display=False,
    output='../storage/results/mot17/MOT17-11-DPM/faster_rcnn-deepsort-person.txt',
    no_report_save=False
    # output=''
)

default_args = {**private_args, **_default_args}


def operators(args):
    args.classes = ''
    return private_operators(args)
