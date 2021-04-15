from .mmt_deepsort_private import operators as private_operators

default_args = dict(
    video='/media/ytchen/hdd/dataset/videos/MOT16-03.mp4',
    config = './e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py',
    # pylint: disable=line-too-long
    checkpoint = 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth',
    display=False,
    output='../storage/results/mot16/MOT16-03-faster_rcnn-deepsort-person.txt',
    no_report_save=False
    # output=''
)

def operators(args):
    args.classes=''
    return private_operators(args)
