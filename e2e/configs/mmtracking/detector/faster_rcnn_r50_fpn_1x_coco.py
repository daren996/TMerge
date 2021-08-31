# based on the mmdet configs
project_base = '../../../../'
mmdet_config_base = project_base + '../mmdetection/configs/_base_'
mmtrack_config_base = project_base + '../mmtracking/configs/_base_'
_base_ = [
    mmdet_config_base + '/models/faster_rcnn_r50_fpn.py',
    mmtrack_config_base + '/datasets/mot_challenge.py', 
    mmtrack_config_base + '/default_runtime.py'
]

model = dict(
    rpn_head=dict(bbox_coder=dict(clip_border=False)),
    roi_head=dict(
        bbox_head=dict(bbox_coder=dict(
            clip_border=False), num_classes=80)),
)
