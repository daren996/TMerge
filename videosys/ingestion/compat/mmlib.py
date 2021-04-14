from mmdet.datasets.pipelines.test_time_aug import MultiScaleFlipAug

from mmcv.parallel import collate, scatter

from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class MMLibCompatable(Operator):
    
    def process(self, tables):
        # construct
        img = tables[fields.DATA_FRAME]
        img_info = {
            'width': img.shape[1],
            'height': img.shape[0],
        }
        results = {
            'img': img,
            'img_shape': img.shape,
            'ori_shape': img.shape,
            'img_fields': ['img'],
            'img_info': img_info,
            'filename': None,
            'ori_filename': None,
        }

        tables[fields.COMPAT_MMLIB] = results
        self.collector.emit(tables)

class MMLibMoveData(Operator):
    def process(self, tables):
        device = 'cuda:0'
        data = tables[fields.COMPAT_MMLIB]
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [device])[0]
        tables[fields.COMPAT_MMLIB] = data
        self.collector.emit(tables)

class MMMultiScaleFlipAug(Operator):

    def __init__(self, transforms, img_scale=None, scale_factor = None, flip=False, 
            flip_direction='horizontal'):
        self.aug = MultiScaleFlipAug(transforms, img_scale, scale_factor, flip, flip_direction)
    
    def process(self, tables):
        tables[fields.COMPAT_MMLIB] = self.aug(tables[fields.COMPAT_MMLIB])
        self.collector.emit(tables)
