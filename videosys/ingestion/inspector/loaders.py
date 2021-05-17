from videosys.ingestion.data import TrackFeature
import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from mmtrack.models.reid.base_reid import BaseReID
from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class ExtractFeature(Operator):
    def __init__(self, backbone, neck, head, img_scale = None, rescale = False):
        super().__init__()
        self.img_scale = img_scale
        self.rescale = rescale
        self.model = BaseReID(backbone=backbone, neck=neck, head=head)
    
    def prepare(self):
        self.model.to('cuda:0')
        self.model.eval()

    def process(self, tables):
        mmlibdata = tables[fields.COMPAT_MMLIB]
        reid_img = mmlibdata['img'][0]
        img_metas = mmlibdata['img_metas'][0]
        tracks = tables[fields.DATA_OBJECT_TRACK]

        # print('meta', img_metas)
        # print('shapes', reid_img.shape, len(img_metas))
        
        # collect bbox
        bboxes = []
        for track in tracks:
            # print(track.uid, track.bbox)
            bboxes.append(track.bbox)
            # scale to size.,
            
        if len(bboxes) == 0:
            tables[fields.DATA_TRACK_FEAT] = []
        else:
            with torch.no_grad():
                features = self.model.simple_test(
                    self.crop_imgs(reid_img, img_metas, torch.tensor(bboxes), self.rescale)
                )
            
            result = []
            for t, feat in zip(tracks, features):
                result.append(TrackFeature(t.uid, t.bbox, feat.cpu()))

            tables[fields.DATA_TRACK_FEAT] = result
        self.collector.emit(tables)

    def crop_imgs(self, img, img_metas, bboxes, rescale=False):
        """Crop the images according to some bounding boxes. Typically for re-
        identification sub-module.

        Modified from 
        https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/models/mot/trackers/base_tracker.py

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            bboxes (Tensor): of shape (N, 4) or (N, 5).
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the scale of the image. Defaults to False.

        Returns:
            Tensor: Image tensor of shape (N, C, H, W).
        """
        h, w, _ = img_metas[0]['img_shape']
        img = img[:, :, :h, :w]
        if rescale:
            bboxes[:, :4] *= torch.tensor(img_metas[0]['scale_factor']).to(
                bboxes.device)
        bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], min=0, max=w)
        bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], min=0, max=h)

        crop_imgs = []
        for bbox in bboxes:
            # print(bbox)
            x1, y1, x2, y2 = map(int, bbox)
            if x2 == x1:
                x2 = x1 + 1
            if y2 == y1:
                y2 = y1 + 1
            crop_img = img[:, :, y1:y2, x1:x2]
            if self.img_scale:
                crop_img = F.interpolate(
                    crop_img,
                    size=self.img_scale,
                    mode='bilinear',
                    align_corners=False)
            crop_imgs.append(crop_img)

        if len(crop_imgs) > 0:
            return torch.cat(crop_imgs, dim=0)
        else:
            return img.new_zeros((0, ))


class TrackFeatureSink(Operator):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        parent_folder = os.path.dirname(file_path)
        if not os.path.isdir(parent_folder):
            os.makedirs(parent_folder)
        self.data = []
        self.head = ['fid', 'left', 'top', 'right', 'bottom', 'id', 'label', 'score', 'feature']

    def process(self, tables):
        frame_id = tables[fields.DATA_FRAME_ID]
        tracks = tables[fields.DATA_OBJECT_TRACK]
        track_feats = tables[fields.DATA_TRACK_FEAT]

        # save as numpy array: [fid, bboxes, id, label, score]
        track_dict = dict()
        for track in tracks:
            track_dict[track.uid] = track
        
        for track_f in track_feats:
            track = track_dict[track_f.uid]
            # print(track_f.feature.get_device())
            self.data.append([frame_id, *track_f.bbox, track_f.uid, \
                track.label, track.confidence, [track_f.feature]])

        self.collector.emit(tables)
        
    def cleanup(self):
        df = pd.DataFrame(data = np.array(self.data), columns = self.head)
        df.to_pickle(self.file_path)
