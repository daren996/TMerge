from mmtrack.apis import inference_mot, init_model
from mmtrack.core import restore_result, track2result
from mmtrack.models.mot import DeepSORT, Tracktor

import torch

from videosys.ingestion.data import ObjectTrackingResult
from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class MMTrackingMOT(Operator):

    def __init__(self, config_file, device='cuda:0'):
        self.config_file = config_file
        self.device = device
        super().__init__()

    def prepare(self):
        self.model = init_model(self.config_file, None, self.device)

    def process(self, tables):
        frame = tables[fields.DATA_FRAME]
        fid = tables[fields.DATA_FRAME_ID]
        result = inference_mot(self.model, frame, fid)
        result = result['track_results']
        bboxes, labels, ids = restore_result(result, return_ids=True)
        result = []
        for bbox, label, uid in zip(bboxes, labels, ids):
            result.append(ObjectTrackingResult(uid, label, bbox[:4], bbox[4]))
            # print(result[-1])
        tables[fields.DATA_OBJECT_TRACK] = result
        self.collector.emit(tables)

class MMTrackingSORT(Operator):
    """
    use the SORT algorithm implemented in mm-tracking lib.
    """
    def __init__(self, motion=None, tracker=None): 
        self.model = DeepSORT(
            motion=motion,
            tracker=tracker
        )
        super().__init__()
    
    def prepare(self):
        self.tracker = self.model.tracker
        self.model.to('cuda:0')

    def process(self, tables):
        num_classes = len(self.context.get(fields.META_OBJECT_DETECTION_CLASSES))
        fid = tables[fields.DATA_FRAME_ID]
        detections = tables[fields.DATA_OBJECT_DETECTION]
        img_meta = tables[fields.COMPAT_MMLIB]['img_metas'][0]

        # normalize image.
        
        # frame: H, W, C
        # images: [N, C, H, W]
        # images = torch.unsqueeze(torch.from_numpy(frame.transpose(2, 0, 1)), 0)
        image = tables[fields.COMPAT_MMLIB]['img'][0]

        # print('shape:', image.shape)
        # print('img_meta', img_meta)

        bboxes_tensor = torch.tensor([[*d.bbox, d.confidence] for d in detections])
        labels_tensor = torch.tensor([d.label for d in detections])

        with torch.no_grad():
            # convert tensor results to np.array
            bboxes, labels, ids = self.tracker.track(image, img_meta, 
                self.model, bboxes_tensor, labels_tensor, fid)
        tracking_result = track2result(bboxes, labels, ids, num_classes)
        bboxes, labels, ids = restore_result(tracking_result, return_ids=True)
        
        result = []
        for bbox, label, uid in zip(bboxes, labels, ids):
            result.append(ObjectTrackingResult(uid, label, bbox[:4], bbox[4]))
        # print(result[-1])
        tables[fields.DATA_OBJECT_TRACK] = result
        self.collector.emit(tables)
        
        
class MMTrackingDeepSORT(MMTrackingSORT):

    # pylint: disable=super-init-not-called
    def __init__(self, pretrains=None, motion=None, reid=None, tracker=None):
        self.model = DeepSORT(
            pretrains=pretrains, motion=motion, reid=reid, tracker=tracker
        )

    # pylint: disable=super-init-not-called
    def prepare(self):
        self.tracker = self.model.tracker
        self.model.to('cuda:0')

class MMTrackingTracktor(Operator):

    def __init__(self, pretrains=None, motion=None, reid=None, tracker=None):
        self.model = Tracktor(
            pretrains=pretrains, motion=motion, reid=reid, tracker=tracker
        )
    
    def prepare(self):
        self.tracker = self.model.tracker
        self.model.to('cuda:0')

    def process(self, tables):
        num_classes = len(self.context.get(fields.META_OBJECT_DETECTION_CLASSES))
        fid = tables[fields.DATA_FRAME_ID]
        detections = tables[fields.DATA_OBJECT_DETECTION]
        x = tables[fields.DATA_FEATURES]
        data = tables[fields.COMPAT_MMLIB]
        img_meta = data['img_metas'][0]
        img = data['img'][0]
            
        bboxes_tensor = torch.tensor([[*d.bbox, d.confidence] for d in detections])
        labels_tensor = torch.tensor([d.label for d in detections])

        with torch.no_grad():
            # convert tensor results to np.array
            bboxes, labels, ids = self.tracker.track(img, img_meta, 
                self.model, x, bboxes_tensor, labels_tensor, fid, rescale=True)
        tracking_result = track2result(bboxes, labels, ids, num_classes)
        bboxes, labels, ids = restore_result(tracking_result, return_ids=True)
        
        result = []
        for bbox, label, uid in zip(bboxes, labels, ids):
            result.append(ObjectTrackingResult(uid, label, bbox[:4], bbox[4]))
        # print(result[-1])
        tables[fields.DATA_OBJECT_TRACK] = result
        self.collector.emit(tables)
