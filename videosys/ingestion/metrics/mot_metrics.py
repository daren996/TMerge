import motmetrics as mm
from motmetrics.distances import iou_matrix, norm2squared_matrix
from motmetrics.mot import MOTAccumulator
from motmetrics.preprocess import preprocessResult

import numpy as np

from videosys.ingestion.base import Operator
from videosys.ingestion import fields

class MOTMetricsReporter(Operator):
    def prepare(self):
        # Pairs exceeding this threshold are marked 'do-not-pair'.
        self.__dist_threshold = 0.5
        # iou or euc
        self.__dist = 'iou'
        self.compute_dis = iou_matrix if self.__dist == 'iou' else norm2squared_matrix
        self.accumulator = MOTAccumulator()

    def process(self, tables):
        fid = tables[fields.DATA_FRAME_ID]
        track_result = tables[fields.DATA_OBJECT_TRACK]
        track_gt = tables[fields.DATA_OBJECT_TRACK_GT]
        hids = np.array([i.uid for i in track_result])
        oids = np.array([i.uid for i in track_gt])
        hboxes = np.array([i.bbox for i in track_result])
        oboxes = np.array([i.bbox for i in track_gt])
        # transform to x,y, w,h
        hboxes[:,2] = hboxes[:,2] - hboxes[:,0]
        hboxes[:,3] = hboxes[:,3] - hboxes[:,1]
        oboxes[:,2] = oboxes[:,2] - oboxes[:,0]
        oboxes[:,3] = oboxes[:,3] - oboxes[:,1]
        dists = self.compute_dis(oboxes, hboxes, self.__dist_threshold)
        
        self.accumulator.update(oids, hids, dists, fid)

    def cleanup(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.accumulator, metrics=mm.metrics.motchallenge_metrics, name='acc')

        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)
        