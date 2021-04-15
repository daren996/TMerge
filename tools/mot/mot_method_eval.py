import logging
import motmetrics as mm
from motmetrics.io import Format

def compare_data_frames(methods, gt):
    accs = []
    names = []
    for method in methods:
        logging.info('evaluting method: %s', method[0])
        accs.append(mm.utils.compare_to_groundtruth(gt, method[1], 'iou', distth=0.5))
        names.append(method[0])
    return accs, names

def load_and_eval(methods_, gt_):
    methods = [(method[0],mm.io.loadtxt(method[1], fmt=Format.MOT16)) for method in methods_]
    gt = mm.io.loadtxt(gt_, fmt=Format.MOT16, min_confidence=1)
    mh = mm.metrics.create()
    accs, names = compare_data_frames(methods, gt)
    metrics = list(mm.metrics.motchallenge_metrics)
    
    logging.info('Running metrics')

    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=False)
    print(mm.io.render_summary(summary, formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')


if __name__ == '__main__':
    methods_list = [
        ('SORT', '../storage/results/mot17/MOT17-11-DPM-faster_rcnn-sort-person.txt'),
        ('DeepSORT', '../storage/results/mot17/MOT17-11-DPM-faster_rcnn-deepsort-person.txt'),
        ('Tracktor', '../storage/results/mot17/MOT17-11-DPM-faster_rcnn-tracktor-person.txt'),
        ('UMA-MOT', '../storage/results/mot17/MOT17-11-DPM-faster_rcnn-uma-person.txt'),
    ]
    gt_file = '/media/ytchen/hdd/dataset/MOT17/train/MOT17-11-DPM/gt/gt.txt'
    load_and_eval(methods_list, gt_file)
