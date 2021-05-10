# pylint: disable-all

import logging
import os
import motmetrics as mm
from motmetrics.io import Format
import pandas as pd

def run_datasets(dataset_list):

    dataset_template = '../storage/dataset/MOT17/train/{dataset_name}/img1'
    model_config = './e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py'
    checkpoint = 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'
    output_template = '../storage/results/mot17/{datasets}/{det_method}-{method}-person.txt'

    # name, file, detection_method, [params]
    params_for_faster_rcnn = [
            '--config', model_config, 
            '--checkpoint', checkpoint,
    ]
    method_lists = [
        ('sort', 'mmt_sort_private.py', 'faster_rcnn', params_for_faster_rcnn),
        ('deepsort','mmt_deepsort_private.py', 'faster_rcnn', params_for_faster_rcnn),
        ('tracktor', 'mmt_tracktor_private.py', 'faster_rcnn', params_for_faster_rcnn),
        ('uma', 'uma_private.py', 'faster_rcnn', params_for_faster_rcnn),
        ('center_track', 'centertrack_private.py', 'center_net', [])
    ]

    for py in method_lists:
        name, pyf, det_method, extra_params = py
        pathstr = ':'.join([dataset_template.format(dataset_name=d) for d in dataset_list])
        tokens = [
            'python', 'e2e/ingestion_runner.py', 'e2e/configs/tracking/'+pyf,
            '--path', pathstr,
            '--output', output_template.format(datasets='_'.join([d for d in dataset_list]), 
                method=name, det_method=det_method),
            *extra_params
        ]
        command = ' '.join(tokens)
        print('working on method: ', name)
        print('executing command: ', command)
        os.system(command)
        # print(' '.join(tokens))
    pass

def eval_result_for_datasets(dataset_list):
    methods_list = [
        ('SORT', 'sort', 'faster_rcnn'),
        ('DeepSORT', 'deepsort', 'faster_rcnn'),
        ('Tracktor', 'tracktor', 'faster_rcnn'),
        ('UMA-MOT', 'uma', 'faster_rcnn'),
        ('CenterTrack', 'center_track', 'center_net')
    ]

    dataset_frame_count={
        'MOT17-09-DPM': 525,
        'MOT17-11-DPM': 900,
        'MOT17-13-DPM': 750,
    }

    gt_file = '../storage/dataset/MOT17/train/{dataset}/gt/gt.txt'
    # without concate
    single_result_path_template = '../storage/results/mot17/{dataset}/{det_method}-{method}-person.txt'
    # with concate
    concate_result_template = '../storage/results/mot17/{datasets}/{det_method}-{method}-person.txt'
    
    output_template = '../storage/results/mot17/{datasets}/summary.txt'

    all_gt = pd.DataFrame()
    frame_count = 0
    for idx, dataset in enumerate(dataset_list):
        # load gt
        gt = mm.io.loadtxt(gt_file.format(dataset=dataset), fmt=Format.MOT16, min_confidence=1)
        # gt.loc([0])['Id'].apply(lambda x : x+1000*idx, inplace=True)
        # update index.
        gt.reset_index(inplace=True)
        gt['Id'] = gt['Id'].apply(lambda x : x+1000*idx)
        gt['FrameId'] = gt['FrameId'].apply(lambda x : x+ frame_count)
        frame_count += dataset_frame_count[dataset]
        all_gt = all_gt.append(gt)

    all_gt.set_index(['FrameId', 'Id'], inplace=True)
    # print(all_gt)

    # eval
    mh = mm.metrics.create()
    accs = []
    names = []
    
    logging.info('running metrics')

    for method_name, method, det_method in methods_list:
        # without concate
        frame_count = 0
        all_no_concate = pd.DataFrame()

        for idx, dataset in enumerate(dataset_list):
            no_concate_pd = mm.io.loadtxt(single_result_path_template.format(
                dataset=dataset, det_method=det_method, method=method
            ), fmt=Format.MOT16)
            no_concate_pd.reset_index(inplace=True)
            no_concate_pd['Id'] = no_concate_pd['Id'].apply(lambda x : x + 1000*idx)
            no_concate_pd['FrameId'] = no_concate_pd['FrameId'].apply(lambda x: x+frame_count)
            frame_count += dataset_frame_count[dataset]

            all_no_concate = all_no_concate.append(no_concate_pd)
        all_no_concate.set_index(['FrameId', 'Id'], inplace=True)

        # load concate
        all_concate = mm.io.loadtxt(concate_result_template.format(
            datasets='_'.join([v for v in dataset_list]),
            det_method=det_method, method=method
        ))

        logging.info('evaluting method: %s', method_name)
        accs.append(mm.utils.compare_to_groundtruth(all_gt, all_no_concate, 'iou', distth=0.5))
        names.append('{}-seperate'.format(method_name))
        accs.append(mm.utils.compare_to_groundtruth(all_gt, all_concate, 'iou', distth=0.5))
        names.append('{}-concate'.format(method_name))

    summary = mh.compute_many(accs, names=names, metrics=list(mm.metrics.motchallenge_metrics), 
        generate_overall=False)
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names)
    print(strsummary)
    with open(output_template.format(datasets='_'.join([v for v in dataset_list])), 'w') as f:
        f.write(strsummary)
    logging.info("done")



if __name__ == '__main__':
    # run_datasets(['MOT17-09-DPM', 'MOT17-11-DPM'])
    # run_datasets(['MOT17-11-DPM', 'MOT17-13-DPM'])
    # run_datasets(['MOT17-09-DPM', 'MOT17-13-DPM'])
    # eval_result_for_datasets(['MOT17-09-DPM', 'MOT17-11-DPM'])
    eval_result_for_datasets(['MOT17-11-DPM', 'MOT17-13-DPM'])
    eval_result_for_datasets(['MOT17-09-DPM', 'MOT17-13-DPM'])
