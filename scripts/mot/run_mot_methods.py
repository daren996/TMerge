# pylint: disable-all

import os

if __name__ == '__main__':
    dataset_name = 'MOT17-09-DPM'

    dataset_template = '/media/ytchen/hdd/dataset/MOT17/train/{dataset_name}/img1'
    model_config = './e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py'
    checkpoint = 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'
    output_template = '../storage/results/mot17/{dataset_name}/faster_rcnn-{method}-person.txt'

    method_lists = [
        ('sort', 'mmt_sort_private.py'),
        ('deepsort','mmt_deepsort_private.py'),
        ('tracktor', 'mmt_tracktor_private.py'),
        ('uma', 'uma_private.py'),
    ]

    for py in method_lists:
        name, pyf = py
        tokens = [
            'python', 'e2e/ingestion_runner.py', 'e2e/configs/tracking/'+pyf,
            '--path', dataset_template.format(dataset_name=dataset_name),
            '--config', model_config, 
            '--checkpoint', checkpoint,
            '--output', output_template.format(dataset_name=dataset_name, method=name),
        ]
        command = ' '.join(tokens)
        print('working on method: ', name)
        print('executing command: ', command)
        os.system(command)
        # print(' '.join(tokens))