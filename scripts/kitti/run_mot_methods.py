# pylint: disable-all

import os


def run_all_methods(video_name, train_test='training'):
    dataset_template = '../storage/dataset/KITTI/{train_test}/image_02/{video_name}/'
    model_config = './e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py'
    checkpoint = 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'
    output_template = '../storage/results/kitti/{video_name}/{det_method}-{method}-person.txt'

    # name, file, detection_method, [params]
    params_for_faster_rcnn = [
            '--config', model_config, 
            '--checkpoint', checkpoint,
    ]
    method_lists = [
        # ('sort', 'mmt_sort_private.py', 'faster_rcnn', params_for_faster_rcnn),
        # ('deepsort','mmt_deepsort_private.py', 'faster_rcnn', params_for_faster_rcnn),
        ('tracktor', 'mmt_tracktor_private.py', 'faster_rcnn', params_for_faster_rcnn),
        # ('uma', 'uma_private.py', 'faster_rcnn', params_for_faster_rcnn),
        # ('center_track', 'centertrack_private.py', 'center_net', [])
    ]

    for py in method_lists:
        name, pyf, det_method, extra_params = py
        tokens = [
            'python', 'e2e/ingestion_runner.py', 'e2e/configs/tracking/'+pyf,
            '--path', dataset_template.format(train_test=train_test, video_name=video_name),
            '--output', output_template.format(video_name=video_name, method=name, det_method=det_method),
            *extra_params
        ]
        command = ' '.join(tokens)
        print('working on method: ', name)
        print('executing command: ', command)
        os.system(command)
        # print(' '.join(tokens))


if __name__ == '__main__':
    video_names = ['0013', '0014', '0015', '0016', '0017', '0019']
    for dn in video_names:
        run_all_methods(dn)
