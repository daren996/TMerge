import os

if __name__ == '__main__':
    dataset_name = 'MOT17-09-DPM'
    fps = '30'

    dataset_template = '/media/ytchen/hdd/dataset/MOT17/train/{dataset_name}/img1'
    result_template = '../storage/results/mot17/{dataset_name}/faster_rcnn-{method}-person.txt'
    gt_template='/media/ytchen/hdd/dataset/MOT17/train/{dataset_name}/gt/gt.txt'
    output_template = '../storage/results/mot17/{dataset_name}/videos'
    
    method_list = [
        'sort', 'deepsort', 'tracktor', 'uma'
    ]

    common_args = [
        'python', 'e2e/ingestion_runner.py', 'e2e/configs/tools/gen_mot_result_video.py',
        '--data_path', dataset_template.format(dataset_name=dataset_name),
        '--output_folder', output_template.format(dataset_name=dataset_name),
        '--fps', fps, '--thickness', '2', '--font_scale', '0.5',
    ]

    for m in method_list:
        tokens = [
            *common_args,
            '--result_path', result_template.format(dataset_name=dataset_name, method=m),
            '--output_name', '{}-person'.format(m),
        ]
        command = ' '.join(tokens)
        print('processing method: ', m)
        print('exeucting command: ', command)
        os.system(command)
    
    # produce gt

    tokens = [
        *common_args,
        '--result_path', gt_template.format(dataset_name=dataset_name),
        '--output_name', 'gt',
    ]
    command = ' '.join(tokens)
    print('processing method: gt')
    print('exeucting command: ', command)
    os.system(command)
