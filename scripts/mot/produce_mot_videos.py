import os

def produce_all_videos(dataset_name, fps, thickness, font_scale):

    dataset_template = '../storage/dataset/MOT17/train/{dataset_name}/img1'
    result_template = '../storage/results/mot17/{dataset_name}/{det_method}-{method}-person.txt'
    gt_template='../storage/dataset/MOT17/train/{dataset_name}/gt/gt.txt'
    output_template = '../storage/results/mot17/{dataset_name}/videos'
    
    method_list = [
        ('sort', 'faster_rcnn'), 
        ('deepsort', 'faster_rcnn'),
        ('tracktor', 'faster_rcnn'), 
        ('uma', 'faster_rcnn'),
        ('center_track', 'center_net')
    ]

    common_args = [
        'python', 'e2e/ingestion_runner.py', 'e2e/configs/tools/gen_mot_result_video.py',
        '--data_path', dataset_template.format(dataset_name=dataset_name),
        '--output_folder', output_template.format(dataset_name=dataset_name),
        '--fps', fps, '--thickness', thickness, '--font_scale', font_scale,
    ]

    for m in method_list:
        tokens = [
            *common_args,
            '--result_path', 
                result_template.format(dataset_name=dataset_name, method=m[0], det_method=m[1]),
            '--output_name', '{}-{}-person'.format(m[0], m[1]),
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

if __name__ == '__main__':
    produce_all_videos('MOT17-09-DPM', '30', '2', '0.5')
    produce_all_videos('MOT17-11-DPM', '30', '3', '0.7')
    produce_all_videos('MOT17-13-DPM', '25', '2', '0.5')
