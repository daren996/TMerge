import os

def run_on_sequence(parent_folder, sequence_name, output_folder):
    sequence_folder = os.path.join(parent_folder, sequence_name)
    output_path = os.path.join(output_folder, '{}.txt'.format(sequence_name))
    tokens = [
        'python', 'e2e/ingestion_runner.py', 
        'e2e/configs/tracking/mmt_tracktor_private.py',
        '--path', sequence_folder, 
        '--image_prefix', 'img',
        '--output', output_path,
        '--checkpoint', '../mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        '--save_type',
        '--detect_classes', 'person,car,bicycle,motorcycle,bus,truck,traffic_light,stop_sign'
    ]
    command = ' '.join(tokens)
    print('running command: ', command)
    os.system(command)

def run_mot_on_folder(folder_path, output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    all_sequences = os.listdir(folder_path)
    count = 0
    for sequence_name in all_sequences:
        count += 1
        print('processing:', sequence_name, 'progress {}/{}'.format(count, len(all_sequences)))
        run_on_sequence(folder_path, sequence_name, output_folder)

if __name__ == '__main__':
    # run_mot_on_folder(
    #     '/media/ytchen/hdd/dataset/Detrac/Insight-MVT_Annotation_Test',
    #     '../storage/results/detrac/test-mot/'
    # )
    run_mot_on_folder(
        '/media/ytchen/hdd/dataset/Detrac/Insight-MVT_Annotation_Train',
        '../storage/results/detrac/train-mot/'
    )