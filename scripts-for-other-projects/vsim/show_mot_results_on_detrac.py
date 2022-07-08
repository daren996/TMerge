import os

def show_result(sequence_folder, result_folder, sequence_name, start_frame):
    sequence_path = os.path.join(sequence_folder, sequence_name)
    result_path = os.path.join(result_folder, '{}.txt'.format(sequence_name))
    
    tokens = [
        'python', 'e2e/ingestion_runner.py', 
        'e2e/configs/tools/visualize_mot_result.py',
        '--data_path', sequence_path,
        '--result_path', result_path,
        '--start_frame', start_frame,
        '--image_prefix', 'img',
        '--thickness','1',
        '--font_scale', '0.3',
    ]
    tokens = [str(x) for x in tokens]
    os.system(' '.join(tokens))

if __name__ == '__main__':
    show_result(
        '/media/ytchen/hdd/dataset/Detrac/Insight-MVT_Annotation_Test',
        '../storage/results/detrac/test-mot/',
        'MVI_40701',1120,
    )
