# pylint: disable-all
# extract reid features.
import os

def extract_features_for_dataset(dataset):
    # result_template = '../storage/results/mot17/{}/faster_rcnn-{}-person.txt'
    data_path_template = '../storage/dataset/MOT17/train/{}/img1'
    # feature_save_template = '../storage/results/mot17/{}/feats-raw/faster_rcnn-{}-person-feat-{}.pkl'
    result_template = '../storage/results/mot17/{}/filtered-tracked/faster_rcnn-{}-person.txt'
    feature_save_template = '../storage/results/mot17/{}/feats-raw-filtered/faster_rcnn-{}-person-feat-{}.pkl'
    methods = [
        'sort', 'deepsort', 'tracktor'
    ]
    reid_models = [
        'osnet_x1_0', 'resnet50_fc512'
    ]
    # gen 
    for method in methods:
        for reid_model in reid_models:
            tokens = [
                'python', 'e2e/ingestion_runner.py', 'e2e/configs/tools/gen_track_features_torchreid.py',
                '--data_path', data_path_template.format(dataset),
                '--result_path', result_template.format(dataset, method),
                '--feature_save_path', feature_save_template.format(dataset, method, reid_model),
                '--model_name', reid_model
            ]
            command = ' '.join(tokens)
            print('working on method: ', method)
            print('executing command: ', command)
            os.system(command)

if __name__ == '__main__':
    # extract_features_for_dataset('MOT17-09-DPM')
    # extract_features_for_dataset('MOT17-11-DPM')
    # extract_features_for_dataset('MOT17-13-DPM')
    # extract_features_for_dataset('MOT17-02-DPM')
    # extract_features_for_dataset('MOT17-04-DPM')
    # extract_features_for_dataset('MOT17-09-DPM')
    extract_features_for_dataset('MOT17-11-DPM')
