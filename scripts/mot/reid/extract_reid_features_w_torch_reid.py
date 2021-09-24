# pylint: disable-all
# extract reid features.
import os


def extract_features_for_dataset(dataset):
    # result_template = '../storage/results/mot17/{}/faster_rcnn-{}-person.txt'
    data_path_template = '../storage/dataset/MOT17/train/{}/img1'
    # feature_save_template = '../storage/results/mot17/{}/feats-raw/faster_rcnn-{}-person-feat-{}.pkl'
    result_template = '../storage/results/mot17/{}/filtered-tracked/faster_rcnn-{}-person.txt'
    feature_save_template = '../storage/results/mot17/{}/feats-raw-filtered/faster_rcnn-{}-person-feat-{}-{}.pkl'
    methods = [
        # 'sort', 'deepsort', 
        'tracktor'
    ]
    reid_models = [
        'osnet_x1_0',
        # 'resnet50_fc512'
    ]
    model_file_mapping = {
        # default is
        'pretrained': '',
        'mot1': '../storage/models/reid/osnet_x1_0_mot17det_softmax_epoch2.pth',
        'mot2': '../storage/models/reid/osnet_x1_0_mot17det_softmax_r2.pth',
        'mot3': '../storage/models/reid/osnet_x1_0_mot17det_softmax_r3.pth',
        'MOT17-02-DPM': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_02.pth',
        'MOT17-04-DPM': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_04.pth',
        'MOT17-05-DPM': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_05.pth',
        'MOT17-09-DPM': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_09.pth',
        'MOT17-10-DPM': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_10.pth',
        'MOT17-11-DPM': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_11.pth',
        'MOT17-13-DPM': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_13.pth',
    }
    model_path_name = 'mot3'
    # gen 
    for method in methods:
        for reid_model in reid_models:
            tokens = [
                'python', 'e2e/ingestion_runner.py', 'e2e/configs/tools/gen_track_features_torchreid.py',
                '--data_path', data_path_template.format(dataset),
                '--result_path', result_template.format(dataset, method),
                '--feature_save_path', feature_save_template.format(dataset, method, reid_model, model_path_name),
                '--model_name', reid_model,
                # '--model_path', model_file_mapping[model_path_name]
                '--model_path', model_file_mapping[dataset]
            ]
            command = ' '.join(tokens)
            print('working on method: ', method)
            print('executing command: ', command)
            os.system(command)


if __name__ == '__main__':
    extract_features_for_dataset('MOT17-02-DPM')
    extract_features_for_dataset('MOT17-04-DPM')
    extract_features_for_dataset('MOT17-05-DPM')
    extract_features_for_dataset('MOT17-09-DPM')
    extract_features_for_dataset('MOT17-10-DPM')
    extract_features_for_dataset('MOT17-11-DPM')
    extract_features_for_dataset('MOT17-13-DPM')
