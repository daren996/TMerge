# pylint: disable-all
# extract reid features.
import os


def extract_features_for_dataset(dataset, noexc=False, train_test='train'):
    data_path_template = '../storage/dataset/MOT17/{}/{}/img1'
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
        'mot1': '../storage/models/reid/osnet_x1_0_mot17det_softmax_epoch2.pth',
        'mot2': '../storage/models/reid/osnet_x1_0_mot17det_softmax_r2.pth',
        'mot3': '../storage/models/reid/osnet_x1_0_mot17det_softmax_r3.pth',
        'MOT17-02-FRCNN': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_02.pth',  # _bgs for bgs
        'MOT17-05-FRCNN': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_05.pth',
        'MOT17-10-FRCNN': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_10.pth',
        'MOT17-13-FRCNN': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_13.pth',
        'MOT17-04-FRCNN': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_04.pth',
        'MOT17-09-FRCNN': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_09.pth',
        'MOT17-11-FRCNN': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_11.pth',
        # 'noexc': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_noexc.pth',
        'noexc': '../storage/models/reid/osnet_x1_0_mot17det_softmax_rx_noexc.pth',
    }
    model_path_name = 'mot3'
    # gen 
    for method in methods:
        for reid_model in reid_models:
            tokens = [
                'python', 'e2e/ingestion_runner.py', 'e2e/configs/tools/gen_track_features_torchreid.py',
                '--data_path', data_path_template.format(train_test, dataset),
                '--result_path', result_template.format(dataset, method),
                '--feature_save_path', feature_save_template.format(dataset, method, reid_model, model_path_name),
                '--model_name', reid_model,
                '--model_path', model_file_mapping['noexc'] if noexc else model_file_mapping[dataset]
            ]
            command = ' '.join(tokens)
            print('working on method: ', method)
            print('executing command: ', command)
            os.system(command)


if __name__ == '__main__':
    no_exclusion = True
    # # training set
    # extract_features_for_dataset('MOT17-02-FRCNN', noexc=no_exclusion)
    # extract_features_for_dataset('MOT17-05-FRCNN', noexc=no_exclusion)
    # extract_features_for_dataset('MOT17-10-FRCNN', noexc=no_exclusion)
    # extract_features_for_dataset('MOT17-13-FRCNN', noexc=no_exclusion)
    # extract_features_for_dataset('MOT17-04-FRCNN', noexc=no_exclusion)
    # extract_features_for_dataset('MOT17-09-FRCNN', noexc=no_exclusion)
    # extract_features_for_dataset('MOT17-11-FRCNN', noexc=no_exclusion)
    # # test set
    extract_features_for_dataset('MOT17-01-FRCNN', noexc=no_exclusion, train_test='test')
    extract_features_for_dataset('MOT17-03-FRCNN', noexc=no_exclusion, train_test='test')
    extract_features_for_dataset('MOT17-06-FRCNN', noexc=no_exclusion, train_test='test')
    extract_features_for_dataset('MOT17-07-FRCNN', noexc=no_exclusion, train_test='test')
    extract_features_for_dataset('MOT17-08-FRCNN', noexc=no_exclusion, train_test='test')
    extract_features_for_dataset('MOT17-12-FRCNN', noexc=no_exclusion, train_test='test')
    extract_features_for_dataset('MOT17-14-FRCNN', noexc=no_exclusion, train_test='test')
