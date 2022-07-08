# pylint: disable-all
# extract reid features.
import os

def extract_features_for_dataset(dataset, noexc=True, train_test='selected', loss_name='softmax'):
    mrg, wt, wx = '005', '10', '05'
    if loss_name == 'triplet':
        loss_str = 'triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)
    else:
        loss_str = 'softmax_rx_noexc'
    data_path_template = '../storage/dataset/PathTrack/{}/{}/img1'
    result_template = '../storage/results/pathtrack/{}/filtered-tracked/faster_rcnn-{}-person.txt'
    feature_save_template = '../storage/results/pathtrack/{}/feats-raw-filtered_%s' + \
        '/faster_rcnn-{}-person-feat-{}-{}.pkl'
    feature_save_template = feature_save_template % loss_str
    methods = [
        # 'sort', 'deepsort', 
        'tracktor'
    ]
    reid_models = [
        'osnet_x1_0',
        # 'resnet50_fc512'
    ]
    model_file_mapping = {
        'noexc': '../storage/models/reid/osnet_x1_0_pathtrack_{}.pth'.format(loss_str),
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
    dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000', '0Xtp77A4zF4_3000_5000', '0Xtp77A4zF4_4000_6000', ]
    # dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000', '0Xtp77A4zF4_3000_5000', '0Xtp77A4zF4_4000_6000', '0Xtp77A4zF4_5000_7000', '0Xtp77A4zF4_6000_8000', '0Xtp77A4zF4_7000_9000', '0Xtp77A4zF4_8000_10000', '0Xtp77A4zF4_9000_11000', '0Xtp77A4zF4_10000_12000', '0Xtp77A4zF4_11000_13000', '0Xtp77A4zF4_12000_14000', '0Xtp77A4zF4_13000_15000', '0Xtp77A4zF4_14000_16000', '0Xtp77A4zF4_15000_17000', '0Xtp77A4zF4_16000_18000', '0Xtp77A4zF4_17000_19000', '0Xtp77A4zF4_18000_20000', '0Xtp77A4zF4_19000_21000', '0Xtp77A4zF4_20000_21635',]
    # dataset_names = ['4LFwzgwRyrY_0_2000', '4LFwzgwRyrY_1000_3000', '4LFwzgwRyrY_2000_4000', '4LFwzgwRyrY_3000_5000', '4LFwzgwRyrY_4000_6000', '4LFwzgwRyrY_5000_7000', '4LFwzgwRyrY_6000_8000', '4LFwzgwRyrY_7000_9000', '4LFwzgwRyrY_8000_10000', '4LFwzgwRyrY_9000_11000', '4LFwzgwRyrY_10000_12000', '4LFwzgwRyrY_11000_12000', ]
    # dataset_names = ['5fhJSO5al8o_0_2000', '5fhJSO5al8o_1000_3000', '5fhJSO5al8o_2000_4000', '5fhJSO5al8o_3000_5000', '5fhJSO5al8o_4000_6000', '5fhJSO5al8o_5000_7000', '5fhJSO5al8o_6000_8000', '5fhJSO5al8o_7000_9000', '5fhJSO5al8o_8000_10000', '5fhJSO5al8o_9000_11000', '5fhJSO5al8o_10000_12000', '5fhJSO5al8o_11000_13000', '5fhJSO5al8o_12000_14000', '5fhJSO5al8o_13000_15000', '5fhJSO5al8o_14000_16000', '5fhJSO5al8o_15000_16000', ]
    # dataset_names = ['81ahuidFUWw_0_2000', '81ahuidFUWw_1000_3000', '81ahuidFUWw_2000_4000', '81ahuidFUWw_3000_5000', '81ahuidFUWw_4000_6000', '81ahuidFUWw_5000_7000', '81ahuidFUWw_6000_8000', '81ahuidFUWw_7000_9000', '81ahuidFUWw_8000_10000', '81ahuidFUWw_9000_11000', '81ahuidFUWw_10000_12000', '81ahuidFUWw_11000_13000', '81ahuidFUWw_12000_14000', '81ahuidFUWw_13000_15000', '81ahuidFUWw_14000_16000', '81ahuidFUWw_15000_17000', '81ahuidFUWw_16000_17500', ]
    # dataset_names = ['KQqFixbb-o8_0_2000', 'KQqFixbb-o8_1000_3000', 'KQqFixbb-o8_2000_4000', 'KQqFixbb-o8_3000_5000', 'KQqFixbb-o8_4000_6000', 'KQqFixbb-o8_5000_7000', 'KQqFixbb-o8_6000_8000', 'KQqFixbb-o8_7000_9000', 'KQqFixbb-o8_8000_9099']
    # dataset_names = []

    for dn in dataset_names:
        extract_features_for_dataset(dn, loss_name='triplet')
