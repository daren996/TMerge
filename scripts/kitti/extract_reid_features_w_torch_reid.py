# pylint: disable-all
# extract reid features.
import os

def extract_features_for_dataset(video_name, noexc=True, train_test='training', loss_name='softmax'):
    mrg, wt, wx = '005', '10', '05'
    if loss_name == 'triplet':
        loss_str = 'triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)
    else:
        loss_str = 'softmax_rx_noexc'
    data_path_template = '../storage/dataset/KITTI/{}/image_02/{}/'
    result_template = '../storage/results/kitti/{}/filtered-tracked/faster_rcnn-{}-person.txt'
    feature_save_template = '../storage/results/kitti/{}/feats-raw-filtered_%s' + \
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
        'noexc': '../storage/models/reid/osnet_x1_0_kitti_{}.pth'.format(loss_str),
    }
    model_path_name = 'mot3'
    # gen 
    for method in methods:
        for reid_model in reid_models:
            tokens = [
                'python', 'e2e/ingestion_runner.py', 'e2e/configs/tools/gen_track_features_torchreid.py',
                '--data_path', data_path_template.format(train_test, video_name),
                '--result_path', result_template.format(video_name, method),
                '--feature_save_path', feature_save_template.format(video_name, method, reid_model, model_path_name),
                '--model_name', reid_model,
                '--model_path', model_file_mapping['noexc'] if noexc else model_file_mapping[video_name]
            ]
            command = ' '.join(tokens)
            print('working on method: ', method)
            print('executing command: ', command)
            os.system(command)


if __name__ == '__main__':
    video_names = ['0013', '0014', '0015', '0016', '0017', '0019']
    for vn in video_names:
        extract_features_for_dataset(vn, loss_name='triplet')

    