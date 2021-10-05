from collections import defaultdict
import os
import sys 
sys.path.append('.')
import pandas as pd
import numpy as np
from cv2 import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tools.mot.mot_mapping import generate_summary, load_and_compute_mapping

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

def to_intervals(arr):
    _start = arr[0]
    intervals = []
    last_interval = [_start, _start]
    intervals.append(last_interval)
    for _next in arr[1:]:
        if _next == _start + 1:
            _start = _next
            last_interval[1] = _next
        else:
            _start = _next
            last_interval = [_start, _start]
            intervals.append(last_interval)
    return intervals

def produce_track_distance(pkl_file_path, gt_file, track_file, select_method):
    oid_results, event_df = load_and_compute_mapping(track_file, gt_file)
    mapping_pd = generate_summary(oid_results, event_df)

    oid_hids = [(x, y) for x, y in zip(mapping_pd['oid'], mapping_pd['hids_detail'])]
    hid_oids_dict = defaultdict(list)
    for oid, hids in oid_hids:
        for hid in hids:
            hid_oids_dict[hid].append(oid)

    # convert types
    feat_data = pd.read_pickle(pkl_file_path).astype({
        'fid': 'int32',
        'id': 'int32',
    })
    # filter feat_data 

    # print(feat_data.dtypes)

    unique_ids = feat_data['id'].unique()

    format_results = [] 
    # 3. additional info for each hid.
    additional_info = []
    # detailed_additional info
    detailed_additional_info_dict = dict()

    hid_result_tuples_dict = dict()
    for hid in unique_ids:
        # 1. get intervals
        hid_intervals = to_intervals(feat_data.loc[feat_data['id'] == hid]['fid'].tolist())

        # key: other_hid, value: (distance, hid_frame, other_hid_frame)
        track_pair_distance_dict = dict()

        # for all intervals, retrieve images
        for [_interval_start, _interval_end] in hid_intervals:
            
            _current_interval_table = feat_data.loc[
                (feat_data['fid'] >= _interval_start) \
                    & (feat_data['fid'] <= _interval_end) \
                    & (feat_data['id'] == hid)
                ]
            _current_interval_features = [x[0].numpy() for x in _current_interval_table['feature']]
            

            tracks_in_current_frame = set(feat_data.loc[feat_data['fid'] == \
                _interval_start]['id'].tolist())

            # use interval_start 
            appeared_track_table = feat_data.loc[(feat_data['fid'] < _interval_start) \
                & (~feat_data['id'].isin(tracks_in_current_frame))]

            # get the centroid of this track.
            appeared_ids = appeared_track_table['id'].unique()

            tracks_in_current_frame = set(feat_data.loc[feat_data['fid'] == \
                _interval_end]['id'].tolist())

            future_track_table = feat_data.loc[(feat_data['fid'] > _interval_end) \
                & (~feat_data['id'].isin(tracks_in_current_frame))]
            
            future_ids = future_track_table['id'].unique()
            
            id_and_feat_dict = dict()
            for other_hid in appeared_ids:
                # get features
                this_hid_table = appeared_track_table.loc[appeared_track_table['id'] == other_hid]
                if len(this_hid_table) <= 0:
                    continue
                _feats = [x[0].numpy() for x in this_hid_table['feature']]
                
                _fids = [x for x in this_hid_table['fid']]
                id_and_feat_dict[other_hid] = (_feats, _fids)
            
            for other_hid in future_ids:
                # get features
                this_hid_table = future_track_table.loc[future_track_table['id'] == other_hid]
                if len(this_hid_table) <= 0:
                    continue
                _feats = [x[0].numpy() for x in this_hid_table['feature']]
                
                _fids = [x for x in this_hid_table['fid']]
                if other_hid in id_and_feat_dict:
                    id_and_feat_dict[other_hid][0].extend(_feats)
                    id_and_feat_dict[other_hid][1].extend(_fids)
                else:
                    id_and_feat_dict[other_hid] = (_feats, _fids)

            # loop other tracks
            for _key, (_features, _fids) in id_and_feat_dict.items():
                _distances = pairwise_distances(_current_interval_features, _features)
                # get average value.
                if select_method == 'avg':
                    _select_distance = np.average(_distances)
                    # pick the first frame for each track
                    _row_idx, _col_idx = 0, 0
                elif select_method == 'median':
                    _select_distance = np.median(_distances)
                    _idx = np.argsort(_distances, None)[_distances.size//2]
                    _row_idx, _col_idx = np.unravel_index(_idx, _distances.shape)
                else:
                    print('error: expecting avg or median')
                    import sys
                    sys.exit(-1)
                _hid_fid = _current_interval_table.iloc[_row_idx]['fid']
                _other_hid_fid = _fids[_col_idx]
                # _min_distance = np.min(_distances)
                # _row_idx, _col_idx = np.unravel_index(_distances.argmin(), _distances.shape)

                # print('min pos', hid, _key, _row_idx, _col_idx)

                if _key not in track_pair_distance_dict or \
                        track_pair_distance_dict[_key][0] > _select_distance:
                    track_pair_distance_dict[_key] = (_select_distance, _hid_fid, _other_hid_fid, _distances)

                # draw distance distribution.


            # import sys
            # sys.exit(0)
        id_distance_tuples = []
        for _k, _v in track_pair_distance_dict.items():
            id_distance_tuples.append([_k, _v[0], _v[1], _v[2], _v[3]])
        id_distance_tuples.sort(key= lambda x: x[1])

        # print(id_distance_tuples[:10])

        # o ids.
        oids = hid_oids_dict[hid]

        oids_info = []
        detailed_oids_info = []
        for oid in oids:
            # compute distance
            oid_row = mapping_pd.loc[mapping_pd['oid'] == oid]
            # compute distance between hid and row[0]
            hids_row = list(map(int, oid_row['hids_detail'].iloc[0]))
            dists_row = []
            hid_distance_type = []
            for cell in hids_row:
                if cell == hid:
                    dists_row.append(0)
                    hid_distance_type.append('x')
                    continue
                # get from dict
                if cell in track_pair_distance_dict:
                    value = track_pair_distance_dict[cell]
                    dists_row.append(value[0])
                    hid_distance_type.append('[{},{}]'.format(value[1], value[2]))
            # print(hids_row, dists_row, hid_distance_type)
            dist_format = ['{:.0f}:{}/{:.4f}'.format(x,z, y) \
                for x, y, z in zip(hids_row, dists_row, hid_distance_type)]
            
            if len(hids_row) > 1:
                detailed_oids_info.append((oid, [(x, y, z) \
                    for x, y, z in zip(hids_row, dists_row, hid_distance_type)]))

            if len(hids_row) > 1:
                oids_info.append('=>oid:{:.0f}, hids: {}'.format(oid, ';'.join(dist_format)))
            else:
                oids_info.append('  oid:{:.0f}, hids: {}'.format(oid, ';'.join(dist_format)))
        additional_info.append(' | '.join(oids_info if len(oids_info) > 0 else ['NO MATCHING OID']))

        detailed_additional_info_dict[hid] = detailed_oids_info

        row_result = [str(hid)]
        for cell in id_distance_tuples:
            cell_oids = hid_oids_dict[cell[0]]
            # print('oids', oids, cell_oids)
            if len(set(oids) & set(cell_oids)) > 0:
                row_result.append('*{}:[{},{}]/{:.4f}'.format(cell[0], cell[2], cell[3], cell[1]))
            else:
                row_result.append(' {}:[{},{}]/{:.4f}'.format(cell[0], cell[2], cell[3], cell[1]))
        format_results.append(row_result)
        # print(oids_info)
        # print(format_results)
        # import sys
        # sys.exit(0)
        hid_result_tuples_dict[hid] = id_distance_tuples
    return additional_info, format_results, hid_result_tuples_dict, feat_data, detailed_additional_info_dict

    # print(first_fid_rows)
    # print(last_fid_rows)

def format_results_for_output(additional_info, format_results, hid_result_tuples_dict, nn):
    format_output, format_out_dis = [], []
    for info, row in zip(additional_info, format_results):
        # skip no matching ids
        if '=>' not in info:
            continue
        format_output.append(info)
        format_output.append(''.join([
            '{:<20s}'.format(x) for x in row[:nn+1]
        ]))
    format_out_dis.append('# HId1-HId2;min;max;avg;median;std;avg-std;avg+std')
    for hid, results in hid_result_tuples_dict.items():
        for _, (other_hid, _, _, _, _distances) in enumerate(results[:nn]):
            _min = np.min(_distances)
            _max = np.max(_distances)
            _average = np.average(_distances)
            _median = np.median(_distances)
            _std = np.std(_distances)
            format_out_dis.append('%s-%s;%.2f;%.2f;%.2f;%.2f;%.2f;%.2f;%.2f' 
            % (hid, other_hid, _min, _max, _average, _median, _std, _average-_std, _average+_std))
    return format_output, format_out_dis


def produce_images_for_tracks(additional_info, format_results, hid_result_tuples_dict, \
        nn, store_folder, feat_data, frame_path_template,\
            detailed_additional_info_dict,
            generate_raw_frames=True, generate_multi_match_only=False):
    if not os.path.isdir(store_folder):
        os.makedirs(store_folder)
    
    image_template_selected = '{folder}/{pos}-{hid}-{fid}-{dist:.4f}-SELECTED.jpg'
    image_template = '{folder}/{pos}-{hid}-{fid}-{dist:.4f}.jpg'

    distribution_template_selected = '{folder}/{pos}-{hid}-distribution-SELECTED.jpg'
    distribution_template = '{folder}/{pos}-{hid}-distribution.jpg'

    def __clamp(x, minimum, maximum):
        return max(minimum, min(x, maximum))

    def __crop_image(hid_row):
        # write hid image.
        image = cv2.imread(frame_path_template.format(hid_row['fid']))
        [h,w,c] = image.shape
        hid_bbox = [
            __clamp(int(hid_row['left']), 0, w), __clamp(int(hid_row['top']), 0, h), 
            __clamp(int(hid_row['right']), 0, w), __clamp(int(hid_row['bottom']), 0, h)
        ]
        # raw_bbox = [
        #     hid_row['left'], hid_row['top'], hid_row['right'], hid_row['bottom']
        # ]
        # print('reading file from ', frame_path_template.format(hid_row['fid']))
        if hid_bbox[2] == hid_bbox[0]:
            hid_bbox[2] += 1
        if hid_bbox[3] == hid_bbox[1]:
            hid_bbox[3] += 1
        # print('bbox', hid_bbox, raw_bbox, w, h)

        image = image[hid_bbox[1]:hid_bbox[3], hid_bbox[0]:hid_bbox[2]]
        return image

    def __crop_and_save_img_from_row(hid_row, pos, hid, distance, selected=False):
        # print('image shape', hid, image.shape)
        template = image_template_selected if selected else image_template
        cv2.imwrite(template.format(
            folder=hid_folder, pos=pos, hid=hid, fid=hid_row['fid'], dist=distance, 
            selected=selected
        ), __crop_image(hid_row))

    for hid, results in hid_result_tuples_dict.items():
        
        other_hid_dist_dict = dict()
        for [ohid, *rest] in results:
            other_hid_dist_dict[ohid] = rest

        hid_folder = '{}/{}'.format(store_folder, hid)

        oids_info = detailed_additional_info_dict[hid]

        other_hid_set = set()
        hid_oids = defaultdict(list)
        pos = 0
        for oid, hid_list in oids_info:
            for other_hid, other_hid_dist, other_hid_type in hid_list:
                # save.
                if other_hid != hid and other_hid in other_hid_dist_dict:
                    pos += 1
                    if not os.path.isdir(hid_folder):
                        os.makedirs(hid_folder)
                    hid_oids[hid].append(oid)
                    # generate 
                    other_hid_frames_folder = hid_folder + '/matched_gt_frames/'
                    if not os.path.isdir(other_hid_frames_folder):
                        os.makedirs(other_hid_frames_folder)

                    # save it.
                    _dist, _hid_fid, _other_hid_fid, _distances = other_hid_dist_dict[other_hid]
                    _hid_row = feat_data.loc[(feat_data['id'] == hid)\
                        & (feat_data['fid'] == _hid_fid)].iloc[0]
                    _other_hid_row = feat_data.loc[(feat_data['id'] == other_hid)\
                        & (feat_data['fid'] == _other_hid_fid)].iloc[0]
                    cv2.imwrite(image_template.format(
                        folder=other_hid_frames_folder, pos=pos, hid=hid, 
                        fid=_hid_row['fid'], dist=_dist
                    ), __crop_image(_hid_row))

                    cv2.imwrite(image_template.format(
                        folder=other_hid_frames_folder, pos=pos, hid=other_hid, 
                        fid=_other_hid_row['fid'], dist=_dist
                    ), __crop_image(_other_hid_row))
                    other_hid_set.add(other_hid)
        # skip if no other hid shares the same oid.
        if generate_multi_match_only and pos == 0:
            continue

        if not os.path.isdir(hid_folder):
            os.makedirs(hid_folder)
        # store all bounding boxes of hid
        if generate_raw_frames:
            hid_raw_frames_folder = hid_folder + '/frames/'
            if not os.path.isdir(hid_raw_frames_folder):
                os.makedirs(hid_raw_frames_folder)
            for idx, _hid_row in feat_data.loc[feat_data['id'] == hid].iterrows():
                _image = __crop_image(_hid_row)
                cv2.imwrite('{}/{}.jpg'.format(hid_raw_frames_folder, _hid_row['fid']), _image)

        for pos, (other_hid, distance, hid_fid, other_hid_fid, _distances) in enumerate(results[:nn]):
            hid_row = feat_data.loc[(feat_data['id'] == hid)\
                 & (feat_data['fid'] == hid_fid)].iloc[0]
            # print('getting value: ', other_hid, other_hid_fid)
            other_hid_row = feat_data.loc[(feat_data['id'] == other_hid)\
                 & (feat_data['fid'] == other_hid_fid)].iloc[0]

            _is_selected = other_hid in other_hid_set
            __crop_and_save_img_from_row(hid_row, pos, hid, distance, _is_selected)
            __crop_and_save_img_from_row(other_hid_row, pos, other_hid, distance, _is_selected)
            # save distance distribution

            _template = distribution_template_selected if _is_selected else distribution_template
            plot_distribution(_distances, _template.format(
                folder=hid_folder, pos=pos, hid=other_hid, 
                selected=_is_selected), hid, other_hid)

        # import sys
        # sys.exit(0)

def plot_distribution(distances, file_path=None, hid=-1, other_hid=-1):
    snsplot = sns.displot(distances.flatten())
    fig = snsplot.fig
    ax = snsplot.ax
    _min = np.min(distances)
    _max = np.max(distances)
    _average = np.average(distances)
    _median = np.median(distances)
    _std = np.std(distances)
    ax.axvline(_average, color='red', linestyle='--', alpha=0.7)
    ax.axvspan(_average - _std, _average + _std, facecolor='green', alpha=0.2)
    ax.text(.6, .9, \
        'min:%.2f;\nmax:%.2f;\navg:%.2f;\nmedian:%.2f;\nstd:%.2f;\navg-std:%.2f;\navg+std:%.2f'
        % (_min, _max, _average, _median, _std, _average-_std, _average+_std), \
        verticalalignment='top', horizontalalignment='left', transform = ax.transAxes, fontsize=15)
    plt.title('HId:%s & HId:%s' % (hid, other_hid))
    plt.tight_layout()
    if file_path is None:
        plt.show()
    else:
        fig.savefig(file_path)
    plt.close(fig)

def simple_test(dataset, method, reid_network, select_method, reid_model_pth_name='pretrained'):

    # model_file_mapping = {
    #     # default is
    #     'pretrained': '',
    #     'mot1': '../storage/models/reid/osnet_x1_0_mot17det_softmax_epoch2.pth',
    #     'mot2': '../storage/models/reid/osnet_x1_0_mot17det_softmax_r2.pth',
    #     'mot3': '../storage/models/reid/osnet_x1_0_mot17det_softmax_r3.pth',
    # }
    # reid_model_pth = model_file_mapping[reid_model_pth_name]

    gt_template = '../storage/dataset/MOT17/train/{}/gt/gt.txt'
    method_result_template = '../storage/results/mot17/{}/faster_rcnn-{}-person.txt'
    feat_template = '../storage/results/mot17/{}/feats-raw/faster_rcnn-{}-person-feat-{}-{}.pkl'
    # feat_template = '../storage/results/mot17/{}/faster_rcnn-{}-person-feat.pkl'
    result_path = '../storage/results/mot17/{}/reid-feat-person/{}-{}-pairwise-{}-{}.txt'

    frame_path_template = '../storage/dataset/MOT17BGS/train/'+dataset[:-6]+'/img1/{:06d}.jpg'
    # frame_path_template = '../storage/dataset/MOT17/train/'+dataset+'/img1/{:06d}.jpg'
    image_result_path = '../storage/results/mot17/{}/reid-feat-person-images/{}-{}-pairwise-{}-{}/'

    filtered_method_template = '../storage/results/mot17/{}/filtered-tracked/faster_rcnn-{}-person.txt'
    filtered_result_path = '../storage/results/mot17/{}/reid-feat-person-filtered/{}-{}-pairwise-{}-{}.txt'
    filtered_image_result_path = '../storage/results/mot17/{}/reid-feat-person-images-filtered/{}-{}-pairwise-{}-{}/'
    filtered_feat_template ='../storage/results/mot17/{}/feats-raw-filtered/faster_rcnn-{}-person-feat-{}-{}.pkl'
    filtered_dis_path = '../storage/results/mot17/{}/reid-feat-person-filtered/{}-{}-dis-{}.txt'
 
    result_path = filtered_result_path
    dis_path = filtered_dis_path
    method_result_template = filtered_method_template
    image_result_path = filtered_image_result_path
    feat_template = filtered_feat_template

    print("processing method [{}] with reid model {}".format(method, reid_network))

    import time
    start = time.process_time()

    a, b, c, d, e = produce_track_distance(
                feat_template.format(dataset, method , reid_network, reid_model_pth_name), 
                gt_template.format(dataset),
                method_result_template.format(dataset, method, reid_network, select_method, reid_model_pth_name), 
                select_method
                )
    end = time.process_time()
    print('time used', end - start)

    outputs, out_dis = format_results_for_output(a, b, c, 10)

    # output.
    output_path = result_path.format(dataset, method, reid_network, select_method, reid_model_pth_name)
    parent_dir = os.path.dirname(output_path)
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    with open(output_path, 'w') as f:
        f.write('\n'.join(outputs))
    # distance file
    output_dis_path = dis_path.format(dataset, method, reid_network, reid_model_pth_name)
    with open(output_dis_path, 'w') as f:
        f.write('\n'.join(out_dis))
    
    produce_images_for_tracks(a, b, c, 10, \
        image_result_path.format(dataset, method, reid_network, select_method, reid_model_pth_name), \
            d, frame_path_template, e, generate_raw_frames=True)

def test_dataset(dataset):
    print('processing dataset', dataset)
    methods = ['sort', 'deepsort', 'tracktor']
    reid_models = ['osnet_x1_0', 'resnet50_fc512']
    for method in methods:
        for model in reid_models:
            simple_test(dataset, method, model, 'avg')
            simple_test(dataset, method, model, 'median')

if __name__ == '__main__':
    # test_dataset('MOT17-09-DPM')
    
    for did in ['04', '09', '11']:  # '04', '09', '10', '11'
        simple_test('MOT17-%s-FRCNN' % did, 'tracktor', 'osnet_x1_0', 'median', 'mot3')
        simple_test('MOT17-%s-FRCNN' % did, 'tracktor', 'osnet_x1_0', 'avg', 'mot3')
