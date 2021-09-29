from collections import defaultdict
import os
import pandas as pd
import numpy as np
from cv2 import cv2
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tools.mot.mot_mapping import generate_summary, load_and_compute_mapping

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

def produce_track_distance(pkl_file_path, gt_file, track_file):
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
                # get min value.
                _min_distance = np.min(_distances)
                _row_idx, _col_idx = np.unravel_index(_distances.argmin(), _distances.shape)

                # print('min pos', hid, _key, _row_idx, _col_idx)
                
                _hid_fid = _current_interval_table.iloc[_row_idx]['fid']
                _other_hid_fid = _fids[_col_idx]

                if _key not in track_pair_distance_dict or \
                        track_pair_distance_dict[_key][0] > _min_distance:
                    track_pair_distance_dict[_key] = (_min_distance, _hid_fid, _other_hid_fid)

            # import sys
            # sys.exit(0)
        id_distance_tuples = []
        for _k, _v in track_pair_distance_dict.items():
            id_distance_tuples.append([_k, _v[0], _v[1], _v[2]])
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
    format_output = []
    for info, row in zip(additional_info, format_results):
        # skip no matching ids
        if '=>' not in info:
            continue
        format_output.append(info)
        format_output.append(''.join([
            '{:<20s}'.format(x) for x in row[:nn+1]
        ]))
    return format_output


def produce_images_for_tracks(additional_info, format_results, hid_result_tuples_dict, \
        nn, store_folder, feat_data, frame_path_template,\
            detailed_additional_info_dict,
            generate_raw_frames=True, generate_multi_match_only=True):
    if not os.path.isdir(store_folder):
        os.makedirs(store_folder)
    
    image_template_selected = '{folder}/{pos}-{hid}-{fid}-{dist:.4f}-SELECTED.jpg'
    image_template = '{folder}/{pos}-{hid}-{fid}-{dist:.4f}g'

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
                    _dist, _hid_fid, _other_hid_fid = other_hid_dist_dict[other_hid]
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

        for pos, (other_hid, distance, hid_fid, other_hid_fid) in enumerate(results[:nn]):
            hid_row = feat_data.loc[(feat_data['id'] == hid)\
                 & (feat_data['fid'] == hid_fid)].iloc[0]
            # print('getting value: ', other_hid, other_hid_fid)
            other_hid_row = feat_data.loc[(feat_data['id'] == other_hid)\
                 & (feat_data['fid'] == other_hid_fid)].iloc[0]

            __crop_and_save_img_from_row(hid_row, pos, hid, distance, other_hid in other_hid_set)
            __crop_and_save_img_from_row(other_hid_row, pos, other_hid, distance, other_hid in other_hid_set)
        # import sys
        # sys.exit(0)

def simple_test(dataset, method, reid_network):
    gt_template = '../storage/dataset/MOT17/train/{}/gt/gt.txt'
    method_result_template = '../storage/results/mot17/{}/faster_rcnn-{}-person.txt'
    feat_template = '../storage/results/mot17/{}/feats-raw/faster_rcnn-{}-person-feat-{}.pkl'
    # feat_template = '../storage/results/mot17/{}/faster_rcnn-{}-person-feat.pkl'
    result_path = '../storage/results/mot17/{}/reid-feat-person/{}-{}-pairwise.txt'

    frame_path_template = '../storage/dataset/MOT17/train/'+dataset+'/img1/{:06d}.jpg'
    image_result_path = '../storage/results/mot17/{}/reid-feat-person-images/{}-{}-pairwise/'

    filtered_method_template = '../storage/results/mot17/{}/filtered-tracked/faster_rcnn-{}-person.txt'
    filtered_result_path = '../storage/results/mot17/{}/reid-feat-person-filtered/{}-{}-pairwise.txt'
    filtered_image_result_path = '../storage/results/mot17/{}/reid-feat-person-images-filtered/{}-{}-pairwise/'
    filtered_feat_template ='../storage/results/mot17/{}/feats-raw-filtered/faster_rcnn-{}-person-feat-{}.pkl'

    result_path = filtered_result_path
    method_result_template = filtered_method_template
    image_result_path = filtered_image_result_path
    feat_template = filtered_feat_template

    print("processing method [{}] with reid model {}".format(method, reid_network))

    import time
    start = time.process_time()

    a, b, c, d, e = produce_track_distance(
                feat_template.format(dataset, method , reid_network), 
                gt_template.format(dataset),
                method_result_template.format(dataset, method, reid_network))
    end = time.process_time()
    print('time used', end - start)

    outputs = format_results_for_output(a, b, c, 10)

    # output.
    output_path = result_path.format(dataset, method, reid_network)
    parent_dir = os.path.dirname(output_path)
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    with open(output_path, 'w') as f:
        f.write('\n'.join(outputs))
    
    produce_images_for_tracks(a, b, c, 10, \
        image_result_path.format(dataset, method, reid_network), \
            d, frame_path_template, e, generate_raw_frames=True)

def test_dataset(dataset):
    print('processing dataset', dataset)
    methods = ['sort', 'deepsort', 'tracktor']
    reid_models = ['osnet_x1_0', 'resnet50_fc512']

    for method in methods:
        for model in reid_models:
            simple_test(dataset, method, model)

if __name__ == '__main__':
    # simple_test('MOT17-11-DPM', 'deepsort', 'osnet_x1_0')
    # simple_test('MOT17-13-DPM', 'deepsort', 'osnet_x1_0')
    # test_dataset('MOT17-11-DPM')
    # test_dataset('MOT17-09-DPM')
    # test_dataset('MOT17-13-DPM')
    # test_dataset('MOT17-02-DPM')
    # test_dataset('MOT17-04-DPM')
    # simple_test('MOT17-02-DPM', 'deepsort', 'osnet_x1_0')
    # simple_test('MOT17-04-DPM', 'deepsort', 'osnet_x1_0')
    # simple_test('MOT17-04-DPM', 'tracktor', 'osnet_x1_0')
    # simple_test('MOT17-09-DPM', 'tracktor', 'osnet_x1_0')
    simple_test('MOT17-11-DPM', 'tracktor', 'osnet_x1_0')
    