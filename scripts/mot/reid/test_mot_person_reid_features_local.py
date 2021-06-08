from collections import defaultdict
import os
import pandas as pd
import cv2
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
    # print(feat_data.dtypes)

    unique_ids = feat_data['id'].unique()

    format_results = [] 
    # 3. additional info for each hid.
    additional_info = []

    hid_result_tuples_dict = dict()
    for hid in unique_ids:
        # 1. get intervals
        hid_intervals = to_intervals(feat_data.loc[feat_data['id'] == hid]['fid'].tolist())

        # key: other_hid, value: (distance, hid_frame, other_hid_frame)
        track_pair_distance_dict = dict()

        for [_interval_start, _interval_end] in hid_intervals:
            _start_feature = feat_data.loc[(feat_data['fid'] == _interval_start) \
                & (feat_data['id'] == hid)].iloc[0]['feature'][0].numpy()

            tracks_in_current_frame = set(feat_data.loc[feat_data['fid'] == \
                _interval_start]['id'].tolist())
            # use interval_start 
            appeared_track_table = feat_data.loc[(feat_data['fid'] < _interval_start) \
                & (~feat_data['id'].isin(tracks_in_current_frame))]

            # get the last row of each track.
            appeared_track_table = appeared_track_table.loc[\
                appeared_track_table.groupby('id')['fid'].idxmax()]

            # loop other tracks
            for idx, track_row in appeared_track_table.iterrows():
                _feature = track_row['feature'][0].numpy()
                _distance = pairwise_distances([_start_feature], [_feature])[0][0]
                _key = track_row['id']
                if _key not in track_pair_distance_dict or \
                        track_pair_distance_dict[_key][0] > _distance:
                    track_pair_distance_dict[_key] = (_distance, _interval_start, track_row['fid'])
            
            # compute end_feature
            _end_feature = feat_data.loc[(feat_data['fid'] == _interval_end) \
                & (feat_data['id'] == hid)].iloc[0]['feature'][0].numpy()
            
            tracks_in_current_frame = set(feat_data.loc[feat_data['fid'] == \
                _interval_end]['id'].tolist())

            future_track_table = feat_data.loc[(feat_data['fid'] > _interval_end) \
                & (~feat_data['id'].isin(tracks_in_current_frame))]
            # get the first row of each track
            future_track_table = future_track_table.loc[\
                future_track_table.groupby('id')['fid'].idxmin()]
            
            for idx, track_row in future_track_table.iterrows():
                _feature = track_row['feature'][0].numpy()
                _distance = pairwise_distances([_end_feature], [_feature])[0][0]
                _key = track_row['id']
                if _key not in track_pair_distance_dict or \
                        track_pair_distance_dict[_key][0] > _distance:
                    track_pair_distance_dict[_key] = (_distance, _interval_end, track_row['fid'])

        id_distance_tuples = []
        for _k, _v in track_pair_distance_dict.items():
            id_distance_tuples.append([_k, _v[0], _v[1], _v[2]])
        id_distance_tuples.sort(key= lambda x: x[1])

        # print(id_distance_tuples[:10])

        # o ids.
        oids = hid_oids_dict[hid]

        oids_info = []
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
                oids_info.append('=>oid:{:.0f}, hids: {}'.format(oid, ';'.join(dist_format)))
            else:
                oids_info.append('  oid:{:.0f}, hids: {}'.format(oid, ';'.join(dist_format)))
        additional_info.append(' | '.join(oids_info if len(oids_info) > 0 else ['NO MATCHING OID']))

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
    return additional_info, format_results, hid_result_tuples_dict, feat_data

    # print(first_fid_rows)
    # print(last_fid_rows)

def format_results_for_output(additional_info, format_results, hid_result_tuples_dict, nn):
    format_output = []
    for info, row in zip(additional_info, format_results):
        format_output.append(info)
        format_output.append(''.join([
            '{:<20s}'.format(x) for x in row[:nn+1]
        ]))
    return format_output


def produce_images_for_tracks(additional_info, format_results, hid_result_tuples_dict, \
        nn, store_folder, feat_data, frame_path_template):
    if not os.path.isdir(store_folder):
        os.makedirs(store_folder)
    
    image_template = '{folder}/{pos}-{hid}-{fid}-{dist}.jpg'

    def __clamp(x, minimum, maximum):
        return max(minimum, min(x, maximum))

    def __crop_and_save_img_from_row(hid_row, pos, hid, distance):
        # write hid image.
        image = cv2.imread(frame_path_template.format(hid_row['fid']))
        [h,w,c] = image.shape
        hid_bbox = [
            __clamp(int(hid_row['left']), 0, w), __clamp(int(hid_row['top']), 0, h), 
            __clamp(int(hid_row['right']), 0, w), __clamp(int(hid_row['bottom']), 0, h)
        ]
        raw_bbox = [
            hid_row['left'], hid_row['top'], hid_row['right'], hid_row['bottom']
        ]
        # print('reading file from ', frame_path_template.format(hid_row['fid']))
        if hid_bbox[2] == hid_bbox[0]:
            hid_bbox[2] += 1
        if hid_bbox[3] == hid_bbox[1]:
            hid_bbox[3] += 1
        # print('bbox', hid_bbox, raw_bbox, w, h)

        image = image[hid_bbox[1]:hid_bbox[3], hid_bbox[0]:hid_bbox[2]]
        # print('image shape', hid, image.shape)
        cv2.imwrite(image_template.format(
            folder=hid_folder, pos=pos, hid=hid, fid=hid_row['fid'], dist=distance
        ), image)

    for hid, results in hid_result_tuples_dict.items():
        hid_folder = '{}/{}'.format(store_folder, hid)
        if not os.path.isdir(hid_folder):
            os.makedirs(hid_folder)

        for pos, (other_hid, distance, hid_fid, other_hid_fid) in enumerate(results[:nn]):
            hid_row = feat_data.loc[(feat_data['id'] == hid)\
                 & (feat_data['fid'] == hid_fid)].iloc[0]
            other_hid_row = feat_data.loc[(feat_data['id'] == other_hid)\
                 & (feat_data['fid'] == other_hid_fid)].iloc[0]

            __crop_and_save_img_from_row(hid_row, pos, hid, distance)
            __crop_and_save_img_from_row(other_hid_row, pos, other_hid, distance)
        # import sys
        # sys.exit(0)

def simple_test(dataset, method, reid_network):
    gt_template = '../storage/dataset/MOT17/train/{}/gt/gt.txt'
    method_result_template = '../storage/results/mot17/{}/faster_rcnn-{}-person.txt'
    feat_template = '../storage/results/mot17/{}/feats-raw/faster_rcnn-{}-person-feat-{}.pkl'
    # feat_template = '../storage/results/mot17/{}/faster_rcnn-{}-person-feat.pkl'
    result_path = '../storage/results/mot17/{}/reid-feat-person/{}-{}-local.txt'

    frame_path_template = '../storage/dataset/MOT17/train/'+dataset+'/img1/{:06d}.jpg'
    image_result_path = '../storage/results/mot17/{}/reid-feat-person-images/{}-{}-local/'

    print("processing method [{}] with reid model {}".format(method, reid_network))

    a, b, c, d = produce_track_distance(
                feat_template.format(dataset, method , reid_network), 
                gt_template.format(dataset),
                method_result_template.format(dataset, method, reid_network))
    
    # outputs = format_results_for_output(a, b, c, 10)

    # # output.
    # output_path = result_path.format(dataset, method, reid_network)
    # parent_dir = os.path.dirname(output_path)
    # if not os.path.isdir(parent_dir):
    #     os.makedirs(parent_dir)
    # with open(output_path, 'w') as f:
    #     f.write('\n'.join(outputs))
    
    produce_images_for_tracks(a, b, c, 10, \
        image_result_path.format(dataset, method, reid_network), \
            d, frame_path_template)

def test_dataset(dataset):
    print('processing dataset', dataset)
    methods = ['sort', 'deepsort', 'tracktor']
    reid_models = ['osnet_x1_0', 'resnet50_fc512']

    for method in methods:
        for model in reid_models:
            simple_test(dataset, method, model)

if __name__ == '__main__':
    simple_test('MOT17-11-DPM', 'deepsort', 'osnet_x1_0')
    # test_dataset('MOT17-11-DPM')
    # test_dataset('MOT17-09-DPM')
    # test_dataset('MOT17-13-DPM')
    