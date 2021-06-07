from collections import defaultdict
import os
import pandas as pd
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

    # for each object, compute the first and last frame it appears
    first_fid_rows = feat_data.loc[feat_data.groupby('id')['fid'].idxmin()]
    last_fid_rows = feat_data.loc[feat_data.groupby('id')['fid'].idxmax()]

    unique_ids = feat_data['id'].unique()

    format_results = [] 
    # 3. additional info for each hid.
    additional_info = []

    for hid in unique_ids:
        # 1. get fid
        hid_first_row = first_fid_rows.loc[first_fid_rows['id'] == hid].iloc[0]
        hid_last_row = last_fid_rows.loc[last_fid_rows['id'] == hid].iloc[0]
        # print('first, last', hid, hid_first_row['fid'], hid_last_row['fid'])

        # get previous ids
        exipred_objs_table = last_fid_rows.loc[last_fid_rows['fid'] < hid_first_row['fid']]
        
        future_objs_table = first_fid_rows.loc[first_fid_rows['fid'] > hid_last_row['fid']]

        # compute distances
        id_distance_tuples = []

        _first_feature = hid_first_row['feature'][0].numpy()
        for idx, row in exipred_objs_table.iterrows():
            _feature = row['feature'][0].numpy()
            _distance = pairwise_distances([_first_feature], [_feature])[0][0]
            id_distance_tuples.append((row['id'], _distance, 'F'))

        _last_feature = hid_last_row['feature'][0].numpy()
        for idx, row in future_objs_table.iterrows():
            _feature = row['feature'][0].numpy()
            _distance = pairwise_distances([_last_feature], [_feature])[0][0]
            id_distance_tuples.append((row['id'], _distance, 'L'))
        id_distance_tuples.sort(key= lambda x: x[1])

        # print(id_distance_tuples[:10])
        
        hid_expired_ids = exipred_objs_table['id'].tolist()
        hid_future_ids = future_objs_table['id'].tolist()

        # print('range', hid_first_row['fid'], hid_last_row['fid'])
        # print('interval', to_intervals(feat_data.loc[feat_data['id'] == hid]['fid'].tolist()))
        # print('hid', hid_expired_ids, hid_future_ids)

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
                # print('cell', cell, cell in hid_expired_ids, cell in hid_future_ids)
                # print(cell, hid_expired_ids, hid_future_ids)
                # get distance from 
                if cell in hid_expired_ids:
                    _feature = exipred_objs_table.loc[exipred_objs_table['id'] == cell].iloc[0]
                    _feature = _feature['feature'][0].numpy()
                    _dist = pairwise_distances([_first_feature], [_feature])[0][0]
                    hid_distance_type.append('F')
                elif cell in hid_future_ids:
                    _feature = future_objs_table.loc[future_objs_table['id'] == cell].iloc[0]
                    _feature = _feature['feature'][0].numpy()
                    _dist = pairwise_distances([_last_feature], [_feature])[0][0]
                    hid_distance_type.append('L')
                else:
                    hid_distance_type.append('-')
                    _dist = -1
                dists_row.append(_dist)
            # print(hids_row, dists_row, hid_distance_type)
            dist_format = ['{:.0f}:{}/{:.4f}'.format(x,z, y) \
                for x, y, z in zip(hids_row, dists_row, hid_distance_type)]
            oids_info.append('oid:{:.0f}, hids: {}'.format(oid, ';'.join(dist_format)))
        additional_info.append(' | '.join(oids_info if len(oids_info) > 0 else ['NO MATCHING OID']))

        row_result = [str(hid)]
        for cell in id_distance_tuples:
            cell_oids = hid_oids_dict[cell[0]]
            # print('oids', oids, cell_oids)
            if len(set(oids) & set(cell_oids)) > 0:
                row_result.append('*{}:{}/{:.4f}'.format(cell[0], cell[2], cell[1]))
            else:
                row_result.append(' {}:{}/{:.4f}'.format(cell[0], cell[2], cell[1]))
        format_results.append(row_result)
        # print(oids_info)
        # print(format_results)
        # import sys
        # sys.exit(0)
    return additional_info, format_results, id_distance_tuples

    # print(first_fid_rows)
    # print(last_fid_rows)

def format_results_for_output(additional_info, format_results, id_distance_tuples, nn):
    format_output = []
    for info, row in zip(additional_info, format_results):
        format_output.append(info)
        format_output.append(''.join([
            '{:<15s}'.format(x) for x in row[:nn+1]
        ]))
    return format_output

def simple_test(dataset, method, reid_network):
    gt_template = '../storage/dataset/MOT17/train/{}/gt/gt.txt'
    method_result_template = '../storage/results/mot17/{}/faster_rcnn-{}-person.txt'
    feat_template = '../storage/results/mot17/{}/feats-raw/faster_rcnn-{}-person-feat-{}.pkl'
    # feat_template = '../storage/results/mot17/{}/faster_rcnn-{}-person-feat.pkl'
    result_path = '../storage/results/mot17/{}/reid-feat-person/{}-{}-global.txt'

    print("processing method [{}] with reid model {}".format(method, reid_network))

    a, b, c = produce_track_distance(
                feat_template.format(dataset, method , reid_network), 
                gt_template.format(dataset),
                method_result_template.format(dataset, method, reid_network))
    
    outputs = format_results_for_output(a, b, c, 10)

    # output.
    output_path = result_path.format(dataset, method, reid_network)
    parent_dir = os.path.dirname(output_path)
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    with open(output_path, 'w') as f:
        f.write('\n'.join(outputs))

def test_dataset(dataset):
    print('processing dataset', dataset)
    methods = ['sort', 'deepsort', 'tracktor']
    reid_models = ['osnet_x1_0', 'resnet50_fc512']

    for method in methods:
        for model in reid_models:
            simple_test(dataset, method, model)

if __name__ == '__main__':
    # simple_test('MOT17-11-DPM', 'deepsort', 'osnet_x1_0')
    # test_dataset('MOT17-11-DPM')
    test_dataset('MOT17-09-DPM')
    test_dataset('MOT17-13-DPM')
    