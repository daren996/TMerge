import sys
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import pairwise_distances

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

def get_hid_result(pkl_file_path, select_method):
    hid_result_tuples_dict = dict()  # {hid: [(other_hid, avg/med, fid1, fid2, [dis]), ...]}
    all_track_pair_distance_dict = dict()  # {hid: {other_hid: (avg/med, fid1, fid2, [dis])}}
    feat_data = pd.read_pickle(pkl_file_path).astype({
            'fid': 'int32',
            'id': 'int32',  # HID, or Tracking ID
        })
    unique_ids = feat_data['id'].unique()
    for hid in unique_ids:
        track_pair_distance_dict = dict()  # {other_hid: (avg/med, fid1, fid2, [dis])}
        # get interval start & end
        _interval_start = min(feat_data.loc[feat_data['id'] == hid]['fid'].tolist())
        _interval_end = max(feat_data.loc[feat_data['id'] == hid]['fid'].tolist())
        # get the centroid of this track
        current_feat_data = feat_data.loc[
            (feat_data['fid'] >= _interval_start) \
                & (feat_data['fid'] <= _interval_end) \
                & (feat_data['id'] == hid)
            ]
        current_interval_features = [x[0].numpy() for x in current_feat_data['feature']]
        # get track(s) before start frame
        tracks_in_start_frame = set(feat_data.loc[feat_data['fid'] == \
            _interval_start]['id'].tolist())  # hid(s) in start frame
        appeared_feat_data = feat_data.loc[(feat_data['fid'] < _interval_start) \
            & (~feat_data['id'].isin(tracks_in_start_frame))]
        appeared_ids = appeared_feat_data['id'].unique()
        # get track(s) after end frame
        tracks_in_end_frame = set(feat_data.loc[feat_data['fid'] == \
            _interval_end]['id'].tolist())  # hid(s) in start frame
        future_feat_data = feat_data.loc[(feat_data['fid'] > _interval_end) \
            & (~feat_data['id'].isin(tracks_in_end_frame))]
        future_ids = future_feat_data['id'].unique()
        # get their info
        id_and_feat_dict = dict()  # {hid: (feats, fids)}
        for other_hid in appeared_ids:
            this_hid_table = feat_data.loc[feat_data['id'] == other_hid]
            if len(this_hid_table) <= 0:
                continue
            _feats = [x[0].numpy() for x in this_hid_table['feature']]
            _fids = [x for x in this_hid_table['fid']]
            id_and_feat_dict[other_hid] = (_feats, _fids)
        for other_hid in future_ids:
            this_hid_table = feat_data.loc[feat_data['id'] == other_hid]
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
            _distances = pairwise_distances(current_interval_features, _features)  # !!! error? same length distances vector
            if select_method == 'avg':  # get average distance
                _select_distance = np.average(_distances)
            elif select_method == 'median':  # get median distance
                _select_distance = np.median(_distances)
            else:
                print('ERROR: Expecting "avg" or "median" for select_method.')
                sys.exit(-1)
            # _idx = np.argsort(_distances, None)[_distances.size//2]
            # _row_idx, _col_idx = np.unravel_index(_idx, _distances.shape)
            _row_idx, _col_idx = np.unravel_index(_distances.argmin(), _distances.shape)
            _hid_fid = current_feat_data.iloc[_row_idx]['fid']
            _other_hid_fid = _fids[_col_idx]
            if _key not in track_pair_distance_dict or \
                    track_pair_distance_dict[_key][0] > _select_distance:
                track_pair_distance_dict[_key] = (_select_distance, _hid_fid, _other_hid_fid, _distances)
            # draw distance distribution.
        all_track_pair_distance_dict[hid] = track_pair_distance_dict
        id_distance_tuples = []
        for _k, _v in track_pair_distance_dict.items():
            id_distance_tuples.append([_k, _v[0], _v[1], _v[2], _v[3]])
        id_distance_tuples.sort(key= lambda x: x[1])
        hid_result_tuples_dict[hid] = id_distance_tuples
    return hid_result_tuples_dict, all_track_pair_distance_dict, feat_data

def get_prior_knowledge(pkl_file_path):
    hid_temporal_info = dict()  # {hid: (s, r)}
    hid_spatial_info = dict()  # {hid: [(left, top, right, bottom), (left, top, right, bottom)]}
    feat_data = pd.read_pickle(pkl_file_path).astype({'fid': 'int32', 'id': 'int32'})
    unique_ids = feat_data['id'].unique()
    for hid in unique_ids:
        interval_start = min(feat_data.loc[feat_data['id'] == hid]['fid'].tolist())
        interval_end = max(feat_data.loc[feat_data['id'] == hid]['fid'].tolist())
        hid_temporal_info[hid] = (interval_start, interval_end)
        feat_start = feat_data.loc[(feat_data['id'] == hid) & (feat_data['fid'] == interval_start)]
        feat_end = feat_data.loc[(feat_data['id'] == hid) & (feat_data['fid'] == interval_end)]
        hid_spatial_info[hid] = [
            (feat_start['left'].tolist()[0], feat_start['top'].tolist()[0],
            feat_start['right'].tolist()[0], feat_start['bottom'].tolist()[0]),
            (feat_end['left'].tolist()[0], feat_end['top'].tolist()[0],
            feat_end['right'].tolist()[0], feat_end['bottom'].tolist()[0]) ]
    return hid_temporal_info, hid_spatial_info
