import sys
from cv2 import LMEDS
import pandas as pd
import numpy as np
import time
from scipy.sparse import data
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

def get_hid_result(dataset, feat_path, dataset_pre, feat_path_pre, select_method, l_max=1000):
    hid_result_tuples_dict = dict()  # {dataset_hid: [(other_hid, avg/med, fid1, fid2, [dis]), ...]}
    all_track_pair_distance_dict = dict()  # {dataset_hid: {other_hid: (avg/med, fid1, fid2, [dis])}}
    
    feat_data = pd.read_pickle(feat_path).astype({
            'fid': 'int32',
            'id': 'int32',  # HID, or Tracking ID
        })
    unique_ids = []
    for hid in feat_data['id'].unique():
        if min(feat_data.loc[feat_data['id'] == hid]['fid'].tolist()) <= l_max:
            unique_ids.append(hid)
    
    feat_data_pre = None
    if dataset_pre:
        feat_data_pre = pd.read_pickle(feat_path_pre).astype({
            'fid': 'int32',
            'id': 'int32',  # HID, or Tracking ID
        })
        unique_ids_pre = []
        for hid_pre in feat_data_pre['id'].unique():
            if min(feat_data_pre.loc[feat_data_pre['id'] == hid_pre]['fid'].tolist()) <= l_max:
                unique_ids_pre.append(hid_pre)
        # get their info
        id_and_feat_dict_pre = dict()  # {other_hid_pre: (feats, fids)}
        for other_hid_pre in unique_ids_pre:
            this_hid_table = feat_data_pre.loc[feat_data_pre['id'] == other_hid_pre]
            if len(this_hid_table) <= 0:
                continue
            _feats = [x[0].numpy() for x in this_hid_table['feature']]
            _fids = [x for x in this_hid_table['fid']]
            id_and_feat_dict_pre[other_hid_pre] = (_feats, _fids)

    for hid in unique_ids:
        dataset_hid = '%s_%d' % (dataset, hid)
        track_pair_distance_dict = dict()  # {dataset_other_hid: (avg/med, fid1, fid2, [dis])}
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
            if other_hid not in unique_ids:
                continue
            this_hid_table = feat_data.loc[feat_data['id'] == other_hid]
            if len(this_hid_table) <= 0:
                continue
            _feats = [x[0].numpy() for x in this_hid_table['feature']]
            _fids = [x for x in this_hid_table['fid']]
            id_and_feat_dict[other_hid] = (_feats, _fids)
        for other_hid in future_ids:
            if other_hid not in unique_ids:
                continue
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
            dataset_other_hid = '%s_%d' % (dataset, _key)
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
            if dataset_other_hid not in track_pair_distance_dict or \
                    track_pair_distance_dict[dataset_other_hid][0] > _select_distance:
                track_pair_distance_dict[dataset_other_hid] = (_select_distance, _hid_fid, _other_hid_fid, _distances)
        
        if dataset_pre:
            for _key, (_features, _fids) in id_and_feat_dict_pre.items():
                dataset_other_hid = '%s_%d' % (dataset_pre, _key)
                _distances = pairwise_distances(current_interval_features, _features)  
                if select_method == 'avg':  # get average distance
                    _select_distance = np.average(_distances)
                elif select_method == 'median':  # get median distance
                    _select_distance = np.median(_distances)
                else:
                    print('ERROR: Expecting "avg" or "median" for select_method.')
                    sys.exit(-1)
                _row_idx, _col_idx = np.unravel_index(_distances.argmin(), _distances.shape)
                _hid_fid = current_feat_data.iloc[_row_idx]['fid']
                _other_hid_fid = _fids[_col_idx]
                if dataset_other_hid not in track_pair_distance_dict or \
                        track_pair_distance_dict[dataset_other_hid][0] > _select_distance:
                    track_pair_distance_dict[dataset_other_hid] = (_select_distance, _hid_fid, _other_hid_fid, _distances)
        
        all_track_pair_distance_dict[dataset_hid] = track_pair_distance_dict
        id_distance_tuples = []
        for _k, _v in track_pair_distance_dict.items():
            id_distance_tuples.append([_k, _v[0], _v[1], _v[2], _v[3]])
        id_distance_tuples.sort(key= lambda x: x[1])
        hid_result_tuples_dict[dataset_hid] = id_distance_tuples

    return hid_result_tuples_dict, all_track_pair_distance_dict, feat_data, feat_data_pre

def get_prior_knowledge(dataset, feat_path, dataset_pre, feat_path_pre, l_max=1000):
    hid_temporal_info = dict()  # {dataset_hid: (s, r)}
    hid_spatial_info = dict()  # {dataset_hid: [(left, top, right, bottom), (left, top, right, bottom)]}
    feat_data = pd.read_pickle(feat_path).astype({'fid': 'int32', 'id': 'int32'})
    unique_ids = feat_data['id'].unique()
    for hid in unique_ids:
        dataset_hid = '%s_%d' % (dataset, hid)
        interval_start = min(feat_data.loc[feat_data['id'] == hid]['fid'].tolist())
        interval_end = max(feat_data.loc[feat_data['id'] == hid]['fid'].tolist())
        hid_temporal_info[dataset_hid] = (interval_start, interval_end)
        feat_start = feat_data.loc[(feat_data['id'] == hid) & (feat_data['fid'] == interval_start)]
        feat_end = feat_data.loc[(feat_data['id'] == hid) & (feat_data['fid'] == interval_end)]
        hid_spatial_info[dataset_hid] = [
            (feat_start['left'].tolist()[0], feat_start['top'].tolist()[0], 
            feat_start['right'].tolist()[0], feat_start['bottom'].tolist()[0]), 
            (feat_end['left'].tolist()[0], feat_end['top'].tolist()[0], 
            feat_end['right'].tolist()[0], feat_end['bottom'].tolist()[0]) ]
    if dataset_pre:
        feat_data_pre = pd.read_pickle(feat_path_pre).astype({'fid': 'int32', 'id': 'int32'})
        unique_ids_pre = feat_data_pre['id'].unique()
        for hid in unique_ids_pre:
            dataset_hid = '%s_%d' % (dataset_pre, hid)
            interval_start = min(feat_data_pre.loc[feat_data_pre['id'] == hid]['fid'].tolist())
            interval_end = max(feat_data_pre.loc[feat_data_pre['id'] == hid]['fid'].tolist())
            hid_temporal_info[dataset_hid] = (interval_start - l_max, interval_end - l_max)
            feat_start = feat_data_pre.loc[(feat_data_pre['id'] == hid) & (feat_data_pre['fid'] == interval_start)]
            feat_end = feat_data_pre.loc[(feat_data_pre['id'] == hid) & (feat_data_pre['fid'] == interval_end)]
            hid_spatial_info[dataset_hid] = [
                (feat_start['left'].tolist()[0], feat_start['top'].tolist()[0], 
                feat_start['right'].tolist()[0], feat_start['bottom'].tolist()[0]), 
                (feat_end['left'].tolist()[0], feat_end['top'].tolist()[0], 
                feat_end['right'].tolist()[0], feat_end['bottom'].tolist()[0]) ]
    return hid_temporal_info, hid_spatial_info
