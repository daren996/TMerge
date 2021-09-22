from collections import defaultdict
import os
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, NearestCentroid
from sklearn.metrics import pairwise_distances
from tools.mot.mot_mapping import generate_summary, load_and_compute_mapping

def pick_first_func(group):
    return group['feature'].iloc[0]

def pick_last_func(group):
    return group['feature'].iloc[-1]

def pick_mid_func(group):
    total_count = group['feature'].count()
    return group['feature'].iloc[total_count//2]

def extract_ids_and_feats_pick_one(data, pick_one_func):
    # print(data.columns)
    grouped = data.groupby(by='id')

    ids = [] # list of ids
    features = [] # select features
    for name, group in grouped:
        ids.append(name)
        # feature is one dim list of tensor: [tensor]
        features.append(pick_one_func(group)[0].numpy())
    return ids, features

def extract_ids_and_feats_centroid(data):
    # id, label
    labels = np.array(data['id'].tolist())
    # 2d list to numpy ndarray
    feats = np.array(data['feature'].apply(lambda x : x[0].tolist()).tolist())

    clf = NearestCentroid()
    clf.fit(feats, labels)
    # print(clf.centroids_)
    # print(len(clf.classes_), len(clf.centroids_))
    return clf.classes_.tolist(), clf.centroids_.tolist()

def produce_track_distance(pkl_file_path, gt_file, track_file, 
        extract_ids_and_feats_func=extract_ids_and_feats_centroid,
        nn=15):

    oid_results, event_df = load_and_compute_mapping(track_file, gt_file)
    mapping_pd = generate_summary(oid_results, event_df)
    # print(mapping_pd.columns)
    # store mapping hid -> oids
    oid_hids = [(x, y) for x, y in zip(mapping_pd['oid'], mapping_pd['hids_detail'])]
    hid_oids_dict = defaultdict(list)
    for oid, hids in oid_hids:
        for hid in hids:
            hid_oids_dict[hid].append(oid)

    data = pd.read_pickle(pkl_file_path)
    
    # ids, features = extract_ids_and_feats_pick_one(data, pick_first_func)
    ids, features = extract_ids_and_feats_func(data)

    nbrs = NearestNeighbors(n_neighbors=nn+1, algorithm='ball_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)


    # 1. replace indices with hids;
    hid_results = [[ids[cell] for cell in row] for row in indices]
    # 2. format
    format_results = [] 
    # 3. additional info for each hid.
    additional_info = []
    for row in hid_results:
        oids = hid_oids_dict[row[0]]

        oids_info = []
        for oid in oids:
            oid_row = mapping_pd.loc[mapping_pd['oid'] == oid]
            # compute distance between hid and row[0]
            hids_row = oid_row['hids_detail'].iloc[0]
            dists_row = []
            for cell in hids_row:
                dist = pairwise_distances([features[ids.index(row[0])]], 
                    [features[ids.index(cell)]])
                dists_row.append(dist[0])
            # print(dists_row)
            dist_format = ['{:.0f}:{:.4f}'.format(x,y[0]) for x, y in zip(hids_row, dists_row)]
            oids_info.append('oid:{:.0f}, hids: {}'.format(oid, ';'.join(dist_format)))
        additional_info.append(' | '.join(oids_info if len(oids_info) > 0 else ['NO MATCHING OID']))

        row_result = [str(row[0])]
        for cell in row[1:]:
            cell_oids = hid_oids_dict[cell]
            if len(set(oids) & set(cell_oids)) > 0:
                row_result.append('*{}'.format(cell))
            else:
                row_result.append(' {}'.format(cell))
        format_results.append(row_result)
    return additional_info, format_results, distances

def format_results_for_output(additional_info, format_results, distances):
    format_output = []
    for info, row, distance in zip(additional_info, format_results, distances):
        format_output.append(info)
        format_output.append(''.join(['{:<10s}'.format('{:>4s}:{:.4f}'.format(x, y)) 
            for x, y in zip(row, distance)]))
    
    return format_output

def test_dataset(dataset):
    print('dealing with dataset:', dataset)
    gt_template = '../storage/dataset/MOT17/train/{}/gt/gt.txt'
    method_result_template = '../storage/results/mot17/{}/faster_rcnn-{}-person.txt'
    feat_template = '../storage/results/mot17/{}/faster_rcnn-{}-person-feat.pkl'
    result_path = '../storage/results/mot17/{}/reid-feat/{}-{}.txt'

    methods = ['sort', 'deepsort', 'tracktor']
    strategies = [
        ('mid', lambda x: extract_ids_and_feats_pick_one(x, pick_mid_func)),
        ('first', lambda x: extract_ids_and_feats_pick_one(x, pick_first_func)),
        ('last', lambda x: extract_ids_and_feats_pick_one(x, pick_last_func)),
        ('centroid', extract_ids_and_feats_centroid)
    ]
    for method in methods:
        for strategy_name, strategy in strategies:
            print('processing method: [{}] with strategy: {}'.format(method, strategy_name))
            results = produce_track_distance(
                feat_template.format(dataset, method), 
                gt_template.format(dataset),
                method_result_template.format(dataset, method),
                strategy
            )
            output_content = format_results_for_output(*results)
            output_path = result_path.format(dataset, method, strategy_name)
            parent_dir = os.path.dirname(output_path)
            if not os.path.isdir(parent_dir):
                os.makedirs(parent_dir)
            with open(output_path, 'w') as f:
                f.write('\n'.join(output_content))


if __name__ == '__main__':    
    # fp='../storage/results/mot17/MOT17-11-DPM/faster_rcnn-deepsort-person-feat.pkl'
    # gt_ = '../storage/dataset/MOT17/train/MOT17-11-DPM/gt/gt.txt'
    # method_ = '../storage/results/mot17/MOT17-11-DPM/faster_rcnn-deepsort-person.txt'
    # results = produce_track_distance(fp, gt_, method_, 
    #     extract_ids_and_feats_centroid
    #     # lambda x: extract_ids_and_feats_pick_one(x, pick_mid_func)
    #     # lambda x: extract_ids_and_feats_pick_one(x, pick_first_func)
    #     # lambda x: extract_ids_and_feats_pick_one(x, pick_last_func)
    # )
    # print('\n'.join(format_results_for_output(*results)))

    # test_dataset('MOT17-04-DPM')
    # test_dataset('MOT17-09-DPM')
    test_dataset('MOT17-11-DPM')
    # test_dataset('MOT17-13-DPM')

