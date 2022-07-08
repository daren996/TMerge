from collections import defaultdict
import os
import sys
import shutil
from joblib.logger import PrintTime 
import pandas as pd
import numpy as np
import time
from cv2 import cv2
from scipy.sparse import data
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from torch import count_nonzero

from scripts.pathtrack.produce_track_dis import get_hid_result, get_prior_knowledge
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

def produce_track_distance(gt_file, track_file, hid_result_tuples_dict, all_track_pair_distance_dict):
    oid_results, event_df = load_and_compute_mapping(track_file, gt_file)
    mapping_pd = generate_summary(oid_results, event_df)
    oid_hids = [(x, y) for x, y in zip(mapping_pd['oid'], mapping_pd['hids_detail'])]
    hid_oids_dict = defaultdict(list)
    for oid, hids in oid_hids:
        for hid in hids:
            hid_oids_dict[hid].append(oid)
    format_results = [] 
    additional_info = []
    detailed_additional_info_dict = dict()
    for hid in hid_result_tuples_dict:
        id_distance_tuples = hid_result_tuples_dict[hid]
        track_pair_distance_dict = all_track_pair_distance_dict[hid]
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
            if len(set(oids) & set(cell_oids)) > 0:
                row_result.append('*{}:[{},{}]/{:.4f}'.format(cell[0], cell[2], cell[3], cell[1]))
            else:
                row_result.append(' {}:[{},{}]/{:.4f}'.format(cell[0], cell[2], cell[3], cell[1]))
        format_results.append(row_result)
    return additional_info, format_results, detailed_additional_info_dict

def format_results_for_output(additional_info, format_results, nn):
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

def format_results_for_dis(hid_result_tuples_dict, nn):
    num_dis = 0
    format_out_dis = []
    format_out_dis.append('# HId1-HId2;min;max;avg;median;std;avg-std;avg+std')
    for hid, results in hid_result_tuples_dict.items():
        num_dis += len(results)
        for _, (other_hid, _, _, _, _distances) in enumerate(results[:nn]):
            _min = np.min(_distances)
            _max = np.max(_distances)
            _average = np.average(_distances)
            _median = np.median(_distances)
            _std = np.std(_distances)
            format_out_dis.append('%s-%s;%.2f;%.2f;%.2f;%.2f;%.2f;%.2f;%.2f' 
            % (hid, other_hid, _min, _max, _average, _median, _std, _average-_std, _average+_std))
    return format_out_dis, num_dis

def produce_images_for_tracks(hid_result_tuples_dict, nn, store_folder, feat_data, feat_data_pre, \
        dataset, frame_tmp, dataset_pre, frame_tmp_pre, generate_raw_frames=True):
    if os.path.isdir(store_folder):
        shutil.rmtree(store_folder)
    os.makedirs(store_folder)
    image_template = '{folder}/R[{pos}]-{hid}-{dist:.1f}.jpg'
    distribution_template = '{folder}/{pos}-{hid}-distribution.jpg'

    def __clamp(x, minimum, maximum):
        return max(minimum, min(x, maximum))

    def __crop_image(hid_row, is_pre=False):
        # write hid image.
        if is_pre:
            fid = int(dataset_pre.split('_')[-2]) + hid_row['fid']
            image = cv2.imread(frame_tmp_pre.format(fid))
        else:
            fid = int(dataset.split('_')[-2]) + hid_row['fid']
            image = cv2.imread(frame_tmp.format(fid))
        [h,w,c] = image.shape
        hid_bbox = [
            __clamp(int(hid_row['left']), 0, w), __clamp(int(hid_row['top']), 0, h), 
            __clamp(int(hid_row['right']), 0, w), __clamp(int(hid_row['bottom']), 0, h)
        ]
        if hid_bbox[2] == hid_bbox[0]:
            hid_bbox[2] += 1
        if hid_bbox[3] == hid_bbox[1]:
            hid_bbox[3] += 1
        image = image[hid_bbox[1]:hid_bbox[3], hid_bbox[0]:hid_bbox[2]]
        return image

    def __crop_and_save_img_from_row(hid_row, pos, hid, dist, is_pre=False):
        # print('image shape', hid, image.shape)
        image = __crop_image(hid_row, is_pre)
        cv2.imwrite(image_template.format(
            folder=hid_folder, pos=pos, hid=hid, dist=dist), image)

    for dataset_hid, results in hid_result_tuples_dict.items():
        hid = int(dataset_hid.split('_')[-1])

        # other_hid_dist_dict = dict()
        # for [ohid, *rest] in results:
        #     other_hid_dist_dict[ohid] = rest

        hid_folder = '{}/{}'.format(store_folder, hid)

        if not os.path.isdir(hid_folder):
            os.makedirs(hid_folder)
        # store all bounding boxes of hid
        if generate_raw_frames:
            hid_raw_frames_folder = hid_folder + '/frames/'
            if not os.path.isdir(hid_raw_frames_folder):
                os.makedirs(hid_raw_frames_folder)
            for _, _hid_row in feat_data.loc[feat_data['id'] == hid].iterrows():
                _image = __crop_image(_hid_row)
                cv2.imwrite('{}/{}.jpg'.format(hid_raw_frames_folder, _hid_row['fid']), _image)

        for pos, (dataset_other_hid, dist, hid_fid, other_hid_fid, _distances) in enumerate(results[:nn]):
            other_hid = int(dataset_other_hid.split('_')[-1])
            is_pre = False
            if '_'.join(dataset_other_hid.split('_')[:-1]) == dataset_pre:
                is_pre = True
            hid_row = feat_data.loc[(feat_data['id'] == hid)\
                 & (feat_data['fid'] == hid_fid)].iloc[0]
            if is_pre:
                other_hid_row = feat_data_pre.loc[(feat_data_pre['id'] == other_hid) & \
                    (feat_data_pre['fid'] == other_hid_fid)].iloc[0]
            else:
                other_hid_row = feat_data.loc[(feat_data['id'] == other_hid) & \
                    (feat_data['fid'] == other_hid_fid)].iloc[0]

            # __crop_and_save_img_from_row(hid_row, pos, hid, dist, is_pre=is_pre)
            if is_pre:
                __crop_and_save_img_from_row(other_hid_row, pos, 'PRE%d' % other_hid, dist, is_pre=is_pre)
            else:
                __crop_and_save_img_from_row(other_hid_row, pos, other_hid, dist, is_pre=is_pre)
            
            # save distance distribution
            # _template = distribution_template_selected if _is_selected else distribution_template
            # plot_distribution(_distances, _template.format(
            #     folder=hid_folder, pos=pos, hid=other_hid, 
            #     selected=_is_selected), hid, other_hid)

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

def simple_test(dataset_pair, method, reid_network, select_method, reid_model_pth_name='pretrained', train_test='selected'):
    dataset_pre, dataset = dataset_pair[0], dataset_pair[1]
    loss_str = '_triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)
    frame_tmp = '../storage/dataset/PathTrack/{train_test}/'.format(train_test=train_test)+dataset+'/img1/{:d}.png'
    feat_path_tmp = '../storage/results/pathtrack/{}/feats-raw-filtered%s/faster_rcnn-{}-person-feat-{}-{}.pkl' % loss_str
    feat_path = feat_path_tmp.format(dataset, method, reid_network, reid_model_pth_name)
    image_result_tmp = '../storage/results/pathtrack/{}/feats-raw-filtered%s/reid-feat-person-images-filtered/{}-{}-pairwise-{}-{}/' % loss_str
    image_result_path = image_result_tmp.format(dataset, method, reid_network, select_method, reid_model_pth_name)
    dis_tmp = '../storage/results/pathtrack/{}/feats-raw-filtered%s/reid-feat-person-filtered/{}-{}-dis-{}.txt' % loss_str
    output_dis_path = dis_tmp.format(dataset, method, reid_network, reid_model_pth_name)
    if dataset_pre:
        frame_tmp_pre = '../storage/dataset/PathTrack/{train_test}/'.format(train_test=train_test)+dataset_pre+'/img1/{:d}.png'
        feat_path_pre = feat_path_tmp.format(dataset_pre, method, reid_network, reid_model_pth_name)
    else:  
        frame_tmp_pre, feat_path_pre = '', ''
    print("dataset: {} and {}".format(dataset_pre, dataset))
    print("\tprocessing method [{}] with reid model {}".format(method, reid_network))
    time1 = time.time()
    hid_result_tuples_dict, _, feat_data, feat_data_pre = get_hid_result(dataset, feat_path, dataset_pre, feat_path_pre, select_method)
    print("# tracks", len(list(hid_result_tuples_dict.keys())))
    print("time:", time.time() - time1)
    # distance file
    out_dis, num_dis = format_results_for_dis(hid_result_tuples_dict, 10)
    print("\tnum_track_pairs:", num_dis)
    # return
    parent_dir = os.path.dirname(output_dis_path)
    if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)
    with open(output_dis_path, 'w') as f:
        f.write('\n'.join(out_dis))
    # producing images
    print("\tproducing images")
    produce_images_for_tracks(hid_result_tuples_dict, 10, image_result_path, feat_data, feat_data_pre,
                              dataset, frame_tmp, dataset_pre, frame_tmp_pre,
                              generate_raw_frames=True)

if __name__ == '__main__':
    dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000', '0Xtp77A4zF4_3000_5000', '0Xtp77A4zF4_4000_6000', ]
    # dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000', '0Xtp77A4zF4_3000_5000', '0Xtp77A4zF4_4000_6000', '0Xtp77A4zF4_5000_7000', '0Xtp77A4zF4_6000_8000', '0Xtp77A4zF4_7000_9000', '0Xtp77A4zF4_8000_10000', '0Xtp77A4zF4_9000_11000', '0Xtp77A4zF4_10000_12000', '0Xtp77A4zF4_11000_13000', '0Xtp77A4zF4_12000_14000', '0Xtp77A4zF4_13000_15000', '0Xtp77A4zF4_14000_16000', '0Xtp77A4zF4_15000_17000', '0Xtp77A4zF4_16000_18000', '0Xtp77A4zF4_17000_19000', '0Xtp77A4zF4_18000_20000', '0Xtp77A4zF4_19000_21000', '0Xtp77A4zF4_20000_21635',]
    # dataset_names = ['4LFwzgwRyrY_0_2000', '4LFwzgwRyrY_1000_3000', '4LFwzgwRyrY_2000_4000', '4LFwzgwRyrY_3000_5000', '4LFwzgwRyrY_4000_6000', '4LFwzgwRyrY_5000_7000', '4LFwzgwRyrY_6000_8000', '4LFwzgwRyrY_7000_9000', '4LFwzgwRyrY_8000_10000', '4LFwzgwRyrY_9000_11000', '4LFwzgwRyrY_10000_12000', '4LFwzgwRyrY_11000_12000', ]
    # dataset_names = ['5fhJSO5al8o_0_2000', '5fhJSO5al8o_1000_3000', '5fhJSO5al8o_2000_4000', '5fhJSO5al8o_3000_5000', '5fhJSO5al8o_4000_6000', '5fhJSO5al8o_5000_7000', '5fhJSO5al8o_6000_8000', '5fhJSO5al8o_7000_9000', '5fhJSO5al8o_8000_10000', '5fhJSO5al8o_9000_11000', '5fhJSO5al8o_10000_12000', '5fhJSO5al8o_11000_13000', '5fhJSO5al8o_12000_14000', '5fhJSO5al8o_13000_15000', '5fhJSO5al8o_14000_16000', '5fhJSO5al8o_15000_16000', ]
    # dataset_names = ['81ahuidFUWw_0_2000', '81ahuidFUWw_1000_3000', '81ahuidFUWw_2000_4000', '81ahuidFUWw_3000_5000', '81ahuidFUWw_4000_6000', '81ahuidFUWw_5000_7000', '81ahuidFUWw_6000_8000', '81ahuidFUWw_7000_9000', '81ahuidFUWw_8000_10000', '81ahuidFUWw_9000_11000', '81ahuidFUWw_10000_12000', '81ahuidFUWw_11000_13000', '81ahuidFUWw_12000_14000', '81ahuidFUWw_13000_15000', '81ahuidFUWw_14000_16000', '81ahuidFUWw_15000_17000', '81ahuidFUWw_16000_17500', ]
    # dataset_names = ['KQqFixbb-o8_0_2000', 'KQqFixbb-o8_1000_3000', 'KQqFixbb-o8_2000_4000', 'KQqFixbb-o8_3000_5000', 'KQqFixbb-o8_4000_6000', 'KQqFixbb-o8_5000_7000', 'KQqFixbb-o8_6000_8000', 'KQqFixbb-o8_7000_9000', 'KQqFixbb-o8_8000_9099']
    # dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000']

    for idx, dn in enumerate(dataset_names):
        if idx == 0:
            dn_pair = [None, dn]
        else:
            dn_pair = [dataset_names[idx - 1], dn]
        for method_sel in ['avg']:  # 'median', 'avg'
            for mrg, wt, wx in zip(['005'], ['10'], ['05']):
                simple_test(dn_pair, 'tracktor', 'osnet_x1_0', method_sel, 'mot3', train_test='selected')
