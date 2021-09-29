"""
(filter short/cropped tracks)
(not eliminate bimodels)
We can attempt all the pairs in the order of avg. distance, 
until get all the matched pairs.
attemts_match() plots curves: # pairs attempt - # matched pairs
samples_distribution() shows the distribution of distances after k number of samples

This script based on:
0. run tracking methods.
1. filter_track_results.py
2. extract_reid_features_w_torch_reid.py
3. test_mot_person_reid_features_pairwise_average.py
4. write match file manually (temporal)
"""

import os
import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random

import scripts.mot.reid.test_mot_person_reid_features_pairwise_average as produce_track_distance

def attemts_match(dataset, method, reid_network, reid_model_pth_name='pretrained'):
    dir_path = '../storage/results/mot17/{}/reid-feat-person-filtered/'.format(dataset)
    dis_path = (dir_path + '{}-{}-dis-{}.txt').format(method, reid_network, reid_model_pth_name)
    match_path = (dir_path + '{}-{}-match-{}.txt').format(method, reid_network, reid_model_pth_name)
    plot_path = (dir_path + '{}-{}-attempts-{}').format(method, reid_network, reid_model_pth_name)
    num_match, match_pairs, avg_dict = 0, [], {}
    with open(match_path, 'r') as f:
        for line in f:
            if not line.strip() or line.strip()[0] == '#':
                continue
            hid1, hid2 = line.strip().split('-')[0], line.strip().split('-')[1]
            match_pairs.append('%s-%s' % (hid1, hid2) if int(hid1) <= int(hid2) else '%s-%s' % (hid2, hid1))
            num_match += 1
    with open(dis_path, 'r') as f:
        for line in f:
            if line.strip()[0] == '#':
                continue
            tmp = line.strip().split(';')
            hid1, hid2 = tmp[0].split('-')[0], tmp[0].split('-')[1]
            pairs = '%s-%s' % (hid1, hid2) if int(hid1) <= int(hid2) else '%s-%s' % (hid2, hid1)
            _avg, _median = float(tmp[3]), float(tmp[4])
            if pairs not in avg_dict:
                avg_dict[pairs] = _avg
    _x, _y = np.arange(1, num_match + 1, 1), []
    attempts = 0
    for pairs, _avg in sorted(avg_dict.items(), key=lambda x: x[1]):
        attempts += 1
        if pairs in match_pairs:
            _y.append(attempts)
    plt.figure()
    plt.plot(_x, _y, '-o')
    plt.xticks(_x)
    plt.xlabel('Matched Pairs Found')
    plt.ylabel('Pairs Attempted')
    plt.title(dataset)
    plt.savefig(plot_path)

def samples_distribution(dataset, method, reid_network, select_method='avg', 
                         reid_model_pth_name='pretrained', nn=10, skip=10):
    def plot_samples_distribution(_distances, _hid, _other_hid, file_path):
        dist_all, dist_sel = list(copy.deepcopy(_distances)), []
        parent_dir = os.path.dirname(file_path)
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)
        for sam in np.arange(skip, min(5000, len(dist_all)) + 1, skip):
            for _ in range(skip):
                dis = random.choice(dist_all)
                dist_all.remove(dis)
                dist_sel.append(dis)
            snsplot = sns.displot(dist_sel)
            fig = snsplot.fig 
            ax = snsplot.ax
            _min = np.min(dist_sel)
            _max = np.max(dist_sel)
            _average = np.average(dist_sel)
            _median = np.median(dist_sel)
            _std = np.std(dist_sel)
            ax.axvline(_average, color='red', linestyle='--', alpha=0.7)
            ax.axvspan(_average - _std, _average + _std, facecolor='green', alpha=0.2)
            ax.text(.6, .9, \
                'min:%.2f;\nmax:%.2f;\navg:%.2f;\nmedian:%.2f;\nstd:%.2f;\navg-std:%.2f;\navg+std:%.2f'
                % (_min, _max, _average, _median, _std, _average-_std, _average+_std), \
                verticalalignment='top', horizontalalignment='left', transform = ax.transAxes, fontsize=15)
            plt.title('HId:%s & HId:%s; # samples: %d' % (_hid, _other_hid, sam))
            plt.tight_layout()
            if file_path is None:
                plt.show()
            else:
                fig.savefig(file_path % (_hid, _other_hid, sam))
            plt.close(fig)       

    focused_hids = focused_hids_dataset[dataset]
    feat_template = '../storage/results/mot17/{}/feats-raw-filtered/faster_rcnn-{}-person-feat-{}-{}.pkl'
    gt_template = '../storage/dataset/MOT17/train/{}/gt/gt.txt'
    method_result_template = '../storage/results/mot17/{}/filtered-tracked/faster_rcnn-{}-person.txt'
    samples_distribution_template = '../storage/results/mot17/{}/reid-samp-dist-person-filtered/{}-{}-{}-{}-{}/%d-%d-%d'
    _, _, hid_result_tuples_dict, _, _ = produce_track_distance.produce_track_distance(
        feat_template.format(dataset, method, reid_network, reid_model_pth_name), 
        gt_template.format(dataset),
        method_result_template.format(dataset, method), 
        select_method)
    for hid, results in hid_result_tuples_dict.items():
        for _, (other_hid, _, _, _, distances) in enumerate(results[:nn]):
            if (int(hid), int(other_hid)) in focused_hids:
                plot_samples_distribution(distances.flatten(), int(hid), int(other_hid), 
                    samples_distribution_template.format(dataset, hid, other_hid, method, reid_network, reid_model_pth_name))
                print(hid, other_hid, len(distances.flatten()))

focused_hids_dataset = {
    'MOT17-04-DPM': [(54, 44)],
    'MOT17-09-DPM': [(4, 16)],
    'MOT17-11-DPM': [(3, 150), (28, 144), (50, 86), (71, 93)],
}

if __name__ == '__main__':
    for did in ['09']:  # '04', '09', '10', '11'
        attemts_match('MOT17-%s-DPM' % did, 'tracktor', 'osnet_x1_0', 'mot3')
    for did in ['09']:  # '04', '09', '10', '11'
        samples_distribution('MOT17-%s-DPM' % did, 'tracktor', 'osnet_x1_0', reid_model_pth_name='mot3')
