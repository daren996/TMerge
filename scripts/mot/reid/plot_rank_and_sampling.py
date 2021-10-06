"""
(filter short/cropped tracks)
(not eliminate bimodels)
We can attempt all the pairs in the order of avg. distance, 
until get all the matched pairs.
attemts_match() plots ranking curves: # pairs attempt - # matched pairs
samples_distribution() shows the distribution of distances after k number of samples

This script based on:
0. run tracking methods.
1. filter_track_results.py
2. extract_reid_features_w_torch_reid.py
3. test_mot_person_reid_features_pairwise_average.py
4. write match file manually (temporal)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

def attemts_match(dataset, method, reid_network, reid_model_pth_name='pretrained', data_arg=''):
    # data_arg: '', '(noexc)', '(noexc;bgs)'
    dir_path = '../storage/results/mot17{}/{}/reid-feat-person-filtered/'.format(data_arg, dataset)
    dis_path = (dir_path + '{}-{}-dis-{}.txt').format(method, reid_network, reid_model_pth_name)
    match_path = (dir_path + '{}-{}-match-{}.txt').format(method, reid_network, reid_model_pth_name)
    plot_path = (dir_path + '{}-{}-attempts-{}').format(method, reid_network, reid_model_pth_name)
    dis_all_path = (dir_path + '{}-{}-allavgdis-{}').format(method, reid_network, reid_model_pth_name)
    num_match, match_pairs, avg_dict = 0, [], {}
    with open(match_path, 'r') as f:
        for line in f:  # HId1-HId2
            if not line.strip() or line.strip()[0] == '#':
                continue
            hid1, hid2 = line.strip().split('-')[0], line.strip().split('-')[1]
            match_pairs.append('%s-%s' % (hid1, hid2) if int(hid1) <= int(hid2) else '%s-%s' % (hid2, hid1))
            num_match += 1
    dis_all = []
    with open(dis_path, 'r') as f:
        for line in f:  # # HId1-HId2;min;max;avg;median;std;avg-std;avg+std
            if line.strip()[0] == '#':
                continue
            tmp = line.strip().split(';')
            hid1, hid2 = tmp[0].split('-')[0], tmp[0].split('-')[1]
            pairs = '%s-%s' % (hid1, hid2) if int(hid1) <= int(hid2) else '%s-%s' % (hid2, hid1)
            _avg, _median = float(tmp[3]), float(tmp[4])
            dis_all.append(_avg)
            if pairs not in avg_dict:
                avg_dict[pairs] = _avg
    plt.figure()
    snsplot = sns.displot(dis_all)
    ax = snsplot.ax
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(dataset)
    plt.xlabel('Avg. Distance')
    plt.tight_layout()
    txt, _y = 'matched pair:\n', []
    attempts = 0
    for pairs, _avg in sorted(avg_dict.items(), key=lambda x: x[1]):
        attempts += 1
        if pairs in match_pairs:
            _y.append(attempts)
            txt += '%s, avg_dis:%.2f, rank:%d\n' % (pairs, _avg, attempts)
    plt.text(.05, .8, txt, verticalalignment='center', horizontalalignment='left', transform=ax.transAxes)
    plt.savefig(dis_all_path)
    _x = np.arange(1, len(_y) + 1, 1)
    plt.figure()
    print(dataset, _y)
    plt.plot(_x, _y, '-o')
    plt.xticks(_x)
    plt.xlabel('Matched Pairs Found')
    plt.ylabel('Pairs Attempted')
    plt.title(dataset)
    plt.savefig(plot_path)

focused_hids_dataset = {
    'MOT17-04-DPM': [(54, 44)],
    'MOT17-09-DPM': [(4, 16)],
    'MOT17-11-DPM': [(3, 150), (28, 144), (50, 86), (71, 93)],
}

if __name__ == '__main__':
    # for did in ['04', '09', '11']:  # '04', '09', '10', '11'
    #     attemts_match('MOT17-%s-FRCNN' % did, 'tracktor', 'osnet_x1_0', 'mot3', data_arg='')
    for did in ['01', '03', '06', '07', '08', '12', '14']:
        attemts_match('MOT17-%s-FRCNN' % did, 'tracktor', 'osnet_x1_0', 'mot3', data_arg='')
