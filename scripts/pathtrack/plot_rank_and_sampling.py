"""
(filter short/cropped tracks)
(not eliminate bimodels)
Cumulative Match Curve.
We can attempt all the pairs in the order of avg. distance, 
until get all the matched pairs.
cmc() plots cmc curves: # pair rank - # matched pairs
samples_distribution() shows the distribution of distances after k number of samples

This script based on:
0. run tracking methods.
1. filter_track_results.py
2. extract_reid_features_w_torch_reid.py
3. test_mot_person_reid_features_pairwise_average.py
4. write match file manually (temporal)
"""

from PIL.Image import preinit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


name_map = {
    'softmax': 'softmax', 
    'triplet_mrg03_wt10_wx00' :'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 0.0' ,
    'triplet_mrg03_wt10_wx05' :'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 0.5' ,
    'triplet_mrg03_wt10_wx10' :'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 1.0' ,
    'triplet_mrg001_wt10_wx10':'triplet margin: 0.01 weight_triplet: 1.0 weight_entropy: 1.0',
    'triplet_mrg005_wt10_wx10':'triplet margin: 0.05 weight_triplet: 1.0 weight_entropy: 1.0',
    'triplet_mrg01_wt10_wx10' :'triplet margin: 0.1 weight_triplet: 1.0 weight_entropy: 1.0' ,
    'triplet_mrg03_wt10_wx10' :'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 1.0' ,
    'triplet_mrg05_wt10_wx10' :'triplet margin: 0.5 weight_triplet: 1.0 weight_entropy: 1.0' ,
    'triplet_mrg05_wt10_wx00' :'triplet margin: 0.5 weight_triplet: 1.0 weight_entropy: 0.0' ,
    'triplet_mrg001_wt10_wx00':'triplet margin: 0.01 weight_triplet: 1.0 weight_entropy: 0.0',
    'triplet_mrg005_wt01_wx10':'triplet margin: 0.05 weight_triplet: 0.1 weight_entropy: 1.0',
    'triplet_mrg005_wt03_wx10':'triplet margin: 0.05 weight_triplet: 0.3 weight_entropy: 1.0',
    'triplet_mrg005_wt05_wx10':'triplet margin: 0.05 weight_triplet: 0.5 weight_entropy: 1.0',
    'triplet_mrg005_wt07_wx10':'triplet margin: 0.05 weight_triplet: 0.7 weight_entropy: 1.0',
    'triplet_mrg005_wt10_wx10':'triplet margin: 0.05 weight_triplet: 1.0 weight_entropy: 1.0',
    'triplet_mrg005_wt10_wx05':'triplet margin: 0.05 weight_triplet: 1.0 weight_entropy: 0.5',
}

def get_ranks(dataset, method, reid_network, reid_model_pth_name='pretrained', loss_name='softmax'):
    loss_str = ''
    if loss_name == 'triplet':
        loss_str = '_triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)
        dir_path = '../storage/results/pathtrack/{}/feats-raw-filtered%s/reid-feat-person-filtered/'.format(dataset) % loss_str
    elif loss_name == 'softmax':
        loss_str = '_softmax_rx_noexc'
        dir_path = '../storage/results/pathtrack/{}/feats-raw-filtered%s/reid-feat-person-filtered/'.format(dataset) % loss_str
    else:
        loss_str = ''
        dir_path = '../storage/results/pathtrack/{}/reid-feat-person-filtered/'.format(dataset)
    # dir_path = '../storage/results/pathtrack/{}/reid-feat-person-filtered/'.format(data_arg, dataset)
    dis_path = (dir_path + '{}-{}-dis-{}.txt').format(method, reid_network, reid_model_pth_name)
    match_path = ('../storage/results/pathtrack/matched.txt')
    plot_path = (dir_path + '{}-{}-attempts-{}').format(method, reid_network, reid_model_pth_name)
    dis_all_path = (dir_path + '{}-{}-allavgdis-{}').format(method, reid_network, reid_model_pth_name)
    num_match, match_pairs, avg_dict = 0, [], {}
    with open(match_path, 'r') as f:
        for line in f:  
            if not line.strip() or line.strip()[0] == '#':
                continue
            if dataset in line.strip().split('-')[0]:
                match_pairs.append(line.strip())
                num_match += 1
    dis_all = []
    with open(dis_path, 'r') as f:
        for line in f:  # HId1-HId2;min;max;avg;median;std;avg-std;avg+std
            if line.strip()[0] == '#':
                continue
            tmp = line.strip().split(';')
            pairs = tmp[0]
            _avg, _median = float(tmp[3]), float(tmp[4])
            dis_all.append(_avg)
            if pairs not in avg_dict and '{}-{}'.format(pairs.split('-')[1], pairs.split('-')[0]) not in avg_dict:
                avg_dict[pairs] = _avg
    print(dataset, loss_str, len(avg_dict.items()))
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
        if pairs in match_pairs or '{}-{}'.format(pairs.split('-')[1], pairs.split('-')[0]) in match_pairs:
            _y.append(attempts)
            txt += '%s-%s, avg_dis:%.2f, rank:%d\n' % (pairs.split('-')[0].split('_')[-1], pairs.split('-')[1], _avg, attempts)
    plt.text(.05, .8, txt, verticalalignment='center', horizontalalignment='left', transform=ax.transAxes)
    plt.savefig(dis_all_path)
    return _y

def plot_cmc(_ranks_map):
    plt.style.use('default')
    plt.figure()
    _x = np.arange(1, 100)
    # _x = np.arange(1, max(_ranks_map[list(_ranks_map.keys())[0]]) + 1, 1)
    for loss_str in _ranks_map:
        _ranks = _ranks_map[loss_str]
        rec = []
        for rk in _x:
            rec.append(np.sum(_ranks <= rk) / len(_ranks))
        plt.plot(_x, rec, '-', label=loss_str, alpha=0.7)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.grid(which='both', axis='both')
    plt.xlabel('Rank')
    plt.ylabel('Recall (Identification Rate) (%)')
    plt.title('CMC Curve')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    # dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000', '0Xtp77A4zF4_3000_5000', '0Xtp77A4zF4_4000_6000', '0Xtp77A4zF4_5000_7000', '0Xtp77A4zF4_6000_8000', '0Xtp77A4zF4_7000_9000', '0Xtp77A4zF4_8000_10000', '0Xtp77A4zF4_9000_11000', '0Xtp77A4zF4_10000_12000', '0Xtp77A4zF4_11000_13000', '0Xtp77A4zF4_12000_14000', '0Xtp77A4zF4_13000_15000', '0Xtp77A4zF4_14000_16000', '0Xtp77A4zF4_15000_17000', '0Xtp77A4zF4_16000_18000', '0Xtp77A4zF4_17000_19000', '0Xtp77A4zF4_18000_20000', '0Xtp77A4zF4_19000_21000', '0Xtp77A4zF4_20000_21635',]
    # dataset_names = ['4LFwzgwRyrY_0_2000', '4LFwzgwRyrY_1000_3000', '4LFwzgwRyrY_2000_4000', '4LFwzgwRyrY_3000_5000', '4LFwzgwRyrY_4000_6000', '4LFwzgwRyrY_5000_7000', '4LFwzgwRyrY_6000_8000', '4LFwzgwRyrY_7000_9000', '4LFwzgwRyrY_8000_10000', '4LFwzgwRyrY_9000_11000', '4LFwzgwRyrY_10000_12000', '4LFwzgwRyrY_11000_12000', ]
    # dataset_names = ['5fhJSO5al8o_0_2000', '5fhJSO5al8o_1000_3000', '5fhJSO5al8o_2000_4000', '5fhJSO5al8o_3000_5000', '5fhJSO5al8o_4000_6000', '5fhJSO5al8o_5000_7000', '5fhJSO5al8o_6000_8000', '5fhJSO5al8o_7000_9000', '5fhJSO5al8o_8000_10000', '5fhJSO5al8o_9000_11000', '5fhJSO5al8o_10000_12000', '5fhJSO5al8o_11000_13000', '5fhJSO5al8o_12000_14000', '5fhJSO5al8o_13000_15000', '5fhJSO5al8o_14000_16000', '5fhJSO5al8o_15000_16000', ]
    # dataset_names = ['81ahuidFUWw_0_2000', '81ahuidFUWw_1000_3000', '81ahuidFUWw_2000_4000', '81ahuidFUWw_3000_5000', '81ahuidFUWw_4000_6000', '81ahuidFUWw_5000_7000', '81ahuidFUWw_6000_8000', '81ahuidFUWw_7000_9000', '81ahuidFUWw_8000_10000', '81ahuidFUWw_9000_11000', '81ahuidFUWw_10000_12000', '81ahuidFUWw_11000_13000', '81ahuidFUWw_12000_14000', '81ahuidFUWw_13000_15000', '81ahuidFUWw_14000_16000', '81ahuidFUWw_15000_17000', '81ahuidFUWw_16000_17500', ]
    # dataset_names = ['KQqFixbb-o8_0_2000', 'KQqFixbb-o8_1000_3000', 'KQqFixbb-o8_2000_4000', 'KQqFixbb-o8_3000_5000', 'KQqFixbb-o8_4000_6000', 'KQqFixbb-o8_5000_7000', 'KQqFixbb-o8_6000_8000', 'KQqFixbb-o8_7000_9000', 'KQqFixbb-o8_8000_9099']
    # dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000']
    dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000', '0Xtp77A4zF4_3000_5000', '0Xtp77A4zF4_4000_6000']

    ranks_map = {}  # {method: [ranks]}
    for idx, dn in enumerate(dataset_names):
        # if 'softmax' not in ranks_map:
        #     ranks_map['softmax'] = []
        # ranks = get_ranks(dn, 'tracktor', 'osnet_x1_0', 'mot3', loss_name='')
        # ranks_map['softmax'] += ranks
        for mrg, wt, wx in zip(['005'], 
                               ['10' ], 
                               ['05' ]):
            if name_map['triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)] not in ranks_map:
                ranks_map[name_map['triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)]] = []
            ranks = get_ranks(dn, 'tracktor', 'osnet_x1_0', 'mot3', loss_name='triplet')
            ranks_map[name_map['triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)]] += ranks
    print(ranks_map)
    
    # # ranks_map = {'triplet margin: 0.05 weight_triplet: 1.0 weight_entropy: 0.5': [1, 2, 4, 11, 1, 2, 5, 7, 9, 14, 23, 25, 33, 44, 2, 1, 2, 3, 5, 43, 1, 2, 3, 4, 5, 20]}
    # ranks_map = {
    #     # 'ori': [1, 2, 4, 11, 1, 2, 5, 7, 9, 14, 23, 25, 33, 44, 2, 1, 2, 3, 5, 43, 1, 2, 3, 4, 5, 20],
    #     'Lmax=500': [1, 1, 2, 3, 4, 7, 21, 1, 2, 3, 4, 5, 9, 10, 17, 25, 27, 35, 47, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 16, 51, 1, 2, 3, 4, 5, 6, 7, 8, 9, 57, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 40, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 26],
    #     'Lmax=1000': [1, 2, 5, 17, 1, 2, 3, 4, 5, 9, 11, 18, 29, 35, 39, 49, 1, 5, 7, 13, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 17, 92, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 55],
    #     'Lmax=1500': [1, 2, 3, 6, 8, 9, 11, 17, 24, 33, 36, 49, 51, 70, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 16, 19, 20, 51, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 19, 23, 27, 40, 75, 162],
    #     'Lmax=2000': [1, 2, 3, 6, 8, 9, 12, 18, 26, 35, 38, 51, 53, 72, 1, 2, 4, 6, 8, 21, 30, 69, 136],
    #     'Lmax=3000': [1, 2, 3, 6, 8, 9, 12, 21, 31, 33, 42, 46, 65, 67, 87],
    # }
    plot_cmc(ranks_map)
