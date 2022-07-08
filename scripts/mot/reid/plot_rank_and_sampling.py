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

def distance_distribution(dataset, method, reid_network, reid_model_pth_name='pretrained', data_arg=''):
    dir_path = '../storage/results/mot17{}/{}/reid-feat-person-filtered/'.format(data_arg, dataset)
    dis_path = (dir_path + '{}-{}-dis-{}.txt').format(method, reid_network, reid_model_pth_name)
    dis_all_path = (dir_path + '{}-{}-allavgdis-{}').format(method, reid_network, reid_model_pth_name)
    dis_all, avg_dict = [], {}
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
    for pairs, _avg in sorted(avg_dict.items(), key=lambda x: x[1]):
        print("{}\t{}\t".format(pairs, _avg))
    plt.figure()
    snsplot = sns.displot(dis_all)
    ax = snsplot.ax
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(dataset)
    plt.xlabel('Avg. Distance')
    plt.tight_layout()
    plt.savefig(dis_all_path)

def get_ranks(dataset, method, reid_network, reid_model_pth_name='pretrained', 
        data_arg='', loss_name='softmax'):
    # data_arg: '', '(noexc)', '(noexc;bgs)'
    # mrg, wt, wx = '001', '10', '00'
    loss_str = ''
    if loss_name == 'triplet':
        loss_str = '_triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)
        dir_path = '../storage/results/mot17{}/{}/feats-raw-filtered%s/reid-feat-person-filtered/'.format(data_arg, dataset) % loss_str
    elif loss_name == 'softmax':
        loss_str = '_softmax_rx_noexc'
        dir_path = '../storage/results/mot17{}/{}/feats-raw-filtered%s/reid-feat-person-filtered/'.format(data_arg, dataset) % loss_str
    else:
        loss_str = ''
        dir_path = '../storage/results/mot17{}/{}/reid-feat-person-filtered/'.format(data_arg, dataset)
    # dir_path = '../storage/results/mot17{}/{}/reid-feat-person-filtered/'.format(data_arg, dataset)
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
    print(loss_str, len(avg_dict.items()))
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
    return _y

def plot_cmc(_ranks_map, fontsize=12):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (6.0, 2.8)
    # plt.style.use('default')
    plt.figure()
    _x = np.arange(1, 81, 1)
    for loss_str in _ranks_map:
        _ranks = _ranks_map[loss_str]
        rec = []
        for rk in _x:
            rec.append(np.sum(_ranks <= rk) / len(_ranks))
        plt.plot(_x, rec, '-', label=loss_str, linewidth=2, alpha=0.7)
    plt.xticks(fontsize=fontsize)
    plt.xlim(0, 80)
    plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=fontsize)
    plt.ylim(0.5, 1)
    plt.grid(linestyle='--', c='black', axis="both")
    plt.xlabel('$K$', fontsize=fontsize)
    plt.ylabel('$REC$', fontsize=fontsize)
    # plt.title('CMC Curve')
    plt.legend(loc='lower right', fontsize=fontsize)
    # plt.show()
    plt.savefig('../storage/results/k_vs_rec.pdf', bbox_inches='tight')

if __name__ == '__main__':
    # for did in ['04', '09', '11']:
    #     get_ranks('MOT17-%s-FRCNN' % did, 'tracktor', 'osnet_x1_0', 'mot3', data_arg='')
    # for did in ['01', '04', '06', '07', '08', '09', '11', '12', '14']:
    #     distance_distribution('MOT17-%s-FRCNN' % did, 'tracktor', 'osnet_x1_0', 'mot3', data_arg='')
    
    # ranks_map = {'softmax': []}  # {method: [ranks]}
    # for did in ['01', '04', '06', '07', '08', '09', '11', '12', '14']:  # '01', '04', '06', '07', '08', '09', '11', '12', '14'
    #     print('MOT17-%s-FRCNN' % did)
    #     ranks = get_ranks('MOT17-%s-FRCNN' % did, 'tracktor', 'osnet_x1_0', 'mot3', data_arg='', loss_name='')
    #     ranks_map['softmax'] += ranks
    #     for mrg, wt, wx in zip(['03', '03', '03', '001', '005', '01', '05', '001', '05', '005', '005', '005', '005', '005'], 
    #                            ['10', '10', '10', '10',  '10' , '10', '10', '10',  '10', '01',  '03',  '05',  '07',  '10' ], 
    #                            ['00', '05', '10', '10',  '10' , '10', '10', '00',  '00', '10',  '10',  '10',  '10',  '05' ]):
    #         if name_map['triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)] not in ranks_map:
    #             ranks_map[name_map['triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)]] = []
    #         ranks = get_ranks('MOT17-%s-FRCNN' % did, 'tracktor', 'osnet_x1_0', 'mot3', data_arg='', loss_name='triplet')
    #         ranks_map[name_map['triplet_mrg{}_wt{}_wx{}'.format(mrg, wt, wx)]] += ranks
    # print(ranks_map)
    
    ranks_map = {
        # 'softmax': [1, 2, 3, 4, 1, 2, 3, 5, 9, 10, 73, 1, 2, 3, 4, 5, 2, 3, 4, 5, 7, 8, 13, 35, 1, 6, 8, 11, 12, 17, 20, 1, 1, 2, 4, 27, 2, 23], 
        # 'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 0.0': [1, 2, 5, 15, 3, 5, 41, 99, 104, 116, 312, 1, 2, 3, 4, 11, 7, 9, 16, 20, 97, 240, 257, 5, 15, 16, 23, 25, 59, 87, 2, 1, 6, 13, 8, 53], 
        # 'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 0.5': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 95, 1, 2, 3, 4, 5, 1, 2, 4, 5, 6, 7, 13, 28, 1, 2, 5, 11, 14, 22, 24, 1, 1, 2, 4, 6, 3, 6], 
        # 'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 115, 1, 2, 3, 5, 6, 1, 2, 3, 5, 7, 9, 13, 17, 1, 3, 5, 12, 19, 22, 44, 1, 1, 2, 5, 6, 1, 3], 
        # 'triplet margin: 0.01 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 7, 117, 1, 2, 3, 4, 5, 1, 2, 4, 5, 6, 8, 9, 27, 1, 7, 10, 11, 15, 20, 21, 1, 1, 2, 3, 8, 1, 6], 
        # 'triplet margin: 0.05 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 60, 1, 2, 3, 4, 5, 1, 2, 3, 4, 7, 9, 11, 67, 1, 2, 5, 6, 12, 16, 26, 1, 1, 2, 4, 11, 3, 10],
        # 'triplet margin: 0.1 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 148, 1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 7, 8, 62, 1, 4, 10, 11, 14, 22, 26, 1, 1, 2, 3, 5, 2, 11], 
        # 'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 115, 1, 2, 3, 5, 6, 1, 2, 3, 5, 7, 9, 13, 17, 1, 3, 5, 12, 19, 22, 44, 1, 1, 2, 5, 6, 1, 3], 
        # 'triplet margin: 0.5 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 221, 1, 2, 3, 4, 6, 1, 2, 3, 4, 7, 9, 11, 45, 1, 2, 6, 8, 22, 31, 49, 1, 1, 2, 3, 10, 3, 7], 
        # 'triplet margin: 0.5 weight_triplet: 1.0 weight_entropy: 0.0': [1, 2, 7, 12, 16, 28, 34, 179, 197, 210, 250, 1, 2, 3, 4, 6, 3, 5, 31, 72, 142, 247, 249, 8, 14, 15, 32, 44, 51, 72, 33, 1, 2, 20, 146, 31, 37], 
        # 'triplet margin: 0.01 weight_triplet: 1.0 weight_entropy: 0.0': [1, 3, 13, 50, 112, 122, 181, 319, 2, 4, 5, 6, 20, 33, 40, 134, 152, 194, 212, 279, 24, 67, 72, 73, 83, 109, 8, 7, 14, 94, 163, 35, 31], 
        # 'triplet margin: 0.05 weight_triplet: 0.1 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 5, 6, 9, 123, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 11, 14, 19, 1, 2, 4, 12, 15, 18, 27, 1, 1, 2, 6, 11, 2, 24], 
        # 'triplet margin: 0.05 weight_triplet: 0.3 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 7, 92, 1, 2, 3, 5, 6, 2, 3, 4, 6, 8, 11, 13, 25, 1, 2, 8, 14, 17, 28, 46, 1, 1, 2, 3, 7, 2, 9], 
        # 'triplet margin: 0.05 weight_triplet: 0.5 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 87, 1, 2, 3, 4, 5, 1, 2, 4, 5, 6, 8, 9, 16, 1, 4, 6, 11, 21, 25, 27, 1, 1, 2, 6, 9, 2, 6], 
        # 'triplet margin: 0.05 weight_triplet: 0.7 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 72, 1, 2, 3, 4, 5, 1, 2, 3, 4, 7, 8, 9, 15, 1, 5, 10, 12, 18, 22, 45, 1, 1, 2, 4, 5, 3, 3],
        # 'triplet margin: 0.05 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 60, 1, 2, 3, 4, 5, 1, 2, 3, 4, 7, 9, 11, 67, 1, 2, 5, 6, 12, 16, 26, 1, 1, 2, 4, 11, 3, 10],
        # 'triplet margin: 0.05 weight_triplet: 1.0 weight_entropy: 0.5': [1, 2, 3, 4, 1, 2, 3, 4, 5, 13, 60, 1, 2, 3, 4, 5, 1, 2, 4, 6, 7, 8, 11, 16, 1, 2, 11, 13, 17, 22, 24, 1, 1, 2, 4, 5, 3, 3]
        # 'softmax': [1, 2, 3, 4, 1, 2, 4, 5, 9, 10, 54, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 11, 66, 1, 3, 8, 9, 12, 14, 1, 1, 2, 4, 27, 2, 14], 
        # 'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 0.0': [1, 2, 4, 10, 2, 21, 60, 65, 74, 96, 265, 1, 2, 3, 9, 10, 2, 3, 4, 35, 38, 97, 211, 228, 2, 6, 7, 9, 10, 63, 1, 1, 5, 11, 5, 41], 
        # 'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 0.5': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 65, 1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 8, 20, 59, 1, 2, 6, 11, 17, 19, 1, 1, 2, 4, 6, 2, 4], 
        # 'triplet margin: 0.3 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 5, 1, 2, 3, 4, 5, 6, 84, 1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 7, 9, 48, 1, 2, 3, 13, 17, 37, 1, 1, 2, 5, 6, 1, 1], 
        # 'triplet margin: 0.01 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 7, 86, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 45, 1, 4, 8, 9, 12, 17, 1, 1, 2, 3, 8, 1, 3], 
        # 'triplet margin: 0.05 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 43, 1, 2, 3, 4, 5, 1, 2, 3, 4, 7, 8, 10, 60, 1, 2, 4, 5, 13, 21, 1, 1, 2, 4, 11, 2, 7], 
        # 'triplet margin: 0.1 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 115, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 65, 1, 2, 6, 9, 18, 21, 1, 1, 2, 3, 5, 2, 7], 
        # 'triplet margin: 0.5 weight_triplet: 1.0 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 221, 1, 2, 3, 4, 6, 1, 2, 3, 4, 7, 9, 11, 45, 1, 2, 6, 8, 22, 31, 49, 1, 1, 2, 3, 10, 3, 7], 
        # 'triplet margin: 0.01 weight_triplet: 1.0 weight_entropy: 0.0': [1, 3, 9, 34, 67, 77, 141, 291, 2, 5, 7, 13, 14, 49, 78, 86, 111, 139, 198, 233, 245, 18, 48, 53, 54, 62, 84, 5, 7, 13, 91, 163, 31, 57], 
        # 'triplet margin: 0.5 weight_triplet: 1.0 weight_entropy: 0.0': [1, 2, 4, 7, 13, 20, 25, 126, 143, 154, 197, 1, 2, 3, 4, 5, 1, 2, 7, 20, 68, 139, 206, 208, 4, 14, 20, 28, 30, 49, 28, 1, 2, 17, 143, 18, 27], 
        # 'triplet margin: 0.05 weight_triplet: 0.1 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 5, 6, 9, 72, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 10, 21, 1, 2, 3, 11, 13, 21, 1, 1, 2, 6, 11, 2, 17], 
        # 'triplet margin: 0.05 weight_triplet: 0.3 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 6, 8, 65, 1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 7, 8, 23, 1, 2, 6, 10, 22, 32, 1, 1, 2, 3, 7, 2, 7], 
        # 'triplet margin: 0.05 weight_triplet: 0.5 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 56, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 8, 29, 1, 3, 4, 16, 20, 22, 1, 1, 2, 6, 9, 2, 4], 
        # 'triplet margin: 0.05 weight_triplet: 0.7 weight_entropy: 1.0': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 57, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 44, 1, 3, 7, 8, 18, 31, 1, 1, 2, 4, 5, 2, 2], 
        # 'triplet margin: 0.05 weight_triplet: 1.0 weight_entropy: 0.5': [1, 2, 3, 4, 1, 2, 3, 4, 6, 13, 41, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 8, 19, 1, 2, 7, 8, 12, 17, 1, 1, 2, 4, 5, 3, 3],

        # 0.01, 1.0, 1.0
        "MOT-17": [1, 2, 3, 4, 1, 2, 3, 4, 5, 7, 86, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 45, 1, 4, 8, 9, 12, 17, 1, 1, 2, 3, 8, 1, 3],
        "PathTrack": [1, 2, 5, 17, 1, 2, 3, 4, 5, 9, 11, 18, 29, 35, 39, 49, 1, 5, 7, 13, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 17, 92, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 55],
    }
    plot_cmc(ranks_map)

