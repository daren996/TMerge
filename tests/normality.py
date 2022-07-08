
import numpy as np
from scipy import stats
import scripts.mot.reid.test_mot_person_reid_features_pairwise_average as track_result

def normality(dataset, method, reid_network, reid_model_pth_name='pretrained'):
    dir_path = '../storage/results/mot17/{}/'.format(dataset)
    match_path = (dir_path + 'reid-feat-person-filtered/{}-{}-match-{}.txt').format(method, reid_network, reid_model_pth_name)
    match_pairs = []
    with open(match_path, 'r') as f:
        for line in f:  # HId1-HId2
            if not line.strip() or line.strip()[0] == '#':
                continue
            hid1, hid2 = line.strip().split('-')[0], line.strip().split('-')[1]
            match_pairs.append('%s-%s' % (hid1, hid2) if int(hid1) <= int(hid2) else '%s-%s' % (hid2, hid1))

    feat_template = dir_path + 'feats-raw-filtered/faster_rcnn-{}-person-feat-{}-{}.pkl'
    pkl_file_path = feat_template.format(method , reid_network, reid_model_pth_name)
    hid_result_tuples_dict, _, _ = track_result.get_hid_result(pkl_file_path, 'avg')
    # hid_result_tuples_dict = {hid: (other_hid, avg/med, fid1, fid2, [dis])}
    count_normal, count_all = 0, 0
    for hid, results in hid_result_tuples_dict.items():
        for _, (other_hid, _, _, _, _distances) in enumerate(results[:10]):
            if True:  # all
            # if "%d-%d" % (hid, other_hid) in match_pairs:  # matched only
                x = _distances.flatten()
                # test_res = stats.kstes(x, 'norm', (np.mean(x), np.std(x)))
                test_res = stats.shapiro(x)
                count_all += 1
                if test_res[1] >= 0.05:
                    count_normal += 1
                # print(hid, other_hid, test_res.pvalue)
    return count_normal, count_all

count1_all, count2_all = 0, 0
for did in ['01', '04', '06', '07', '08', '09', '11', '12', '14']:
    count1, count2 = normality('MOT17-%s-FRCNN' % did, 'tracktor', 'osnet_x1_0', 'mot3')
    count1_all += count1
    count2_all += count2
print(count1_all / count2_all) 

