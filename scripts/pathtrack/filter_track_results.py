import os 
from cv2 import cv2
import motmetrics as mm

def preprocess(dataset, method, train_test='selected', obj_frame_threshold=100):  # 100
    method_result_template = '../storage/results/pathtrack/{}/faster_rcnn-{}-person.txt'
    dataset_img_path = '../storage/dataset/PathTrack/{}/{}/img1/'
    output_template = '../storage/results/pathtrack/{}/filtered-tracked/faster_rcnn-{}-person.txt'

    # read.
    dt_result = mm.io.loadtxt(method_result_template.format(dataset, method))
    dt_result.reset_index(level=['FrameId', 'Id'], inplace=True)
    
    def _filter_short_objs(dt_result):
        id_length = dt_result.groupby('Id')['FrameId'].count().reset_index(name='Count')
        selected_id = set(id_length.loc[id_length['Count'] >= obj_frame_threshold]['Id'])
        dt_result = dt_result[dt_result['Id'].isin(selected_id)]
        return dt_result

    def _filter_bboxes_on_borders(dt_result):
        ip = dataset_img_path.format(train_test, dataset)
        frame_template = cv2.imread(ip + os.listdir(ip)[0])
        height, width, _ = frame_template.shape
        dt_result = dt_result[
            (dt_result['X'] > 0) & (dt_result['Y'] > 0) &
            (dt_result['X'] + dt_result['Width'] < width) &
            (dt_result['Y'] + dt_result['Height'] < height)
        ]
        return dt_result
    
    def _filter_small_bboxes(dt_result):
        min_width, min_height = 20, 30
        dt_result = dt_result[
            (dt_result['Width'] >= min_width) & (dt_result['Height'] >= min_height)
        ]
        return dt_result

    print(dataset, 'origin feats:', len(dt_result), end=', new feats: ')
    # # 1. filter out short objs.
    dt_result = _filter_short_objs(dt_result)
    # # 2. filter out detections on the screen edges
    dt_result = _filter_bboxes_on_borders(dt_result)
    # # 3. filter out small boxes
    dt_result = _filter_small_bboxes(dt_result)
    print(len(dt_result), end=', ')
    print('tracks:', len(dt_result.groupby('Id')), ', avg bboxes:', round(len(dt_result)/len(dt_result.groupby('Id')), 4))
    # print(dt_result)

    save_tracked_txt(output_template.format(dataset, method), dt_result)


def save_tracked_txt(path, pf):
    parent_dir = os.path.dirname(path)
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    with open(path, 'w') as f:
        for _, row in pf.iterrows():
            f.write('{:.0f},{:.0f},{},{},{},{},{},{:.0f},{:.0f},-1\n'.format(
                row['FrameId'], row['Id'], row['X'], row['Y'], row['Width'], 
                row['Height'],row['Confidence'], row['ClassId'], row['Visibility']
            ))
    

if __name__ == '__main__':
    dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000', '0Xtp77A4zF4_3000_5000', '0Xtp77A4zF4_4000_6000', ]
    # dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000', '0Xtp77A4zF4_3000_5000', '0Xtp77A4zF4_4000_6000', '0Xtp77A4zF4_5000_7000', '0Xtp77A4zF4_6000_8000', '0Xtp77A4zF4_7000_9000', '0Xtp77A4zF4_8000_10000', '0Xtp77A4zF4_9000_11000', '0Xtp77A4zF4_10000_12000', '0Xtp77A4zF4_11000_13000', '0Xtp77A4zF4_12000_14000', '0Xtp77A4zF4_13000_15000', '0Xtp77A4zF4_14000_16000', '0Xtp77A4zF4_15000_17000', '0Xtp77A4zF4_16000_18000', '0Xtp77A4zF4_17000_19000', '0Xtp77A4zF4_18000_20000', '0Xtp77A4zF4_19000_21000', '0Xtp77A4zF4_20000_21635',]
    # dataset_names = ['4LFwzgwRyrY_0_2000', '4LFwzgwRyrY_1000_3000', '4LFwzgwRyrY_2000_4000', '4LFwzgwRyrY_3000_5000', '4LFwzgwRyrY_4000_6000', '4LFwzgwRyrY_5000_7000', '4LFwzgwRyrY_6000_8000', '4LFwzgwRyrY_7000_9000', '4LFwzgwRyrY_8000_10000', '4LFwzgwRyrY_9000_11000', '4LFwzgwRyrY_10000_12000', '4LFwzgwRyrY_11000_12000', ]
    # dataset_names = ['5fhJSO5al8o_0_2000', '5fhJSO5al8o_1000_3000', '5fhJSO5al8o_2000_4000', '5fhJSO5al8o_3000_5000', '5fhJSO5al8o_4000_6000', '5fhJSO5al8o_5000_7000', '5fhJSO5al8o_6000_8000', '5fhJSO5al8o_7000_9000', '5fhJSO5al8o_8000_10000', '5fhJSO5al8o_9000_11000', '5fhJSO5al8o_10000_12000', '5fhJSO5al8o_11000_13000', '5fhJSO5al8o_12000_14000', '5fhJSO5al8o_13000_15000', '5fhJSO5al8o_14000_16000', '5fhJSO5al8o_15000_16000', ]
    # dataset_names = ['81ahuidFUWw_0_2000', '81ahuidFUWw_1000_3000', '81ahuidFUWw_2000_4000', '81ahuidFUWw_3000_5000', '81ahuidFUWw_4000_6000', '81ahuidFUWw_5000_7000', '81ahuidFUWw_6000_8000', '81ahuidFUWw_7000_9000', '81ahuidFUWw_8000_10000', '81ahuidFUWw_9000_11000', '81ahuidFUWw_10000_12000', '81ahuidFUWw_11000_13000', '81ahuidFUWw_12000_14000', '81ahuidFUWw_13000_15000', '81ahuidFUWw_14000_16000', '81ahuidFUWw_15000_17000', '81ahuidFUWw_16000_17500', ]
    # dataset_names = ['KQqFixbb-o8_0_2000', 'KQqFixbb-o8_1000_3000', 'KQqFixbb-o8_2000_4000', 'KQqFixbb-o8_3000_5000', 'KQqFixbb-o8_4000_6000', 'KQqFixbb-o8_5000_7000', 'KQqFixbb-o8_6000_8000', 'KQqFixbb-o8_7000_9000', 'KQqFixbb-o8_8000_9099']
    # dataset_names = []

    for dn in dataset_names:
        preprocess(dn, 'tracktor')
