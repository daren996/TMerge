import os 
from cv2 import cv2
import motmetrics as mm

def preprocess(dataset, method, obj_frame_threshold=100):
    method_result_template = '../storage/results/mot17/{}/faster_rcnn-{}-person.txt'
    dataset_img_path = '../storage/dataset/MOT17/train/{}/img1/{:06d}.jpg'
    output_template = '../storage/results/mot17/{}/filtered-tracked/faster_rcnn-{}-person.txt'

    # read.
    dt_result = mm.io.loadtxt(method_result_template.format(dataset, method))
    # dt_result = mm.io.loadtxt(output_template.format(dataset, method))
    # print(dt_result)
    # return
    # add columns fid, id
    dt_result.reset_index(level=['FrameId', 'Id'], inplace=True)
    # print(dt_result.index)
    # print(dt_result)
    
    def _filter_short_objs(dt_result):
        id_length = dt_result.groupby('Id')['FrameId'].count().reset_index(name='Count')
        selected_id = set(id_length.loc[id_length['Count'] >= obj_frame_threshold]['Id'])
        # print(selected_id)

        dt_result = dt_result[dt_result['Id'].isin(selected_id)]
        return dt_result

    def _filter_bboxes_on_borders(dt_result):
        frame_template = cv2.imread(dataset_img_path.format(dataset, 1))
        height, width, channels = frame_template.shape
        dt_result = dt_result[
            (dt_result['X'] > 0) & (dt_result['Y'] > 0) &
            (dt_result['X'] + dt_result['Width'] < width) &
            (dt_result['Y'] + dt_result['Height'] < height)
        ]
        return dt_result

    # 1. filter out short objs.
    dt_result = _filter_short_objs(dt_result)
    # print(dt_result)
    # 2. filter out detections on the screen edges
    dt_result = _filter_bboxes_on_borders(dt_result)
    print(dt_result)

    save_tracked_txt(output_template.format(dataset, method), dt_result)


def save_tracked_txt(path, pf):
    parent_dir = os.path.dirname(path)
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    with open(path, 'w') as f:
        for idx, row in pf.iterrows():
            f.write('{:.0f},{:.0f},{},{},{},{},{},{:.0f},{:.0f},-1\n'.format(
                row['FrameId'], row['Id'], row['X'], row['Y'], row['Width'], 
                row['Height'],row['Confidence'], row['ClassId'], row['Visibility']
            ))
    

if __name__ == '__main__':
    # preprocess('MOT17-04-DPM', 'deepsort')
    # preprocess('MOT17-04-DPM', 'tracktor')
    # preprocess('MOT17-02-DPM', 'deepsort')
    preprocess('MOT17-02-DPM', 'tracktor')
    preprocess('MOT17-04-DPM', 'tracktor')
    preprocess('MOT17-05-DPM', 'tracktor')
    preprocess('MOT17-09-DPM', 'tracktor')
    preprocess('MOT17-10-DPM', 'tracktor')
    preprocess('MOT17-11-DPM', 'tracktor')
    preprocess('MOT17-13-DPM', 'tracktor')
