import os 
import cv2
import motmetrics as mm

def preprocess(video_name, method, train_test='training', obj_frame_threshold=10):
    method_result_template = '../storage/results/kitti/{}/faster_rcnn-{}-person.txt'
    dataset_img_path = '../storage/dataset/KITTI/{}/image_02/{}/'
    output_template = '../storage/results/kitti/{}/filtered-tracked/faster_rcnn-{}-person.txt'

    # read.
    dt_result = mm.io.loadtxt(method_result_template.format(video_name, method))
    dt_result.reset_index(level=['FrameId', 'Id'], inplace=True)
    
    def _filter_short_objs(dt_result):
        id_length = dt_result.groupby('Id')['FrameId'].count().reset_index(name='Count')
        selected_id = set(id_length.loc[id_length['Count'] >= obj_frame_threshold]['Id'])
        dt_result = dt_result[dt_result['Id'].isin(selected_id)]
        return dt_result

    def _filter_bboxes_on_borders(dt_result):
        ip = dataset_img_path.format(train_test, video_name)
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

    print(video_name, 'origin feats:', len(dt_result), end=', new feats: ')
    # # 1. filter out short objs.
    dt_result = _filter_short_objs(dt_result)
    # # 2. filter out detections on the screen edges
    # dt_result = _filter_bboxes_on_borders(dt_result)
    # # 3. filter out small boxes
    # dt_result = _filter_small_bboxes(dt_result)
    print(len(dt_result), end=', ')
    print('tracks:', len(dt_result.groupby('Id')), ', avg bboxes:', 
          round(len(dt_result)/len(dt_result.groupby('Id')), 4))
    # print(dt_result)

    save_tracked_txt(output_template.format(video_name, method), dt_result)


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
    video_names = ['0013', '0014', '0015', '0016', '0017', '0019']
    for dn in video_names:
        preprocess(dn, 'tracktor')
