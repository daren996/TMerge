import motmetrics as mm
from motmetrics.distances import iou_matrix, norm2squared_matrix
from motmetrics.io import Format
import pandas as pd


def load_and_compute_mapping(file_path, gt_path, dist='iou',  distth=0.5):
    dt = mm.io.loadtxt(file_path, fmt=Format.MOT16)
    gt = mm.io.loadtxt(gt_path, fmt=Format.MOT16)
    acc = mm.utils.compare_to_groundtruth(gt, dt, dist=dist, distth=distth)
    events_df = acc.mot_events
    # print(events_df[events_df['HId'] == 1.0])
    # compute (gt_id, duration, event_dt)
    results = []
    for oid in events_df['OId'].unique():
        oid_events = events_df[events_df['OId'] == oid]
        results.append((oid, oid_events))
    return (results, events_df)

def generate_summary(oid_events, event_df):
    # oid, duration, hids,  # of hids, # of missing frames, # of hid frames, # of hid frames total

    results = []
    for oid, events in oid_events:
        duration = len(events.index.get_level_values('FrameId'))
        hid_frames = len(events[events['HId'].notnull()].index.get_level_values('FrameId'))
        hids = events[events['HId'].notnull()]['HId'].unique()
        # select hid 
        hid_frames_total = event_df[event_df['HId'].isin(hids)].index.get_level_values('FrameId')
        results.append((oid, duration, len(hids),
            hid_frames, duration - hid_frames, len(hid_frames_total), hids))
        # print('oid', oid, duration, hids, hid_frames, hid_frames_total)
    result_df = pd.DataFrame(data = results, columns=['oid', 'duration', 'hids',
        'hid_frames', 'miss_frames', 'hid_frames_total', 'hids_detail'])
    # sort 
    result_df = result_df.sort_values(by='duration', ascending=False)
    return result_df

def load_gt_with_threshold(gt_path, threshold=250):
    gt = mm.io.loadtxt(gt_path, fmt=Format.MOT16)
    grouped = gt.groupby(by = 'Id')
    duration_ids = pd.DataFrame(data={'duration': [], 'id': []})
    for name, group in grouped:
        duration = group.count().max()
        duration_ids = duration_ids.append({'duration': duration, 'id': name}, ignore_index=True)
    ids = duration_ids[duration_ids['duration'] >= threshold]
    ids = ids.sort_values(by='duration', ascending=False)
    return ids['id']
