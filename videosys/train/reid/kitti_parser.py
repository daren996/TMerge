"""
Training labels in KITTI dataset:
- **frame** Position of the sample within the sequence
- **track id** Tracking ID of the object within the sequence
- **type** Object type: 'Car', 'Pedestrian', 'Cyclist', 'Tram', 'Person_sitting', 'Misc' or 'DontCare'
- **truncated** Integer (0,1,2) indicating the level of truncation.
- **occluded** Integer (0,1,2,3) indicating occlusion state.
- **alpha** Observation angle of object, ranging [-Pi; Pi]
- **bbox 2D** (0-based) bounding box of the object: Left, top, right, bottom image coordinates
- **dimensions** 3D object dimensions: height, width, length [m]
- **location** 3D object location x,y,z in camera coords. [m]
- **rotation_y** Rotation around Y-axis in camera coords. [-Pi; Pi]
"""


def label_data_record(record):
    """
    Given a string 'record', this will
    output a dictionary of the appropriate labels
    and data types.
    """
    record_list = record.split(" ")
    record_dict = {
        "frame": int(record_list.pop(0)),
        "tid": int(record_list.pop(0)),
        "type":record_list.pop(0),
        "truncated":float(record_list.pop(0)),
        "occluded":int(record_list.pop(0)),
        "alpha":float(record_list.pop(0)),
        "left":float(record_list.pop(0)),
        "top":float(record_list.pop(0)),
        "right":float(record_list.pop(0)),
        "bottom":float(record_list.pop(0)),
        "height":float(record_list.pop(0)),
        "width":float(record_list.pop(0)),
        "length":float(record_list.pop(0)),
        "rotation_y":float(record_list.pop(0)),
        "x":float(record_list.pop(0)),
        "y":float(record_list.pop(0)),
        "z":float(record_list.pop(0)),
    }
    return record_dict


def get_fid2record(vid2labelfile, vid, obj_specify):
    fid2record = {}
    with open(vid2labelfile[vid]) as in_file:
        for line in in_file:
            single_record = label_data_record(line)
            fid = single_record['frame']
            if fid not in fid2record:
                fid2record[fid] = []
            if single_record['type'] in obj_specify:  # specify on pedestrian now
                fid2record[fid].append(single_record) 
    return fid2record


def extract_bbox(labeled_record):
    """
    Given a labeled record from the training data, 
    this will output a 3-tuple with the bounding box parameters.
    This output can be passed into Rectangle()
    """
    width = labeled_record["right"] - labeled_record["left"]
    height = labeled_record["bottom"] - labeled_record["top"]
    bbox = ((labeled_record["left"], labeled_record["top"]), width, height)
    return bbox


