import os.path as osp
import csv
import copy
import configparser
import numpy as np
import tarfile
import zipfile
from torchreid.utils import read_image, download_url, mkdir_if_missing
from torchreid.data.datasets.dataset import Dataset
from torchreid.data.datasets import ImageDataset, VideoDataset

# code modified from the following projects: 
#   https://github.com/phil-bergmann/tracking_wo_bnw
#   https://github.com/KaiyangZhou/deep-person-reid

class RedefinedDataset(Dataset):

    def __init__(
        self,
        train,
        query,
        gallery,
        transform=None,
        k_tfm=1,
        mode='train',
        combineall=False,
        verbose=True,
        **kwargs
    ):
        # extend 3-tuple (img_path(s), pid, camid) to
        # 4-tuple (img_path(s), pid, camid, dsetid) by
        # adding a dataset indicator "dsetid"
        if len(train[0]) == 5:
            train = [(*items, 0) for items in train]
        if len(query[0]) == 5:
            query = [(*items, 0) for items in query]
        if len(gallery[0]) == 5:
            gallery = [(*items, 0) for items in gallery]

        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.k_tfm = k_tfm
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
        self.num_datasets = self.get_num_datasets(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | query | gallery]'.format(self.mode)
            )

        if self.verbose:
            self.show_summary()

    def get_num_datasets(self, data):
        """Returns the number of datasets included.

        Each tuple in data contains (img_path(s), pid, camid, bbox, vis, dsetid).
        """
        dsets = set()
        for items in data:
            dsetid = items[5]
            dsets.add(dsetid)
        return len(dsets)


class MOTDataset(RedefinedDataset):

    train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
                       'MOT17-10', 'MOT17-11', 'MOT17-13']
    dataset_url = None

    def __init__(self, root='', mot_vis_threshold=0.3, mot_exclude_sequences=[], **kwargs):
        self.root = osp.abspath(root)
        self._vis_threshold=mot_vis_threshold
        self.dataset_dir = osp.join(self.root, 'train')
        self.data_dir = self.dataset_dir
        train_seqs = set(self.train_sequences).difference(set(mot_exclude_sequences))
        print('mot training sequences:', train_seqs)
        required_files = [osp.join(self.data_dir, seq) for seq in train_seqs]

        self.check_before_run(required_files)
        # collect data.
        data = []
        for camid, seq in enumerate(required_files, 1):
            data.extend(self._collect_data(seq, camid))
        # convert pids.
        pid_container = set()
        for item in data:
            pid_container.add(item[1])
        
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        train_data = [(item[0], pid2label[item[1]], item[2], item[3], item[4]) for item in data]

        super().__init__(train_data, data, data, **kwargs)

    def _collect_data(self, seq_path, camid):
        config_file = osp.join(seq_path, 'seqinfo.ini')
        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']

        imDir = osp.join(seq_path, imDir)
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')

        total = []

        visibility = {}
        boxes = {}

        for i in range(1, seqLength+1):
            boxes[i] = {}
            visibility[i] = {}

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                # class person, certainity 1, visibility >= 0.25
                if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                    # Make pixel indexes 0-based, should already be 0-based (or not)
                    x1 = int(row[2]) - 1
                    y1 = int(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + int(row[4]) - 1
                    y2 = y1 + int(row[5]) - 1
                    bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                    boxes[int(row[0])][int(row[1])] = bb
                    visibility[int(row[0])][int(row[1])] = float(row[8])

        for i in range(1, seqLength + 1):
            im_path = osp.join(imDir, f"{i:06d}.jpg")

            # add all ids
            for pid in boxes[i]:
                # im_path, pid, bbox, vis, camid
                
                total.append((im_path, pid + camid * 1000, camid, boxes[i][pid], visibility[i][pid]))
        return total

    def __getitem__(self, index):
        # dsetid: id of the dataset
        img_path, pid, camid, bbox, vis, dsetid = self.data[index]
        img = read_image(img_path)
        # crop

        width, height = img.size
        #blobs, im_scales = _get_blobs(im)
        #im = blobs['data'][0]
        #gt = gt * im_scales[0]
        # clip to image boundary
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        context = 0
        bbox[0] = np.clip(bbox[0]-context*w, 0, width-1)
        bbox[1] = np.clip(bbox[1]-context*h, 0, height-1)
        bbox[2] = np.clip(bbox[2]+context*w, 0, width-1)
        bbox[3] = np.clip(bbox[3]+context*h, 0, height-1)

        img = img.crop(bbox)

        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid
        }
        return item

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  ----------------------------------------')  
    