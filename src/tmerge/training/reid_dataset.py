"""Custom TorchReID datasets used by TMerge training."""

from __future__ import annotations

import configparser
import csv
import os.path as osp
from typing import Any

import numpy as np

from tmerge.training import reid_parser as kitti_parser


def _require_torchreid() -> Any:
    try:
        from torchreid.data.datasets import Dataset, ImageDataset
        from torchreid.utils import read_image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TorchReID dataset support requires optional dependencies. "
            'Install with: python -m pip install -e ".[reid]"'
        ) from exc
    return Dataset, ImageDataset, read_image


Dataset, ImageDataset, read_image = _require_torchreid()


class RedefinedDataset(Dataset):
    def __init__(
        self,
        train,
        query,
        gallery,
        transform=None,
        k_tfm=1,
        mode="train",
        combineall=False,
        verbose=True,
        **kwargs,
    ):
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

        if self.mode == "train":
            self.data = self.train
        elif self.mode == "query":
            self.data = self.query
        elif self.mode == "gallery":
            self.data = self.gallery
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.verbose:
            self.show_summary()

    def get_num_datasets(self, data):
        dsets = set()
        for items in data:
            dsetid = items[5]
            dsets.add(dsetid)
        return len(dsets)

    def _crop_and_transform(self, index):
        """Shared item loading logic for bbox-based ReID datasets."""
        img_path, pid, camid, bbox, vis, dsetid = self.data[index]
        img = read_image(img_path)

        width, height = img.size
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        context = 0
        bbox[0] = np.clip(bbox[0] - context * w, 0, width - 1)
        bbox[1] = np.clip(bbox[1] - context * h, 0, height - 1)
        bbox[2] = np.clip(bbox[2] + context * w, 0, width - 1)
        bbox[3] = np.clip(bbox[3] + context * h, 0, height - 1)

        img = img.crop(bbox)

        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        return {"img": img, "pid": pid, "camid": camid, "impath": img_path, "dsetid": dsetid}


class MOTDataset(RedefinedDataset):
    train_sequences = [
        "MOT17-02",
        "MOT17-04",
        "MOT17-05",
        "MOT17-09",
        "MOT17-10",
        "MOT17-11",
        "MOT17-13",
    ]
    dataset_url = None

    def __init__(self, root="", mot_vis_threshold=0.3, mot_exclude_sequences=None, **kwargs):
        self.root = osp.abspath(root)
        self._vis_threshold = mot_vis_threshold
        self.dataset_dir = osp.join(self.root, "train")
        self.data_dir = self.dataset_dir
        excluded = mot_exclude_sequences or []
        train_seqs = set(self.train_sequences).difference(set(excluded))
        required_files = [osp.join(self.data_dir, seq) for seq in train_seqs]
        self.check_before_run(required_files)

        data = []
        for camid, seq in enumerate(required_files, 1):
            data.extend(self._collect_data(seq, camid))
        pid_container = {item[1] for item in data}
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        train_data = [(item[0], pid2label[item[1]], item[2], item[3], item[4]) for item in data]

        super().__init__(train_data, data, data, **kwargs)

    def _collect_data(self, seq_path, camid):
        config_file = osp.join(seq_path, "seqinfo.ini")
        assert osp.exists(config_file), f"Config file does not exist: {config_file}"

        config = configparser.ConfigParser()
        config.read(config_file)
        seq_length = int(config["Sequence"]["seqLength"])
        image_dir = osp.join(seq_path, config["Sequence"]["imDir"])
        gt_file = osp.join(seq_path, "gt", "gt.txt")

        total = []
        visibility = {i: {} for i in range(1, seq_length + 1)}
        boxes = {i: {} for i in range(1, seq_length + 1)}

        with open(gt_file, encoding="utf-8") as inf:
            reader = csv.reader(inf, delimiter=",")
            for row in reader:
                if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                    x1 = int(row[2]) - 1
                    y1 = int(row[3]) - 1
                    x2 = x1 + int(row[4]) - 1
                    y2 = y1 + int(row[5]) - 1
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                    frame_id = int(row[0])
                    person_id = int(row[1])
                    boxes[frame_id][person_id] = bbox
                    visibility[frame_id][person_id] = float(row[8])

        for frame_id in range(1, seq_length + 1):
            image_path = osp.join(image_dir, f"{frame_id:06d}.jpg")
            for person_id in boxes[frame_id]:
                total.append(
                    (
                        image_path,
                        person_id + camid * 1000,
                        camid,
                        boxes[frame_id][person_id],
                        visibility[frame_id][person_id],
                    )
                )
        return total

    def __getitem__(self, index):
        return self._crop_and_transform(index)


class KITTIDataset(RedefinedDataset):
    train_sequences = ["0013", "0014", "0015", "0016", "0017", "0019"]
    dataset_url = None

    def __init__(self, root="", kitti_exclude_sequences=None, **kwargs):
        self.root = osp.abspath(root)
        self.dataset_dir = osp.join(self.root, "training")
        self.data_dir = self.dataset_dir
        excluded = kitti_exclude_sequences or []
        train_seqs = sorted(set(self.train_sequences).difference(set(excluded)))
        vid2labelfile = {vid: osp.join(self.data_dir, f"label_02/{vid}.txt") for vid in train_seqs}
        vid2imagefolder = {vid: osp.join(self.data_dir, f"image_02/{vid}/") for vid in train_seqs}
        required_files = [vid2imagefolder[vid] for vid in train_seqs]
        self.check_before_run(required_files)
        self.obj_specify = ["Pedestrian", "Person", "Cyclist"]

        data = []
        for camid, vid in enumerate(train_seqs):
            data.extend(self._collect_data(vid, camid, vid2labelfile, vid2imagefolder))
        pid_container = {item[1] for item in data}
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        train_data = [(item[0], pid2label[item[1]], item[2], item[3], item[4]) for item in data]
        super().__init__(train_data, data, data, **kwargs)

    def _collect_data(self, vid, camid, vid2labelfile, vid2imagefolder):
        fid2record = kitti_parser.get_fid2record(vid2labelfile, vid, self.obj_specify)
        fid_all = sorted(fid2record.keys())
        total = []
        visibility = {fid: {} for fid in fid_all}
        boxes = {fid: {} for fid in fid_all}

        for fid in fid_all:
            for record in fid2record[fid]:
                bbox = np.array(
                    [record["left"], record["top"], record["right"], record["bottom"]],
                    dtype=np.float32,
                )
                boxes[fid][record["tid"]] = bbox
                visibility[fid][record["tid"]] = 1

        for fid in fid_all:
            image_path = osp.join(vid2imagefolder[vid], f"{fid:06d}.png")
            for pid in boxes[fid]:
                total.append(
                    (
                        image_path,
                        pid + camid * 1000,
                        camid,
                        boxes[fid][pid],
                        visibility[fid][pid],
                    )
                )
        return total

    def __getitem__(self, index):
        return self._crop_and_transform(index)

