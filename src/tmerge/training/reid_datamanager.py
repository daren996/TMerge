"""Customized TorchReID data manager used by TMerge training."""

from __future__ import annotations

from typing import Any


def _require_torchreid() -> tuple[Any, Any, Any]:
    try:
        from torchreid.data.datamanager import DataManager
        from torchreid.data.datasets import init_image_dataset
        from torchreid.data.sampler import build_train_sampler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TorchReID training requires optional dependencies. "
            'Install with: python -m pip install -e ".[reid]"'
        ) from exc
    return DataManager, init_image_dataset, build_train_sampler


DataManager, init_image_dataset, build_train_sampler = _require_torchreid()


class ImageDataManager(DataManager):
    data_type = "image"

    def __init__(
        self,
        root="",
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms="random_flip",
        k_tfm=1,
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        load_train_targets=False,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        num_instances=4,
        num_cams=1,
        num_datasets=1,
        train_sampler="RandomSampler",
        train_sampler_t="RandomSampler",
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        market1501_500k=False,
        mot_vis_threshold=0.3,
        mot_exclude_sequences=None,
        kitti_exclude_sequences=None,
    ):
        super().__init__(
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu,
        )

        import torch

        mot_exclude_sequences = mot_exclude_sequences or []
        kitti_exclude_sequences = kitti_exclude_sequences or []

        print("=> Loading train (source) dataset")
        trainset = []
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                transform=self.transform_tr,
                k_tfm=k_tfm,
                mode="train",
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
                mot_vis_threshold=mot_vis_threshold,
                mot_exclude_sequences=mot_exclude_sequences,
                kitti_exclude_sequences=kitti_exclude_sequences,
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            sampler=build_train_sampler(
                trainset.train,
                train_sampler,
                batch_size=batch_size_train,
                num_instances=num_instances,
                num_cams=num_cams,
                num_datasets=num_datasets,
            ),
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True,
        )

        self.train_loader_t = None
        if load_train_targets:
            assert len(set(self.sources) & set(self.targets)) == 0
            print("=> Loading train (target) dataset")
            trainset_t = []
            for name in self.targets:
                trainset_t_ = init_image_dataset(
                    name,
                    transform=self.transform_tr,
                    k_tfm=k_tfm,
                    mode="train",
                    combineall=False,
                    root=root,
                    split_id=split_id,
                    cuhk03_labeled=cuhk03_labeled,
                    cuhk03_classic_split=cuhk03_classic_split,
                    market1501_500k=market1501_500k,
                )
                trainset_t.append(trainset_t_)
            trainset_t = sum(trainset_t)

            self.train_loader_t = torch.utils.data.DataLoader(
                trainset_t,
                sampler=build_train_sampler(
                    trainset_t.train,
                    train_sampler_t,
                    batch_size=batch_size_train,
                    num_instances=num_instances,
                    num_cams=num_cams,
                    num_datasets=num_datasets,
                ),
                batch_size=batch_size_train,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=True,
            )

        print("=> Loading test (target) dataset")
        self.test_loader = {name: {"query": None, "gallery": None} for name in self.targets}
        self.test_dataset = {name: {"query": None, "gallery": None} for name in self.targets}

        for name in self.targets:
            queryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode="query",
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
            )
            self.test_loader[name]["query"] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False,
            )

            galleryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode="gallery",
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
            )
            self.test_loader[name]["gallery"] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False,
            )

            self.test_dataset[name]["query"] = queryset.query
            self.test_dataset[name]["gallery"] = galleryset.gallery

        print("\n")

