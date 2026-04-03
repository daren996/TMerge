"""Native TMerge entrypoint for TorchReID training workflows."""

from __future__ import annotations

import argparse
import os.path as osp
import sys
import time
from typing import Any


def _require_training_stack() -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torchreid
        from torchreid.data.datamanager import VideoDataManager
        from torchreid.data.datasets import register_image_dataset
        from torchreid.utils import (
            Logger,
            check_isfile,
            collect_env_info,
            compute_model_complexity,
            load_pretrained_weights,
            resume_from_checkpoint,
            set_random_seed,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TorchReID training requires optional dependencies. "
            'Install with: python -m pip install -e ".[reid]"'
        ) from exc
    return {
        "torch": torch,
        "nn": nn,
        "torchreid": torchreid,
        "VideoDataManager": VideoDataManager,
        "register_image_dataset": register_image_dataset,
        "Logger": Logger,
        "check_isfile": check_isfile,
        "collect_env_info": collect_env_info,
        "compute_model_complexity": compute_model_complexity,
        "load_pretrained_weights": load_pretrained_weights,
        "resume_from_checkpoint": resume_from_checkpoint,
        "set_random_seed": set_random_seed,
    }


def build_datamanager(cfg, register_image_dataset, VideoDataManager):
    from tmerge.training.reid_datamanager import ImageDataManager
    from tmerge.training.reid_dataset import KITTIDataset, MOTDataset
    from tmerge.training.reid_config import imagedata_kwargs, videodata_kwargs

    register_image_dataset("mot17det", MOTDataset)
    register_image_dataset("kitti", KITTIDataset)
    if cfg.data.type == "image":
        return ImageDataManager(**imagedata_kwargs(cfg))
    return VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    from tmerge.training.reid_engine import RefinedImageSoftmaxEngine, RefinedImageTripletEngine

    if cfg.data.type != "image":
        raise NotImplementedError("Only image-based ReID training is currently supported")
    if cfg.loss.name == "softmax":
        return RefinedImageSoftmaxEngine(
            datamanager,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            label_smooth=cfg.loss.softmax.label_smooth,
        )
    return RefinedImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        margin=cfg.loss.triplet.margin,
        weight_t=cfg.loss.triplet.weight_t,
        weight_x=cfg.loss.triplet.weight_x,
        scheduler=scheduler,
        use_gpu=cfg.use_gpu,
        label_smooth=cfg.loss.softmax.label_smooth,
    )


def reset_config(cfg, args) -> None:
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg) -> None:
    if cfg.loss.name == "triplet" and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("-s", "--sources", type=str, nargs="+", help="source datasets")
    parser.add_argument("-t", "--targets", type=str, nargs="+", help="target datasets")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation")
    parser.add_argument("--root", type=str, default="", help="path to data root")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command line",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    stack = _require_training_stack()
    from tmerge.training.reid_config import (
        engine_run_kwargs,
        get_default_config,
        lr_scheduler_kwargs,
        optimizer_kwargs,
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = get_default_config()
    cfg.use_gpu = stack["torch"].cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    stack["set_random_seed"](cfg.train.seed)
    check_cfg(cfg)

    log_name = ("test.log" if cfg.test.evaluate else "train.log") + time.strftime("-%Y-%m-%d-%H-%M-%S")
    sys.stdout = stack["Logger"](osp.join(cfg.data.save_dir, log_name))

    print(f"Show configuration\n{cfg}\n")
    print("Collecting env info ...")
    print(f"** System info **\n{stack['collect_env_info']()}\n")

    if cfg.use_gpu:
        stack["torch"].backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg, stack["register_image_dataset"], stack["VideoDataManager"])

    print(f"Building model: {cfg.model.name}")
    model = stack["torchreid"].models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
    )
    num_params, flops = stack["compute_model_complexity"](model, (1, 3, cfg.data.height, cfg.data.width))
    print(f"Model complexity: params={num_params:,} flops={flops:,}")

    if cfg.model.load_weights and stack["check_isfile"](cfg.model.load_weights):
        stack["load_pretrained_weights"](model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = stack["nn"].DataParallel(model).cuda()

    optimizer = stack["torchreid"].optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = stack["torchreid"].optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if cfg.model.resume and stack["check_isfile"](cfg.model.resume):
        cfg.train.start_epoch = stack["resume_from_checkpoint"](
            cfg.model.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    print(f"Building {cfg.loss.name}-engine for {cfg.data.type}-reid")
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))
    return 0

