"""Customized TorchReID engines used by TMerge training."""

from __future__ import annotations

import datetime
import time
from typing import Any


def _require_engine_types() -> tuple[Any, Any, Any]:
    try:
        from torch.utils.tensorboard import SummaryWriter
        from torchreid.engine.engine import Engine
        from torchreid.engine.image import ImageSoftmaxEngine, ImageTripletEngine
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TorchReID training requires tensorboard and torchreid extras. "
            'Install with: python -m pip install -e ".[reid]"'
        ) from exc
    return SummaryWriter, Engine, ImageSoftmaxEngine, ImageTripletEngine


SummaryWriter, Engine, ImageSoftmaxEngine, ImageTripletEngine = _require_engine_types()


class RefinedEngine(Engine):
    def run(
        self,
        save_dir="log",
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric="euclidean",
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False,
    ):
        if visrank and not test_only:
            raise ValueError("visrank can be set to True only if test_only=True")

        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
            )
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        print("=> Start training")

        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.train(print_freq=print_freq, fixbase_epoch=fixbase_epoch, open_layers=open_layers)
            if (self.epoch + 1) % 5 == 0:
                self.save_model(self.epoch, 0, save_dir)

        self.save_model(self.epoch, 0, save_dir)
        elapsed = str(datetime.timedelta(seconds=round(time.time() - time_start)))
        print(f"Elapsed {elapsed}")
        if self.writer is not None:
            self.writer.close()


class RefinedImageSoftmaxEngine(ImageSoftmaxEngine, RefinedEngine):
    pass


class RefinedImageTripletEngine(ImageTripletEngine, RefinedEngine):
    pass

