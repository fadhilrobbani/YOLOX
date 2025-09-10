#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.output_dir = "/content/drive/MyDrive/Colab Notebooks/trained_models/cup-detection-yolox"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/VOCdevkit"
        self.train_ann = "train.txt"
        self.val_ann = "val.txt"
        
              # ---------- transform config ------------ #
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0

        self.num_classes = 2

        self.max_epoch = 50
        self.data_num_workers = 4
        self.eval_interval = 1

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import VOCDetection, TrainTransform
        from yolox.data.datasets.custom_voc_classes import CUSTOM_VOC_CLASSES
        from yolox.data.datasets.voc import AnnotationTransform

        return VOCDetection(
            data_dir=os.path.join(self.data_dir),
            image_sets=[('2007', 'trainval')],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            target_transform=AnnotationTransform(
                class_to_ind=dict(zip(CUSTOM_VOC_CLASSES, range(self.num_classes)))
            ),
            cache=cache,
            cache_type=cache_type,
            class_names=CUSTOM_VOC_CLASSES,
        )

    def get_eval_dataset(self, cache: bool = False, cache_type: str = "ram", **kwargs):
        from yolox.data import VOCDetection, ValTransform
        from yolox.data.datasets.custom_voc_classes import CUSTOM_VOC_CLASSES
        from yolox.data.datasets.voc import AnnotationTransform
        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            data_dir=os.path.join(self.data_dir),
            image_sets=[('2007', 'test')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            target_transform=AnnotationTransform(
                class_to_ind=dict(zip(CUSTOM_VOC_CLASSES, range(self.num_classes)))
            ),
            cache=cache, # Add cache argument
            cache_type=cache_type, # Add cache_type argument
            class_names=CUSTOM_VOC_CLASSES,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )