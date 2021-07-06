from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from .datasets.coco import COCO
from .datasets.kitti import KITTI
from .datasets.coco_hp import COCOHP
from .datasets.mot import MOT
from .datasets.nuscenes import nuScenes
from .datasets.crowdhuman import CrowdHuman
from .datasets.kitti_tracking import KITTITracking
from .datasets.custom_dataset import CustomDataset
from .datasets.cell_tracking import CellTracking
from .datasets.cell_tracking1 import CellTracking1
from .datasets.cell_tracking_demo import CellTracking1 as CellTrackingDemo

dataset_factory = {
  'custom': CustomDataset,
  'coco': COCO,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'mot': MOT,
  'nuscenes': nuScenes,
  'crowdhuman': CrowdHuman,
  'kitti_tracking': KITTITracking,
  'cell_tracking':CellTracking,
  'cell_tracking1':CellTracking1,
  'cell_tracking_demo':CellTrackingDemo,
}


def get_dataset(dataset):
  return dataset_factory[dataset]