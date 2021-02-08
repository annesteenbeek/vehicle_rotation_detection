#!/usr/bin/python

import os
import torch

from detectron2.config import *
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import launch

from rotated_detectron_utils import RotatedTrainer
from dota_tools import dota2detectron



torch.cuda.set_device(1)

dataset_path = '/data2/anne/UIVER/datasets/DOTA'
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'val')

# train_path = '/data2/anne/UIVER/datasets/DOTA/dotadevkit/example'
# class_name_list = ["small-vehicle", "large-vehicle"]
class_name_list = ["small-vehicle"]

DatasetCatalog.clear()
MetadataCatalog.clear()

DatasetCatalog.register("Train", lambda: dota2detectron(train_path, use_cache=False, whitelist=class_name_list))
DatasetCatalog.register("Test", lambda: dota2detectron(test_path, use_cache=False, whitelist=class_name_list))
MetadataCatalog.get("Train").set(thing_classes=class_name_list)
MetadataCatalog.get("Test").set(thing_classes=class_name_list)


cfg = get_cfg()

cfg.OUTPUT_DIR = os.path.join(dataset_path, 'output_rcnn_small_vehicle_2')

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo

cfg.DATASETS.TRAIN = (["Train"])
cfg.DATASETS.TEST = (["Test"])

cfg.MODEL.MASK_ON=False
cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (10,10,5,5,1)
cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90,-60,-30,0,30,60,90]]
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 
cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_name_list)
cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10,10,5,5,1)
cfg.MODEL.ROI_BOX_HEAD.NUM_CONV=4
cfg.MODEL.ROI_MASK_HEAD.NUM_CONV=8
cfg.SOLVER.IMS_PER_BATCH = 20 # reduce for memory
cfg.SOLVER.CHECKPOINT_PERIOD=1500
cfg.SOLVER.BASE_LR = 0.005
cfg.SOLVER.GAMMA=0.5
cfg.SOLVER.STEPS=[1000,2000,4000,8000, 12000]
cfg.SOLVER.MAX_ITER=28000 # 14000

# epoch is MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES


cfg.DATALOADER.NUM_WORKERS = 16
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True 
cfg.DATALOADER.SAMPLER_TRAIN= "RepeatFactorTrainingSampler"
cfg.DATALOADER.REPEAT_THRESHOLD=0.01
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)#lets just check our output dir exists
cfg.MODEL.BACKBONE.FREEZE_AT=6



def main():
  trainer = RotatedTrainer(cfg) 
  trainer.resume_or_load(resume=True)
  return trainer.train()


if __name__ == "__main__":
  launch(main, 2, dist_url='auto')
  # main()