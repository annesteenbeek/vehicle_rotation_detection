import cv2
import logging
import numpy as np
import torch.utils.data
import os
import copy
from detectron2.structures.rotated_boxes import RotatedBoxes
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import RotatedCOCOEvaluator, DatasetEvaluators
from detectron2.data import build_detection_train_loader, detection_utils, DatasetCatalog, get_detection_dataset_dicts
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.utils.logger import log_every_n
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.comm import get_world_size

import detectron2.data.build
import detectron2.data

from detectron2.data.common import ToIterableDataset

from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

# As of 0.3 the XYWHA_ABS box is not supported in the visualizer, this is fixed in master branch atm (19/11/20)
class RotatedVisualizer(Visualizer):
    def draw_dataset_dict(self, dic):
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYWHA_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                # "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                "%d: %.3f" % (index, a['bbox'][4]) + ("|crowd" if a.get("iscrowd", 0) else "")
                # "%.3f" % (a['bbox'][4]) + ("|crowd" if a.get("iscrowd", 0) else "")
                # ""
                for index, (i, a) in enumerate(zip(labels, annos))
            ]
            self.overlay_instances(labels=labels, boxes=boxes, masks=masks, keypoints=keypts)

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            sem_seg = cv2.imread(dic["sem_seg_file_name"], cv2.IMREAD_GRAYSCALE)
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, detectron_dataset_dict_list):
        super(MyIterableDataset).__init__()
        self.dataset_dict_list = detectron_dataset_dict_list

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        for dict in self.dataset_dict_list:
          yield rotated_mapper(dict)

class RotatedTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name):
      output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
      evaluators = [RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)]
      return DatasetEvaluators(evaluators)
      
  @classmethod
  def build_train_loader(cls, cfg):
    return build_detection_train_loader(cfg, mapper=rotated_mapper)

  # @classmethod
  # def build_train_loader(cls, cfg):
  #   # get dataset in detectron dict format
  #   train_dict = get_detection_dataset_dicts(
  #       cfg.DATASETS.TRAIN,
  #       filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
  #   )

  #   dataset = detectron2.data.DatasetFromList(train_dict, copy=False)
  #   sampler = TrainingSampler(len(train_dict))
  #   # add mapper to dataset
  #   dataset = detectron2.data.MapDataset(dataset, rotated_mapper)

  #   num_workers = cfg.DATALOADER.NUM_WORKERS
  #   total_batch_size = cfg.SOLVER.IMS_PER_BATCH

  #   world_size = get_world_size()
  #   batch_size = total_batch_size // world_size

  #   # wrap sampler to generate batches
  #   batch_sampler = torch.utils.data.sampler.BatchSampler(
  #     sampler, batch_size, drop_last=True
  #   )

  #   dataset = detectron2.data.common.ToIterableDataset(dataset, sampler)

  #   return torch.utils.data.DataLoader(
  #     dataset,
  #     num_workers=num_workers,
  #     batch_sampler=batch_sampler,
  #     collate_fn=detectron2.data.build.trivial_batch_collator,
  #     worker_init_fn=detectron2.data.build.worker_init_reset_seed,
  #   )



class RotatedPredictor(DefaultPredictor):
    def __init__(self, cfg):
        
        self.cfg = cfg.clone()  # cfg can be modified by model
        # self.model = trainer.model
        self.model = build_model(self.cfg)
        self.model.eval()

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.transform_gen.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

'''
https://github.com/facebookresearch/detectron2/issues/21#issuecomment-595522318

utils.transform_instance_annotations does not work for rotated boxes and 
you need a custom version using transform.apply_rotated_box

utils.annotations_to_instances needs to be replaced by utils.annotations_to_instances_rotated
'''

def rotated_transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
  # image_size = [H,W]
  if annotation["bbox_mode"] == BoxMode.XYWHA_ABS:
    annotation["bbox"] = transforms.apply_rotated_box(np.asarray([annotation["bbox"]]))[0]
  else:
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_box([bbox])[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

  return annotation

def convolve_iterator_mapper(dataset_dict):
    pass


def rotated_mapper(original_dataset_dict):
  # Implement a mapper, similar to the default DatasetMapper, but with our own customizations

  dataset_dict = copy.deepcopy(original_dataset_dict)  # it will be modified by code below
  original_gsd = dataset_dict["gsd"]
  target_gsd = np.random.uniform(0.09, 0.13) # randomize target gsd
  scale = original_gsd/target_gsd

  target_size = 400
  target_crop = int(target_size / scale)
  target_crop = (target_crop, target_crop)

  image_np = detection_utils.read_image(dataset_dict["file_name"], format="BGR")

  boxes = np.asarray([anno['bbox'] for anno in dataset_dict['annotations']])


  # select anno at random
  # draw random center

  # h, w = image_np.shape[:2]
  # rand_box = boxes[np.random.randint(len(boxes))]
  # ch, cw = rand_box[:2]
  # xmin = np.min()
  # xmax = np.max()
  # ymin = 3
  # ymax = 4

  # h0 = np.random.randint(min(h, ymin), min(h, ymax) + 1)
  # w0 = np.random.randint(min(w, xmin), min(w, xmax) + 1)
  # assert h >= target_crop[1] and w >= target_crop[0], "Shape computation has bugs."

  # crop = T.CropTransform(w0, h0, target_crop)

  # make sure random crop contains annotations
  i = 0
  while True:
    random_crop = T.RandomCrop('absolute', target_crop).get_transform(image_np)
    cropped_boxes = RotatedBoxes(random_crop.apply_coords(copy.deepcopy(boxes)))
    inside_ind = cropped_boxes.inside_box(target_crop)
    if 1 < sum(inside_ind) <= 100:
      break
    i += 1
    if i > 150:
      return None

  image, transforms = T.apply_transform_gens([
                                              random_crop,
                                              T.Resize((target_size, target_size)),
                                              ], image_np)
  dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

  annos = [
      rotated_transform_instance_annotations(obj, transforms, image.shape[:2]) 
      for obj in dataset_dict.pop("annotations")
      if obj.get("iscrowd", 0) == 0
  ]
  instances = detection_utils.annotations_to_instances_rotated(annos, image.shape[:2])
  instances = detection_utils.filter_empty_instances(instances)
  inside_ind = instances.gt_boxes.inside_box(image.shape[:2])
  instances = instances[inside_ind]

  assert (
      (instances.gt_boxes.tensor.numpy()[:,2] > 0).all().item()
  ), "width not > 0\n\n" + str(instances.gt_boxes.tensor.numpy())

  dataset_dict["instances"] = instances 
  return dataset_dict

def crop_rotated_box(transform, rotated_boxes):
 return transform.apply_coords(rotated_boxes)

T.CropTransform.register_type('rotated_box', crop_rotated_box)


if __name__ == "__main__":
  trainer = RotatedTrainer()