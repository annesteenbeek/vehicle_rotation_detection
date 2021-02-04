
"""
ModelHandler defines a base model handler.
"""

import torch
import sys, io, json, time, os.path
import numpy as np
import logging as logger
from PIL import Image

from detectron2.config import get_cfg

from detectron2.engine import DefaultPredictor
from detectron2.data import transforms as T
from detectron2.modeling import build_model

class RotatedPredictor(DefaultPredictor):
    def __init__(self, cfg):
        
        self.cfg = cfg.clone()  # cfg can be modified by model
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
        logger.info("Called predictor")
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

            logger.info("Predictor finished")
            return predictions



class VehicleHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        logger.info(" ------------------- logger init --------------")
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False
        self.predictor = None
        # self.model_file = "rn50_segmentation_model_final.pth"
        self.config_file = "config.yaml"  

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        logger.info("Initializing starting")

        try:
            self.manifest = context.manifest
            properties = context.system_properties

            model_dir = properties.get("model_dir")
            serialized_file = self.manifest['model']['serializedFile']
            model_pt_path = os.path.join(model_dir, serialized_file)

            assert os.path.exists(model_pt_path), "Model not found in %s" % model_pt_path
            assert os.path.exists(self.config_file), "Config not found in %s" % self.config_file

            if torch.cuda.is_available():
                self.device = torch.device("cuda:" + str(properties.get("gpu_id")))
            else:
                raise Exception("No cuda GPU available")

            cfg = get_cfg()
            cfg.merge_from_file(self.config_file)
            cfg.MODEL.WEIGHTS = model_pt_path

            # set the testing threshold for this model
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

            # self.predictor = RotatedPredictor(cfg)
            self.predictor = DefaultPredictor(cfg)

            self._batch_size = properties["batch_size"]

            logger.info("predictor built on initialize")
        except AssertionError as error:
            # Output expected AssertionErrors.
            logger.error(error)
        except:  # catch *all* exceptions
            e = sys.exc_info()[0]
            logger.error("Error: {}".format(e))

        self.initialized = True
        logger.info("initialized")

    def preprocess(self, batch):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))

        # Take the input data and pre-process it make it inference ready
        logger.info("pre-processing started for a batch of {}".format(len(batch)))

        images = []

        # batch is a list of requests
        for request in batch:
            for request_item in request:
                logger.info(request_item)
                
            # each item in the list is a dictionary with a single body key, get the body of the request
            request_body = request.get("body")

            # read the bytes of the image
            input = io.BytesIO(request_body)
            img = np.asarray(Image.open(input), dtype=np.uint8)

            images.append(img)

        logger.info("pre-processing finished for a batch of {}".format(len(batch)))

        return images

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """

        # Do some inference call to engine here and return output
        logger.info("inference started for a batch of {}".format(len(model_input)))

        outputs = []

        for image in model_input:
            output = self.predictor(image)
            outputs.append(output)

        logger.info("inference finished for a batch of {}".format(len(model_input)))

        return outputs

    def postprocess(self, inference_output):

        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        start_time = time.time()
        
        logger.info("post-processing started at {} for a batch of {}".format(start_time, len(inference_output)))
        
        responses = []

        for output in inference_output:

            # process predictions
            predictions = output["instances"].to("cpu")
            boxes = [box.numpy().tolist() for box in predictions.pred_boxes] if predictions.has("pred_boxes") else None
            scores = predictions.scores.numpy().tolist() if predictions.has("scores") else None
            classes = predictions.pred_classes.numpy().tolist() if predictions.has("pred_classes") else None
        
            responses_json={'classes': classes, 'scores': scores, "boxes": boxes}
            responses.append(json.dumps(responses_json))

        elapsed_time = time.time() - start_time
            
        logger.info("post-processing finished for a batch of {} in {}".format(len(inference_output), elapsed_time))

        return responses

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        logger.info("handling started")

        # process the data through our inference pipeline
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        output = self.postprocess(model_out)

        logger.info("handling finished")

        return output  

_service = VehicleHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)