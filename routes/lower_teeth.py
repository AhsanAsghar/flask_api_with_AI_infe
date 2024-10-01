from flask import Blueprint, request, jsonify
import json
import cv2
import numpy as np
import os
import sys
import base64
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import HTTPCallErrorError
import logging
import torch
from roboflow import Roboflow
import sys
import requests
from collections import OrderedDict
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import EventStorage
from detectron2.modeling import build_model
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from matplotlib import pyplot as plt
from PIL import Image

# Add the utils directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.post_processing_lower import (
    filter_predictions_by_confidence,
    remove_duplicate_predictions,
    correct_predictions,
    draw_predictions
)

cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
#cfg.DATASETS.TRAIN = ("teeth_dataset_train",)#Train dataset registered in a previous cell
#cfg.DATASETS.TEST = ("teeth_dataset_test",)#Test dataset registered in a previous cell
#cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.SOLVER.IMS_PER_BATCH = 2
#cfg.SOLVER.BASE_LR = 0.00025
#cfg.SOLVER.MAX_ITER = 10000 #We found that with a patience of 500, training will early stop before 10,000 iterations
#cfg.SOLVER.STEPS = []
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 600
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17 # 26 letters plus one super class
cfg.SOLVER.IMS_PER_BATCH=2
cfg.INPUT.MASK_FORMAT='bitmask'
cfg.MODEL.DEVICE = "cpu"
cfg.TEST.EVAL_PERIOD = 200 # Increase this number if you want to monitor validation performance during training
cfg.MODEL.WEIGHTS = 'models/model_lower_final.pth' # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17  ## One Background Class + Teeth
predictor = DefaultPredictor(cfg)
PATIENCE = 500 #Early stopping will occur after N iterations of no imporovement in total_loss


lower_teeth_bp = Blueprint('lower_teeth', __name__)
def read_image_from_url(image_url):
    
    # Fetch the image from the URL
    response = requests.get(image_url)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        raise Exception(f"Failed to fetch image. Status code: {response.status_code}")
        

@lower_teeth_bp.route('/detect/lower-teeth', methods=['POST'])
def detect_lower_teeth():
    ## API Request body starts here ##
    data = request.json
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    image = read_image_from_url(image_url)

    confidence_threshold = 0.25
    
    # Save the image temporarily
    temp_image_path = "temp_lower_image.jpg"
    cv2.imwrite(temp_image_path, image)


    class DummyFile(object):
        def write(self, x): pass
        def flush(self): pass


    original_stdout = sys.stdout
    sys.stdout = DummyFile()
    rf = Roboflow(api_key="Your_api_key")
    project = rf.workspace().project("your_project_name")
    model = project.version("version number").model
    sys.stdout = original_stdout
  
    # Define threshold
    confidence_threshold = 0.18

    response = model.predict(temp_image_path, confidence=18)
    result = response.json()

    

    # Filter predictions by confidence threshold
    filtered_predictions = filter_predictions_by_confidence(result['predictions'], confidence_threshold)

    # First, remove duplicate predictions
    unique_predictions = remove_duplicate_predictions(filtered_predictions, image.shape[1])

    # Then, apply the correction logic
    corrected_predictions = correct_predictions(unique_predictions, image.shape[1])

    # Filter out Predictions
    final_predictions = [pred for pred in corrected_predictions if (41 <= int(pred['class']) <= 48) or (31 <= int(pred['class']) <= 38)]

    # Draw predictions with visible labels
    output_image = draw_predictions(image, final_predictions)


    # Encode the image to base64
    _, buffer = cv2.imencode('.jpg', output_image)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    serializable_predictions = []
    for pred in final_predictions:
        serializable_pred = pred.copy()
        serializable_pred['points'] = [{'x': p['x'], 'y': p['y']} for p in pred['points']]
        serializable_predictions.append(serializable_pred)

  
  
    return jsonify({
        "predictions" : serializable_predictions,
        "result_image" : img_base64,
    })
