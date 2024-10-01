import cv2
import numpy as np
from roboflow import Roboflow
import requests
from utils.prosthesis_helper import filter_prosthesis_inside_polygons

def load_image(image_url):
    image_path = image_url
    image = cv2.imread(image_path)
    return image

def get_prosthesis_predictions(image, teeth_predictions):
    rf = Roboflow(api_key="1XjdbDPyGKPxbLAQZwXP")
    project = rf.workspace().project("prosthesis-dentalcaries")
    model = project.version("2").model
    response = model.predict(image, confidence=18, overlap=50)
    prosthesis_predictions = response.json()['predictions']
    
    # Filter only prosthesis that fall inside the polygons
    filtered_prosthesis_predictions = filter_prosthesis_inside_polygons(prosthesis_predictions, teeth_predictions)
    
    return filtered_prosthesis_predictions