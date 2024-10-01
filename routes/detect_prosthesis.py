from flask import Blueprint, request, jsonify
import base64
import cv2
from routes.final_prediction_variable import final_predictions1
from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from utils.prosthesis_helper import draw_predictions, filter_almost_same_position_predictions, annotate_prosthesis_with_teeth
from utils.prosthesis_image_processing import load_image, get_prosthesis_predictions

prosthesis_bp = Blueprint('prosthesis', __name__)
@prosthesis_bp.route('/detect/prosthesis', methods=['POST'])
def detect_prosthesis_route():
    try:
        data = request.json
        #image_url = '/Users/apple/Downloads/Prosthesis_Detection/test_images/12.JPG'
        #final_predictions = final_predictions1
        image_url = data.get('image_url')
        final_predictions = data.get('final_predictions')

        if not image_url:
            return jsonify({"error": "Missing image_path"}), 400
        

        image = load_image(image_url)

        # Get prosthesis predictions
        prosthesis_predictions = get_prosthesis_predictions(image, final_predictions)


        # Filter prosthesis predictions
        filtered_prosthesis_predictions = filter_almost_same_position_predictions(prosthesis_predictions)

        # Annotate prosthesis with corresponding tooth numbers
        annotated_prosthesis, prosthesis_tooth_mapping = annotate_prosthesis_with_teeth(filtered_prosthesis_predictions, final_predictions)

        # Draw predictions on the image
        output_image = draw_predictions(image,  prosthesis_predictions=filtered_prosthesis_predictions)


        cv2.imwrite('predicted_image.jpg', output_image)
        _, buffer = cv2.imencode('.jpg', output_image)
        img_bytes = buffer.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # Prepare the response
        response = {
            "caries_tooth_mapping": prosthesis_tooth_mapping,
            "resulted_image": img_base64
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500