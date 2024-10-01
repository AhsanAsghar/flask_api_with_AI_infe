from flask import Blueprint, request, jsonify
from utils.caries_image_processing import load_image, draw_predictions
from utils.caries_detection import detect_caries, filter_caries_predictions
import base64
import cv2
from utils.caries_post_processing import (
    filter_caries_inside_teeth, annotate_caries_with_teeth
)


caries_bp = Blueprint('caries', __name__)
@caries_bp.route('/detect/caries', methods=['POST'])
def detect_caries_route():
    try:
        data = request.json
        image_path = data.get('image_url')
        final_predictions = data.get('final_predictions')
       


        if not image_path:
            return jsonify({"error": "Missing image_path"}), 400

        # Load the image
        image = load_image(image_path)
        #image = cv2.imread('/Users/apple/Downloads/Flask_APIs/04.JPG')

        # Detect caries using the caries model
        caries_predictions = detect_caries(image)

        # Filter caries predictions for overlaps
        filtered_caries_predictions = filter_caries_predictions(caries_predictions)

        # Filter caries that are inside teeth polygons
        filtered_caries_predictions = filter_caries_inside_teeth(filtered_caries_predictions, final_predictions)

        # Annotate the caries with corresponding tooth numbers
        annotated_caries, caries_tooth_mapping = annotate_caries_with_teeth(filtered_caries_predictions, final_predictions)

        # Draw the predictions on the image
        output_image = draw_predictions(image, teeth_predictions=final_predictions, caries_predictions=filtered_caries_predictions)


        _, buffer = cv2.imencode('.jpg', output_image)
        img_bytes = buffer.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # Prepare the response
        response = {
            "caries_tooth_mapping": caries_tooth_mapping,
            "resulted_image": img_base64
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500