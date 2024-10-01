import cv2
import requests
import numpy as np

def read_image_from_url(image_url):
    
    # Fetch the image from the URL
    response = requests.get(image_url)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        raise Exception(f"Failed to fetch image. Status code: {response.status_code}")


def load_image(image_path):
    """Loads an image from the specified path."""
    image = read_image_from_url(image_path)
    #image = cv2.imread(image_path)
    return image

def draw_predictions(image, teeth_predictions=None, caries_predictions=None):
    overlay = image.copy()
    output = image.copy()
    if caries_predictions:
        for pred in caries_predictions:
            x = int(pred['x'])
            y = int(pred['y'])
            w = int(pred['width'])
            h = int(pred['height'])
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (x + w // 2, y + h // 2)

            color = (0, 255, 0)  # Green for caries
            cv2.rectangle(overlay, top_left, bottom_right, color, 2)

            label = f"{pred['class']}"
            cv2.putText(overlay, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

    alpha = 0.5
    output_image = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    cv2.imwrite('predicted_image.jpg', output_image)
    return output_image
