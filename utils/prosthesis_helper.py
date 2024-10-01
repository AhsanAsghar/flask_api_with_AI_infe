import cv2
import numpy as np

def draw_predictions(image, teeth_predictions=None, prosthesis_predictions=None):
    overlay = image.copy()
    output = image.copy()

    if teeth_predictions:
        for pred in teeth_predictions:
            points = np.array([(pt['x'], pt['y']) for pt in pred['points']], np.int32)
            points = points.reshape((-1, 1, 2))
            color = (255, 0, 0)
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)

    if prosthesis_predictions:
        for pred in prosthesis_predictions:
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
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def filter_prosthesis_inside_polygons(prosthesis_predictions, teeth_predictions):
    filtered_prosthesis = []
    for prosthesis in prosthesis_predictions:
        prosthesis_box = (
            (prosthesis['x'] - prosthesis['width'] / 2, prosthesis['y'] - prosthesis['height'] / 2),
            (prosthesis['x'] + prosthesis['width'] / 2, prosthesis['y'] + prosthesis['height'] / 2)
        )
        # Check if the center of the prosthesis box is inside any polygon
        prosthesis_center = (prosthesis['x'], prosthesis['y'])
        for tooth in teeth_predictions:
            polygon = [(pt['x'], pt['y']) for pt in tooth['points']]
            if is_point_in_polygon(prosthesis_center, polygon):
                filtered_prosthesis.append(prosthesis)
                break
    return filtered_prosthesis

def rectangles_almost_same_position(r1, r2, threshold=30):
    return (abs(r1['x'] - r2['x']) < threshold and
            abs(r1['y'] - r2['y']) < threshold and
            abs(r1['width'] - r2['width']) < threshold and
            abs(r1['height'] - r2['height']) < threshold)


def filter_almost_same_position_predictions(predictions, threshold=30):
    result = []
    used = [False] * len(predictions)

    for i, pred1 in enumerate(predictions):
        if used[i]:
            continue

        max_confidence_pred = pred1

        for j, pred2 in enumerate(predictions):
            if i != j and not used[j] and rectangles_almost_same_position(pred1, pred2, threshold):
                if pred2['confidence'] > max_confidence_pred['confidence']:
                    max_confidence_pred = pred2
                used[j] = True

        result.append(max_confidence_pred)
        used[i] = True

    return result


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou



def find_teeth_for_prosthesis(prosthesis, teeth_predictions):
    max_iou = 0
    assigned_tooth = None

    for tooth in teeth_predictions:
        tooth_box = (tooth['x'] - tooth['width'] / 2, tooth['y'] - tooth['height'] / 2, tooth['width'], tooth['height'])
        prosthesis_box = (prosthesis['x'] - prosthesis['width'] / 2, prosthesis['y'] - prosthesis['height'] / 2, prosthesis['width'], prosthesis['height'])

        iou = calculate_iou(tooth_box, prosthesis_box)
        if iou > max_iou:
            max_iou = iou
            assigned_tooth = tooth

    return assigned_tooth


def annotate_prosthesis_with_teeth(prosthesis_predictions, teeth_predictions):
    annotated_prosthesis = []
    prosthesis_tooth_mapping = {}

    for prosthesis in prosthesis_predictions:
        assigned_tooth = find_teeth_for_prosthesis(prosthesis, teeth_predictions)
        if assigned_tooth:
            prosthesis['assigned_tooth_class'] = assigned_tooth['class']
            tooth_class = assigned_tooth['class']
            prosthesis_class = prosthesis['class']
            if tooth_class in prosthesis_tooth_mapping:
                prosthesis_tooth_mapping[tooth_class].append(prosthesis_class)
            else:
                prosthesis_tooth_mapping[tooth_class] = [prosthesis_class]

        annotated_prosthesis.append(prosthesis)

    return annotated_prosthesis, prosthesis_tooth_mapping
