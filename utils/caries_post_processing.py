import cv2
import numpy as np

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

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def filter_caries_inside_teeth(caries_predictions, teeth_predictions):
    """Filters caries that are inside teeth polygons."""
    filtered_caries = []
    for caries in caries_predictions:
        caries_center = (int(caries['x']), int(caries['y']))
        for tooth in teeth_predictions:
            tooth_polygon = [(pt['x'], pt['y']) for pt in tooth['points']]
            if point_in_polygon(caries_center, tooth_polygon):
                filtered_caries.append(caries)
                break
    return filtered_caries

def find_teeth_for_caries(caries, teeth_predictions):
    """Finds the tooth corresponding to a given caries prediction using IoU."""
    max_iou = 0
    assigned_tooth = None

    for tooth in teeth_predictions:
        tooth_box = (tooth['x'] - tooth['width'] / 2, tooth['y'] - tooth['height'] / 2, tooth['width'], tooth['height'])
        caries_box = (caries['x'] - caries['width'] / 2, caries['y'] - caries['height'] / 2, caries['width'], caries['height'])

        iou = calculate_iou(tooth_box, caries_box)
        if iou > max_iou:
            max_iou = iou
            assigned_tooth = tooth

    return assigned_tooth

def annotate_caries_with_teeth(caries_predictions, teeth_predictions):
    """Annotates caries with the corresponding tooth class."""
    annotated_caries = []
    caries_tooth_mapping = {}

    for caries in caries_predictions:
        assigned_tooth = find_teeth_for_caries(caries, teeth_predictions)
        if assigned_tooth:
            caries['assigned_tooth_class'] = assigned_tooth['class']
            tooth_class = assigned_tooth['class']
            caries_class = caries['class']
            if tooth_class in caries_tooth_mapping:
                caries_tooth_mapping[tooth_class].append(caries_class)
            else:
                caries_tooth_mapping[tooth_class] = [caries_class]

        annotated_caries.append(caries)

    return annotated_caries, caries_tooth_mapping
