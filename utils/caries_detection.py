from roboflow import Roboflow
import sys

class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass


original_stdout = sys.stdout
sys.stdout = DummyFile()
rf = Roboflow(api_key="your api key")
project = rf.workspace().project("project name")
model = project.version("project number").model
sys.stdout = original_stdout


def detect_caries(image):
    """Detects caries in the provided image."""
    response = model.predict(image, confidence=18, overlap=50)
    return response.json()['predictions']

def rectangles_almost_same_position(r1, r2, threshold=30):
    return (abs(r1['x'] - r2['x']) < threshold and
            abs(r1['y'] - r2['y']) < threshold and
            abs(r1['width'] - r2['width']) < threshold and
            abs(r1['height'] - r2['height']) < threshold)

def filter_caries_predictions(predictions, threshold=30):
    """Filters predictions for caries that are close to each other."""
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
