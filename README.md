# Flask Api with AI Inference

This Flask API is built for tooth segmentation, prosthesis detection, and dental caries identification, leveraging multiple machine learning models for accurate dental analysis. It provides a streamlined end-to-end solution, handling everything from image preprocessing to post-processing and inference generation.

# Key Features:
1. Tooth Segmentation: Identifies and segments different regions of the teeth within the provided image.
2. Prosthesis Detection: Detects and annotates any dental prosthetics.
3. Dental Caries Identification: Recognizes and marks areas affected by tooth decay.
   
# Workflow:
1. Image Input: The API accepts image URLs for processing.
2. Preprocessing: Images are resized and prepared for inference, ensuring compatibility with model input requirements.
3. Model Inference: Each image is fed into respective models for segmentation, prosthesis detection, or caries identification.
4. Post-Processing: Refinement of model outputs, including removal of redundant bounding boxes. Correlation of detected caries and prosthetics with specific tooth numbers based on the segmentation model’s output.

Output:
1. Processed Image: The resulting image is returned in bytecode format.
2. Annotations: A JSON file is generated with detailed annotations, including bounding boxes, segmentation data, and tooth numbers associated with caries or prosthesis detections.

This API provides a comprehensive and efficient solution for automated dental image analysis, delivering both visual and structured outputs that aid in diagnosis and treatment planning.

