import cv2
import numpy as np
import os
import natsort
from ultralytics import YOLO

def load_yolo_annotations(image_path, img_width, img_height):
    """
    Load YOLO format annotations from a corresponding text file.
    
    Args:
    image_path (str): Path to the image file
    img_width (int): Width of the image
    img_height (int): Height of the image
    
    Returns:
    list: List of ground truth annotations
    """
    # Construct annotation file path
    annotation_path = image_path.replace('.jpg', '.txt').replace('.png', '.txt')
    
    ground_truth = []
    
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            for line in f:
                # Parse YOLO format: class_id x_center y_center width height
                parts = list(map(float, line.strip().split()))
                
                # Convert YOLO normalized coordinates to pixel coordinates
                class_id = int(parts[0])
                x_center = parts[1] * img_width
                y_center = parts[2] * img_height
                width = parts[3] * img_width
                height = parts[4] * img_height
                
                # Calculate corner coordinates
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                ground_truth.append({
                    'class_id': class_id,
                    'bbox': [x1, y1, x2, y2]
                })
    
    return ground_truth

def compare_predictions(ground_truth, predictions, iou_threshold=0.5):
    """
    Compare ground truth annotations with model predictions.
    
    Args:
    ground_truth (list): List of ground truth annotations
    predictions (list): List of model predictions
    iou_threshold (float): Intersection over Union threshold for matching
    
    Returns:
    list: A list of colors for each prediction
    """
    # If no ground truth is provided, color all predictions blue
    if not ground_truth:
        return [(255, 0, 0)] * len(predictions)  # Blue
    
    # Color coding list
    colors = []
    for pred in predictions.boxes:
        # Get prediction details
        pred_class = int(pred.cls[0])
        pred_bbox = list(map(int, pred.xyxy[0]))
        
        # Check if the prediction matches any ground truth
        matched = False
        for gt in ground_truth:
            # Check class matching
            if pred_class == gt['class_id']:
                # Calculate Intersection over Union (IoU)
                iou = calculate_iou(pred_bbox, gt['bbox'])
                
                # If IoU is above threshold, consider it a match
                if iou > iou_threshold:
                    matched = True
                    break
        
        # Color the box based on match
        colors.append((0, 255, 0) if matched else (0, 0, 255))  # Green if matched, red if not
    
    return colors

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
    box1 (list): First bounding box [x1, y1, x2, y2]
    box2 (list): Second bounding box [x1, y1, x2, y2]
    
    Returns:
    float: IoU value
    """
    # Coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Compute the area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute union
    union = box1_area + box2_area - intersection
    
    # Compute IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def process_frames_from_directory(model, frames_dir, output_path=None, output_video=True, display=True, fps=30):
    """
    Process frames from a directory and perform object detection.
    
    Args:
    model (YOLO): Loaded YOLO model
    frames_dir (str): Directory containing input frames
    output_path (str, optional): Path to save output (images or video)
    output_video (bool): Whether to save as video or individual frames
    display (bool): Whether to display frames in real-time
    fps (int): Frames per second for video output
    """
    # Get list of image files
    image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = natsort.natsorted(image_files)
    
    # Exit if no frames found
    if not image_files:
        print("No image files found in the directory.")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, image_files[0]))
    height, width = first_frame.shape[:2]
    
    # Initialize video writer if saving as video
    video_writer = None
    if output_video and output_path:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create output directory for individual frames if not outputting video
    if not output_video and output_path:
        os.makedirs(output_path, exist_ok=True)
    
    # Process each frame
    for image_filename in image_files:
        # Full path to the image
        image_path = os.path.join(frames_dir, image_filename)
        
        # Read the frame
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Could not read image: {image_path}")
            continue
        
        # Load ground truth annotations
        ground_truth = load_yolo_annotations(image_path, frame.shape[1], frame.shape[0])
        
        # Run YOLO inference on the frame
        results = model(frame)
        
        # Get predictions and their colors
        predictions = results[0]
        box_colors = compare_predictions(ground_truth, predictions)
        
        # Manually annotate the frame with custom colors
        annotated_frame = frame.copy()
        for i, box in enumerate(predictions.boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get the color for this prediction
            color = box_colors[i]
            
            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Optionally add class label and confidence
            label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display the annotated frame if requested
        if display:
            cv2.imshow("YOLO Inference", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Save annotated frame
        if output_video and video_writer:
            # Write frame to video
            video_writer.write(annotated_frame)
        elif not output_video and output_path:
            # Save as individual frames
            output_frame_path = os.path.join(output_path, f"annotated_{image_filename}")
            cv2.imwrite(output_frame_path, annotated_frame)
    
    # Release video writer
    if video_writer:
        video_writer.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Example usage
frames_directory = "path/to/your/frames/directory"

# Option 1: Save as video
output_video_path = "output_annotated_video.mp4"
process_frames_from_directory(
    model, 
    frames_dir=frames_directory, 
    output_path=output_video_path,
    output_video=True,  # Set to True to save as video
    display=True,       # Set to False if you don't want real-time display
    fps=30              # Adjust frames per second as needed
)

# Option 2: Save as individual frames
output_frames_directory = "path/to/output/frames/directory"
process_frames_from_directory(
    model, 
    frames_dir=frames_directory, 
    output_path=output_frames_directory,
    output_video=False,  # Set to False to save as individual frames
    display=True,        # Set to False if you don't want real-time display
    fps=30               # Not used when saving as frames
)
