import torch
from ultralytics import YOLO
import time
import numpy as np

def measure_inference_speed(model_path, test_images, num_warmup=10, num_iterations=100):
    """
    Measure inference speed of a YOLO model
    
    Args:
    - model_path (str): Path to the trained YOLO model
    - test_images (list): List of image paths to use for speed testing
    - num_warmup (int): Number of warmup iterations to stabilize GPU
    - num_iterations (int): Number of iterations to measure speed
    
    Returns:
    - Dictionary with various speed metrics
    """
    # Load the model
    model = YOLO(model_path)
    
    # Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Warmup iterations (not measured)
    for _ in range(num_warmup):
        for img in test_images:
            _ = model(img, verbose=False)
    
    # Timing variables
    total_time = 0
    inference_times = []
    
    # Measure inference time
    for _ in range(num_iterations):
        start_time = time.time()
        
        # Run inference on each image
        for img in test_images:
            results = model(img, verbose=False)
        
        end_time = time.time()
        
        # Calculate time for this iteration
        iteration_time = end_time - start_time
        total_time += iteration_time
        inference_times.append(iteration_time)
    
    # Calculate speed metrics
    speed_metrics = {
        'avg_total_inference_time_ms': np.mean(inference_times) * 1000,
        'std_total_inference_time_ms': np.std(inference_times) * 1000,
        'fps': 1 / (np.mean(inference_times) / len(test_images)),
        'avg_image_inference_time_ms': (np.mean(inference_times) * 1000) / len(test_images),
        'device': str(device)
    }
    
    return speed_metrics

def detailed_speed_profiling(model_path, test_images):
    """
    Perform detailed speed profiling using Ultralytics built-in methods
    """
    model = YOLO(model_path)
    
    # Ultralytics provides built-in speed profiling
    speed = model(test_images[0], verbose=False, speed=True)
    
    return {
        'preprocess_time_ms': speed['preprocess'],
        'inference_time_ms': speed['inference'],
        'postprocess_time_ms': speed['postprocess']
    }

def main():
    # Example usage
    model_path = 'path/to/best.pt'
    test_images = [
        'image1.jpg', 
        'image2.jpg', 
        'image3.jpg'
    ]
    
    # Measure inference speed
    speed_metrics = measure_inference_speed(model_path, test_images)
    print("Speed Metrics:")
    for metric, value in speed_metrics.items():
        print(f"{metric}: {value}")
    
    # Detailed speed profiling
    detailed_speed = detailed_speed_profiling(model_path, test_images)
    print("\nDetailed Speed Breakdown:")
    for stage, time_ms in detailed_speed.items():
        print(f"{stage}: {time_ms} ms")

if __name__ == '__main__':
    main()
