import torch
from ultralytics import YOLO
import os

# Configuration
DATA_YAML = '/kaggle/working/MarioHTV.v1i.yolov11/data.yaml'  # Path to your dataset configuration
TRAINED_WEIGHTS = 'best.pt'     # Pretrained weights (small model)

def prepare_dataset(yaml_path, train_images, val_images, test_images, num_classes, class_names):
    """
    Prepare dataset configuration file (data.yaml)
    
    Example structure:
    ```
    train: /path/to/train/images
    val: /path/to/validation/images
    test: /path/to/test/images
    
    nc: 80  # number of classes
    names: ['class1', 'class2', ...]  # class names
    ```
    """
    # Prepare data.yaml
    with open(yaml_path, 'w') as f:
        f.write(f"train: {train_images}\n")
        f.write(f"val: {val_images}\n")
        f.write(f"test: {test_images}\n")
        f.write(f"nc: {num_classes}\n")
        f.write(f"names: {class_names}\n")
    
    print(f"Dataset configuration saved to {yaml_path}")

def test_model(model):
    """
    Evaluate model on test dataset
    """
    # Test results
    test_metrics = model.val(
        data=DATA_YAML,
        split='test'  # Use test split
    )
    
    # Print key test metrics
    print("Test Metrics:")
    print(f"mAP50: {test_metrics.results_dict['metrics/mAP50(B)']}")
    print(f"mAP50-95: {test_metrics.results_dict['metrics/mAP50-95(B)']}")
    
    return test_metrics

def main():
    # Load the YOLO model
    model = YOLO(TRAINED_WEIGHTS)

    prepare_dataset(DATA_YAML, '/kaggle/working/MarioHTV.v1i.yolov11/train', '/kaggle/working/MarioHTV.v1i.yolov11/train', '/kaggle/working/MarioHTV.v1i.yolov11/train', 1, "['mario']")
    
    # Test the model
    test_metrics = test_model(model)
    
if __name__ == "__main__":
    main()
