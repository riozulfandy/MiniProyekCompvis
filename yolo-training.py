import torch
from ultralytics import YOLO
import os

# Configuration
DATA_YAML = '/kaggle/working/Mario-Detection-5/data.yaml'  # Path to your dataset configuration
PRETRAINED_WEIGHTS = 'yolo11s.pt'     # Pretrained weights (small model)
NUM_EPOCHS = 60
BATCH_SIZE = 16
IMAGE_SIZE = 1280

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

def train_model():
    """
    Train YOLO model
    """
    # Initialize model
    model = YOLO(PRETRAINED_WEIGHTS)
    
    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=NUM_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return model

def validate_model(model):
    """
    Validate model on validation dataset
    """
    # Validation results
    validation_metrics = model.val(
        data=DATA_YAML,
        split='val'  # Use validation split
    )
    
    # Print key validation metrics
    print("Validation Metrics:")
    print(f"mAP50: {validation_metrics.results_dict['metrics/mAP50(B)']}")
    print(f"mAP50-95: {validation_metrics.results_dict['metrics/mAP50-95(B)']}")
    
    return validation_metrics

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
    # Prepare dataset (ensure data.yaml is correctly set up)
    prepare_dataset(DATA_YAML, '/kaggle/working/Mario-Detection-5/train/images', '/kaggle/working/Mario-Detection-5/valid/images', '/kaggle/working/Mario-Detection-5/test/images', 1, "['mario']")
    
    # Train the model
    trained_model = train_model()
    
    # Validate on validation dataset
    validation_results = validate_model(trained_model)

    print("Validation Results:")
    print(validation_results)
    
    # Test on test dataset
    test_results = test_model(trained_model)

    print("Test Results:")
    print(test_results)
    
    # Optional: Export the best model
    trained_model.export(format='onnx')

if __name__ == '__main__':
    main()
