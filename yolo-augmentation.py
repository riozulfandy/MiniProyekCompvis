import albumentations as A
import cv2
import numpy as np
import os

class YOLOAugmentation:
    def __init__(self, img_dir, label_dir, num_augmentations=5):
        """
        Initialize YOLO dataset augmentation class.
        
        :param img_dir: Directory containing original images.
        :param label_dir: Directory containing YOLO format label files.
        :param num_augmentations: Number of augmentations to generate per image.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.num_augmentations = num_augmentations

        # Define augmentation pipeline
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.Affine(shear=(-5, 5), p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0, val_shift_limit=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.GaussNoise(var_limit=(0.001, 0.005), p=0.5),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2,
            ),
        )

    def read_yolo_labels(self, label_path):
        """
        Read YOLO labels from a file.
        """
        with open(label_path, "r") as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]
        return labels

    def write_yolo_labels(self, label_path, labels):
        """
        Write YOLO labels to a file.
        """
        with open(label_path, "w") as f:
            for label in labels:
                f.write(" ".join(map(str, label)) + "\n")

    def clip_bounding_boxes(self, bboxes):
        """
        Clip bounding box coordinates to ensure they stay in the range [0.0, 1.0].
        """
        clipped_bboxes = []
        for bbox in bboxes:
            x_center, y_center, width, height = bbox
            x_min = max(0.0, x_center - width / 2)
            y_min = max(0.0, y_center - height / 2)
            x_max = min(1.0, x_center + width / 2)
            y_max = min(1.0, y_center + height / 2)

            new_x_center = (x_min + x_max) / 2
            new_y_center = (y_min + y_max) / 2
            new_width = x_max - x_min
            new_height = y_max - y_min

            if new_width > 0 and new_height > 0:
                clipped_bboxes.append([new_x_center, new_y_center, new_width, new_height])
        return clipped_bboxes

    def augment_dataset(self):
        """
        Perform dataset augmentation.
        """
        for img_filename in os.listdir(self.img_dir):
            if img_filename.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(self.img_dir, img_filename)
                label_filename = os.path.splitext(img_filename)[0] + ".txt"
                label_path = os.path.join(self.label_dir, label_filename)

                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if not os.path.exists(label_path):
                    print(f"Label file for {img_filename} not found. Skipping.")
                    continue

                labels = self.read_yolo_labels(label_path)
                class_labels = [int(label[0]) for label in labels]
                bboxes = [label[1:] for label in labels]

                # Pre-clip bounding boxes
                bboxes = self.clip_bounding_boxes(bboxes)

                for i in range(self.num_augmentations):
                    try:
                        augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

                        # Post-clip bounding boxes
                        clipped_bboxes = self.clip_bounding_boxes(augmented["bboxes"])

                        if not clipped_bboxes:
                            print(f"No valid bounding boxes for {img_filename} augmentation {i}. Skipping.")
                            continue

                        aug_image = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
                        aug_filename = f"{os.path.splitext(img_filename)[0]}_aug_{i}.jpg"
                        aug_img_path = os.path.join(self.img_dir, aug_filename)

                        cv2.imwrite(aug_img_path, aug_image)

                        aug_labels = [[class_label] + bbox for class_label, bbox in zip(augmented["class_labels"], clipped_bboxes)]
                        aug_label_path = os.path.join(self.label_dir, f"{os.path.splitext(label_filename)[0]}_aug_{i}.txt")
                        self.write_yolo_labels(aug_label_path, aug_labels)

                    except ValueError as e:
                        print(f"Augmentation error for {img_filename}, augmentation {i}: {e}")
                        continue


# Example usage
if __name__ == "__main__":
    augmenter = YOLOAugmentation(
        img_dir="/kaggle/working/Mario-Detection-5/train/images",
        label_dir="/kaggle/working/Mario-Detection-5/train/labels",
        num_augmentations=5,
    )
    augmenter.augment_dataset()
