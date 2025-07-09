import os
import random
from typing import List

import albumentations as A
import cv2

from src import constants as c


def get_random_transform():
    # Random shear values
    shear_x = (random.uniform(-30, -10), random.uniform(10, 30))
    shear_y = (random.uniform(-30, -10), random.uniform(10, 30))

    # Random color jitter values
    brightness = random.uniform(0.1, 0.5)
    contrast = random.uniform(0.1, 0.5)
    saturation = random.uniform(0.1, 0.5)
    hue = random.uniform(0.1, 0.3)

    # Random blur limit
    blur_limit = (3, random.choice([5, 7, 9]))

    transform = A.Compose(
        [
            A.Affine(shear={"x": shear_x, "y": shear_y}, p=0.5),
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=0.8,
            ),
            A.GaussNoise(p=0.5),
            A.MotionBlur(blur_limit=blur_limit, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=100, p=0.5),
            A.RandomScale(scale_limit=0.3, p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    return transform, transform.__hash__()


def augment_images(directory: str, n_augmentations: int, replace: bool = True) -> None:
    assert os.path.isdir(directory), "Directory does not exist for augmentation!"
    # Get paths from original folder
    images_dir = os.path.join(directory, "images")
    labels_dir = os.path.join(directory, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    image_paths = _filepaths_to_list(images_dir)
    label_paths = _filepaths_to_list(labels_dir)

    # Create augmented folder (should not already exist!)
    out_dir = os.path.join(os.path.dirname(directory), f"{directory}_augmented")
    images_out_dir = os.path.join(out_dir, "images")
    labels_out_dir = os.path.join(out_dir, "labels")
    os.makedirs(images_out_dir)
    os.makedirs(labels_out_dir)

    # Load images
    for image_path, label_path in zip(image_paths, label_paths):
        image, bboxes, class_labels = c.get_image_and_bboxes(image_path, label_path)

        # Apply transformation per image
        for i in range(n_augmentations):
            transform, transform_hash = get_random_transform()
            transformed = transform(
                image=image, bboxes=bboxes, class_labels=class_labels
            )

            aug_image = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_labels = transformed["class_labels"]

            # Save transformed image
            out_image_path = os.path.join(
                images_out_dir, "aug_" + str(transform_hash) + ".jpg"
            )
            out_label_path = os.path.join(
                labels_out_dir, "aug_" + str(transform_hash) + ".txt"
            )

            cv2.imwrite(out_image_path, aug_image)
            # Save updated label file
            with open(out_label_path, "w") as f:
                for class_id, bbox in zip(aug_labels, aug_bboxes):
                    line = f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}\n"
                    f.write(line)

    if replace:
        assert not os.path.isdir(
            f"{directory}_original"
        ), "Found other original folder! Did not replace original folder!"
        os.rename(directory, f"{directory}_original")
        os.rename(f"{directory}_augmented", directory)


def _filepaths_to_list(directory: str) -> List[str]:
    file_paths = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path):
            file_paths.append(full_path)

    return file_paths
