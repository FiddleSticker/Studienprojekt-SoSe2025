import os
import random

import albumentations as A
import cv2
import constants as c


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


def augment_image(n_augmentations: int) -> None:
    # Create target folders
    images_dir = os.path.join(c.PROJECT_DIR, "data/train/images")
    labels_dir = os.path.join(c.PROJECT_DIR, "data/train/labels")
    target_images_out_dir = os.path.join(c.PROJECT_DIR, "data/aug/images")
    target_labels_out_dir = os.path.join(c.PROJECT_DIR, "data/aug/labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(target_images_out_dir, exist_ok=True)
    os.makedirs(target_labels_out_dir, exist_ok=True)

    image_paths = []
    for entry in os.listdir(images_dir):
        full_path = os.path.join(images_dir, entry)
        if os.path.isfile(full_path):
            image_paths.append(full_path)

    label_paths = []
    for entry in os.listdir(labels_dir):
        full_path = os.path.join(labels_dir, entry)
        if os.path.isfile(full_path):
            label_paths.append(full_path)

    # Load image
    for img in range(len(image_paths)):
        image, bboxes, class_labels = c.get_image_and_bboxes(
            image_paths[img], label_paths[img]
        )
        for i in range(n_augmentations):

            transform, transform_hash = get_random_transform()

            # Apply transformation
            transformed = transform(
                image=image, bboxes=bboxes, class_labels=class_labels
            )

            aug_image = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_labels = transformed["class_labels"]

            # Save transformed image
            out_image_path = os.path.join(
                target_images_out_dir, "aug_" + str(transform_hash) + ".jpg"
            )
            out_label_path = os.path.join(
                target_labels_out_dir, "aug_" + str(transform_hash) + ".txt"
            )

            cv2.imwrite(out_image_path, aug_image)

            # Save updated label file
            with open(out_label_path, "w") as f:
                for class_id, bbox in zip(aug_labels, aug_bboxes):
                    line = f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}\n"
                    f.write(line)
