import os
import random

import matplotlib.image as mpimg

SRC_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(SRC_DIR)

CLASS_DICT = {0: ("Bauteil", "b"), 1: ("Schaden", "r")}


# some common functions
def get_bboxes(label_path: str) -> tuple:
    bboxes = []
    class_labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, w, h = map(float, parts)
                bboxes.append((x_center, y_center, w, h))
                class_labels.append(int(class_id))

    return bboxes, class_labels


def get_image_and_bboxes(image_path: str, label_path: str) -> tuple:
    # Load image
    image = mpimg.imread(image_path)

    # Read bboxes from label_path
    bboxes, class_labels = get_bboxes(label_path)

    return image, bboxes, class_labels


def _flatten_dir(dir_path: str) -> None:
    valid_extensions = [".jpg", ".jpeg", ".png", ".txt", ".yaml"]
    for dir_, subdirs, files in os.walk(dir_path, topdown=False):
        for filename in files:
            if os.path.splitext(filename)[-1] in valid_extensions:
                os.rename(
                    os.path.join(dir_, filename), os.path.join(dir_path, filename)
                )
            else:
                os.remove(os.path.join(dir_, filename))  # removes cache files
        for subdir in subdirs:
            os.rmdir(os.path.join(dir_, subdir))


def _divide_sets(
    dir_path: str,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    valid_ratio: float = 0.1,
) -> None:
    # Getting images and shuffling
    image_files = [
        f for f in os.listdir(dir_path) if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(image_files)

    # Adjusting ratios
    total = train_ratio + test_ratio + valid_ratio
    train_ratio = train_ratio / total
    test_ratio = test_ratio / total

    # Splitting sets
    total = len(image_files)
    train_end = int(total * train_ratio)
    test_end = train_end + int(total * test_ratio)

    train_image_files = image_files[:train_end]
    test_image_files = image_files[train_end:test_end]
    valid_image_files = image_files[test_end:]

    # Creating dirs for sets
    for set_path, set_files in zip(
        ["train", "test", "valid"],
        [train_image_files, test_image_files, valid_image_files],
    ):
        set_dir = os.path.join(dir_path, set_path)
        img_dir = os.path.join(set_dir, "images")
        label_dir = os.path.join(set_dir, "labels")
        os.makedirs(set_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for image_file in set_files:
            label_file = f"{os.path.splitext(image_file)[0]}.txt"
            os.rename(
                os.path.join(dir_path, label_file), os.path.join(label_dir, label_file)
            )
            os.rename(
                os.path.join(dir_path, image_file), os.path.join(img_dir, image_file)
            )


def shuffle_sets(
    dir_path: str,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    valid_ratio: float = 0.1,
) -> None:
    _flatten_dir(dir_path)
    _divide_sets(dir_path, train_ratio, test_ratio, valid_ratio)
