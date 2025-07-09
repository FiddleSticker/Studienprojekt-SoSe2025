import os
import shutil
from roboflow import Roboflow
from ultralytics import YOLO

from src import constants as c
from src.augment import augment_images

if __name__ == "__main__":
    # Download dataset from Roboflow
    rf = Roboflow(api_key="f2Zm3PDjLvH2GrAMKRuE")
    project = rf.workspace("test-p4ko4").project("test2-xzapr")
    version = project.version(8)
    dataset = version.download("yolov9")

    # Moving dataset to fixed location and shuffling
    if os.path.isdir(c.DATASET_LOCATION):
        shutil.rmtree(c.DATASET_LOCATION)
    os.rename(dataset.location, c.DATASET_LOCATION)
    c.shuffle_sets(c.DATASET_LOCATION, 80, 16, 16)
    augment_images(os.path.join(c.DATASET_LOCATION, "train"), n_augmentations=2)

    # Training model
    model = YOLO("yolov9t.pt")
    model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=1000,
        imgsz=640,
    )
