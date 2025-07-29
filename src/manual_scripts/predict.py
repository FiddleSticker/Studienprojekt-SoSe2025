"""Helper script to predict bounding boxes on images"""

import os
from ultralytics import YOLO

from src import constants as c


def predict(model_path: str, images_path: str) -> None:
    """Predicts on images from a given directory

    Args:
        model_path (str): Path to the trained model
        images_path (str): Path to the images directory
    """

    # Load a model
    model = YOLO(model_path)

    images = [os.path.join(images_path, file) for file in os.listdir(images_path)]

    # Run batched inference on a list of images
    results = model(images, stream=True)  # return a generator of Results objects

    predict_path = c.unique_dir(os.path.join(c.DETECT_DIR, "predict"))
    result_images_path = os.path.join(predict_path, "images")
    result_labels_path = os.path.join(predict_path, "labels")
    os.makedirs(predict_path)
    os.makedirs(result_images_path)
    os.makedirs(result_labels_path)

    # Process results generator
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        # result.show()  # display to screen
        result.save(
            filename=os.path.join(result_images_path, os.path.split(result.path)[-1])
        )
        result.save_txt(
            txt_file=os.path.join(
                result_labels_path,
                f"{os.path.splitext(os.path.split(result.path)[-1])[0]}.txt",
            )
        )


# if __name__ == "__main__":
#     _model_path = os.path.join(c.PROJECT_DIR, r"runs\detect\train10\weights\best.pt")
#     test_set_images_dir = os.path.join(c.DATASET_LOCATION, "test", "images")
#
#     predict(model_path=_model_path, images_path=test_set_images_dir)
