"""Helper script to plot images together with bounding boxes"""

import os
import matplotlib.pyplot as plt

from src import constants as c


def plot_image_with_bboxes(image_path: str, label_path: str) -> None:
    """
    Plots an image together with bounding boxes

    Args:
        image_path (str): Path to the image
        label_path(str) : Path to the labels
    """

    image, bboxes, class_labels = c.get_image_and_bboxes(image_path, label_path)
    height, width = image.shape[:2]

    # Plot image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw bboxes into image
    for (x_center, y_center, w, h), class_id in zip(bboxes, class_labels):
        class_name, class_color = c.CLASS_DICT[class_id]

        x_min = (x_center - w / 2) * width
        y_min = (y_center - h / 2) * height
        x_max = (x_center + w / 2) * width
        y_max = (y_center + h / 2) * height

        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor=class_color,
            facecolor="none",
        )
        ax.add_patch(rect)
        label = f"Klasse = {class_name}"
        ax.text(
            x_min,
            y_min - 5,
            label,
            color=class_color,
            fontsize=10,
            backgroundcolor="black",
        )

    plt.axis("off")
    plt.show()


# if __name__ == "__main__":
#     # display images in a directory for debug purposes
#     dataset_dir = c.DATASET_LOCATION
#     test_set_dir = os.path.join(dataset_dir, "test")
#     test_set_images_dir = os.path.join(test_set_dir, "images")
#     test_set_labels_dir = os.path.join(test_set_dir, "labels")
#
#     for file in os.listdir(test_set_images_dir):
#         file_name = os.path.splitext(file)[0]
#
#         image_path = os.path.join(test_set_images_dir, file)
#         label_path = os.path.join(test_set_labels_dir, f"{file_name}.txt")
#
#         plot_image_with_bboxes(image_path, label_path)
#         # break
