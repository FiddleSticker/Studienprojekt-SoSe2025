import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src import constants as c


def plot_image_with_bboxes(image_path: str, label_path: str) -> None:
    # Load image
    image = mpimg.imread(image_path)
    height, width = image.shape[:2]

    # Read bboxes from label_path
    bboxes = []
    if not os.path.exists(label_path):
        return

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, w, h = map(float, parts)
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                x_max = (x_center + w / 2) * width
                y_max = (y_center + h / 2) * height
                bboxes.append((x_min, y_min, x_max, y_max, int(class_id)))

    # Plot image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw bboxes into image
    for x_min, y_min, x_max, y_max, class_id in bboxes:
        class_name, class_color = c.CLASS_DICT[class_id]
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor=class_color,
            facecolor="none",
        )
        ax.add_patch(rect)
        label = f"Klasse= {class_name}"
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


# # VLT MÜSST IHR HIER EURE DIRECTORY PATHS ANPASSEN!
# dataset_dir = os.path.join(c.PROJECT_DIR, "Test2-5")
# test_set_dir = os.path.join(dataset_dir, "test")
# test_set_images_dir = os.path.join(test_set_dir, "images")
# test_set_labels_dir = os.path.join(test_set_dir, "labels")
#
# for file in os.listdir(test_set_images_dir):
#     file_name = os.path.splitext(file)[0]
#
#     image_path = os.path.join(test_set_images_dir, file)
#     label_path = os.path.join(test_set_labels_dir, f"{file_name}.txt")
#
#     plot_image_with_bboxes(image_path, label_path)
