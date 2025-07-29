"""Helper script to removes classes from label files"""

import os

from src import constants as c


def filter_label_file(input_path: str, output_path: str, class_to_remove: str):
    """
    Filters one class from one label file

    Args:
        input_path (str): Path to the label file
        output_path (str): Path to the output label file
        class_to_remove (str): Class to remove from label file
    """
    with open(input_path, "r") as f:
        lines = f.readlines()

    # Keep all lines, that don't belong to class_to_remove
    filtered_lines = [
        line for line in lines if not line.strip().startswith(class_to_remove + " ")
    ]

    with open(output_path, "w") as f:
        f.writelines(filtered_lines)


def process_all_labels(label_dir, output_dir, class_to_remove):
    """
    Filters one class from all label files within a directory

    Args:
        label_dir (str): Path to the directory with label files
        output_dir (str): Path to the output directory
        class_to_remove (str): Class to remove from label files
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue
        input_path = os.path.join(label_dir, filename)
        output_path = os.path.join(output_dir, filename)
        filter_label_file(input_path, output_path, class_to_remove)

    print(
        f"Class '{class_to_remove}' was removed from label files and stored in '{output_dir}'."
    )


# if __name__ == "__main__":
#     # Remove labels in a directory for debug purposes
#
#     dataset_dir = c.DATASET_LOCATION
#     test_set_labels_dir = os.path.join(dataset_dir, "test", "labels")
#
#     process_all_labels(test_set_labels_dir, test_set_labels_dir, "0")
