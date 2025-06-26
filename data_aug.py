import albumentations as A
import cv2
import os

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=100, p=0.4),
    A.RandomScale(scale_limit=0.3, p=0.4)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Beispielpfade
img_dir = "data/dataset/images/train"
label_dir = "data/dataset/labels/train"
aug_img_dir = "data/augmented/images/train"
aug_label_dir = "data/augmented/labels/train"

os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)


def load_yolo_labels(label_path):
    boxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_labels.append(int(parts[0]))
            box = list(map(float, parts[1:]))  # x_center, y_center, width, height
            boxes.append(box)
    return boxes, class_labels

def save_yolo_labels(output_path, boxes, class_labels):
    with open(output_path, 'w') as f:
        for cls, box in zip(class_labels, boxes):
            f.write(f"{cls} {' '.join(map(str, box))}\n")


for filename in os.listdir(img_dir):

    img_path = os.path.join(img_dir, filename)
    label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # Labels laden
    boxes, class_labels = load_yolo_labels(label_path)
    if not boxes:
        continue

    transformed = transform(image=image, bboxes=boxes, class_labels=class_labels)

    new_img = transformed['image']
    new_boxes = transformed['bboxes']
    new_labels = transformed['class_labels']



    out_image_path = os.path.join(aug_img_dir, "aug_" + filename)
    out_label_path = os.path.join(aug_label_dir, "aug_" + filename)

    cv2.imwrite(out_image_path, new_img)
    save_yolo_labels(out_label_path, new_boxes, new_labels)
