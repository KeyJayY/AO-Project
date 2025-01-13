import os
import cv2
from datasets import load_dataset
from ultralytics import YOLO

from PIL import Image


def prepare_yolo_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "yolo_dataset")

    dataset = load_dataset("keremberke/license-plate-object-detection", "full")

    os.makedirs(os.path.join(dataset_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels/val"), exist_ok=True)

    for idx, sample in enumerate(dataset["train"]):
        image = sample["image"]
        bboxes = sample["objects"]["bbox"]
        labels = sample["objects"]["category"]

        img_output_path = os.path.join(dataset_dir, f"images/train/image_{idx}.jpg")
        image.save(img_output_path, "JPEG")

        width, height = image.size

        label_output_path = os.path.join(dataset_dir, f"labels/train/image_{idx}.txt")
        with open(label_output_path, "w") as f:
            for bbox in bboxes:
                yolo_bbox = convert_bbox_to_yolo(bbox, width, height)
                f.write(f"0 {' '.join(map(str, yolo_bbox))}\n")

    for idx, sample in enumerate(dataset["validation"]):
        image = sample["image"]
        bboxes = sample["objects"]["bbox"]
        labels = sample["objects"]["category"]

        img_output_path = os.path.join(dataset_dir, f"images/val/image_{idx}.jpg")
        image.save(img_output_path, "JPEG")

        width, height = image.size

        label_output_path = os.path.join(dataset_dir, f"labels/val/image_{idx}.txt")
        with open(label_output_path, "w") as f:
            for bbox in bboxes:
                yolo_bbox = convert_bbox_to_yolo(bbox, width, height)
                f.write(f"0 {' '.join(map(str, yolo_bbox))}\n")


def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, bbox_width, bbox_height = bbox

    x_center = (x_min + bbox_width / 2) / img_width
    y_center = (y_min + bbox_height / 2) / img_height

    width = bbox_width / img_width
    height = bbox_height / img_height

    return [x_center, y_center, width, height]


def train_yolo_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "yolo_dataset")

    data_yaml_content = f"""train: {os.path.join(dataset_dir, 'images/train')}
                            val: {os.path.join(dataset_dir, 'images/val')}

                            nc: 1  # Liczba klas (tablice rejestracyjne)
                            names: ['license_plate']
                            """
    with open(os.path.join(dataset_dir, "data.yaml"), "w") as f:
        f.write(data_yaml_content)

    model = YOLO("yolov8n.pt")
    model.train(
        data=os.path.join(dataset_dir, "data.yaml"),
        epochs=5,
        imgsz=640,
        batch=16,
        name="license_plate_detector",
    )


if __name__ == "__main__":
    print("prepare dataset")
    prepare_yolo_dataset()
    print("Training model")
    train_yolo_model()
    print("Trening finished")
