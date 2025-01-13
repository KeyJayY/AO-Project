from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont

model_path = "model/best.pt"


def test_yolo_model():

    images_dir = "images"

    model = YOLO(model_path)

    image_files = [os.path.join(images_dir, f"example_{i}.jpg") for i in range(1, 4)]

    for image_file in image_files:
        print(f"Testowanie obrazu: {image_file}")
        results = model(image_file)

        for result in results:
            print(result.boxes.xyxy)


def draw_results_on_image(image_path):
    model_path = "model/best.pt"

    model = YOLO(model_path)

    results = model(image_path)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for box in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
        confidence = box.conf.item()
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
        text = f"Conf: {confidence:.2f}"

        text_bbox = draw.textbbox((x_min, y_min), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_position = (x_min, y_min - text_height)

        draw.rectangle(
            [text_position, (x_min + text_width, y_min)],
            fill="green",
        )

        draw.text(text_position, text, fill="white")

    image.show()


if __name__ == "__main__":
    test_yolo_model()
    draw_results_on_image("images/example_2.jpg")
