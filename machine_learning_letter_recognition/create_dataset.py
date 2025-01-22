import os
import cv2
import numpy as np
import random
from PIL import ImageFont, ImageDraw, Image


def create_dataset(
    output_dir="letters", image_size=(128, 128), num_images_per_char=1500
):
    characters = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(48, 58)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    font_paths = [
        os.path.join(font_dir, "din1451alt.ttf"),
        os.path.join(font_dir, "LexendGiga-Regular.ttf"),
        os.path.join(font_dir, "Rubik-Regular.ttf"),
        os.path.join(font_dir, "Lato-Regular.ttf"),
    ]

    for font_path in font_paths:
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Nie znaleziono pliku fontu: {font_path}")

    for char in characters:
        char_dir = os.path.join(output_dir, char)
        if not os.path.exists(char_dir):
            os.makedirs(char_dir)

        for i in range(num_images_per_char):
            canvas_size = (image_size[0] * 3, image_size[1] * 3)
            image = np.ones(canvas_size, dtype=np.uint8) * 255

            font_path = random.choice(font_paths)
            font_size = random.randint(20, 80)
            rotation_angle = random.uniform(-30, 30)
            font = ImageFont.truetype(font_path, font_size)

            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

            text_bbox = draw.textbbox((0, 0), char, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (canvas_size[0] - text_width) // 2
            text_y = (canvas_size[1] - text_height) // 2

            draw.text((text_x, text_y), char, font=font, fill=0)

            image = np.array(pil_image)

            M = cv2.getRotationMatrix2D(
                (canvas_size[0] // 2, canvas_size[1] // 2), rotation_angle, 1
            )
            rotated_image = cv2.warpAffine(
                image,
                M,
                canvas_size,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

            center_x, center_y = canvas_size[0] // 2, canvas_size[1] // 2
            cropped_image = rotated_image[
                center_y - image_size[1] // 2 : center_y + image_size[1] // 2,
                center_x - image_size[0] // 2 : center_x + image_size[0] // 2,
            ]

            if random.random() > 0.5:
                kernel_size = random.choice([(3, 3), (5, 5)])
                cropped_image = cv2.GaussianBlur(cropped_image, kernel_size, sigmaX=0)

            # Zapis obrazu
            image_path = os.path.join(char_dir, f"{i + 1}.png")
            cv2.imwrite(image_path, cropped_image)


if __name__ == "__main__":
    create_dataset()
