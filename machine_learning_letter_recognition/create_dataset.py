import os
import cv2
import numpy as np
import random
from PIL import ImageFont, ImageDraw, Image


def create_dataset(
    output_dir="letters", image_size=(128, 128), num_images_per_char=1000
):
    characters = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(48, 58)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    font_paths = [
        os.path.join(font_dir, "din1451alt.ttf"),
        os.path.join(font_dir, "LexendGiga-Regular.ttf"),
        os.path.join(font_dir, "Rubik-Regular.ttf"),
        os.path.join(font_dir, "arial.ttf"),
    ]

    for font_path in font_paths:
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Nie znaleziono pliku fontu: {font_path}")

    for char in characters:
        char_dir = os.path.join(output_dir, char)
        if not os.path.exists(char_dir):
            os.makedirs(char_dir)

        for i in range(num_images_per_char):
            image = np.ones((image_size[0], image_size[1]), dtype=np.uint8) * 255

            font_path = random.choice(font_paths)
            font_size = random.randint(20, 80)
            rotation_angle = random.uniform(-30, 30)
            font = ImageFont.truetype(font_path, font_size)

            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

            text_bbox = draw.textbbox((0, 0), char, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            max_x = max(0, image_size[0] - text_width)
            max_y = max(0, image_size[1] - text_height)
            text_x = random.randint(0, max_x)
            text_y = random.randint(0, max_y)

            draw.text((text_x, text_y), char, font=font, fill=0)

            image = np.array(pil_image)

            M = cv2.getRotationMatrix2D(
                (image_size[0] // 2, image_size[1] // 2), rotation_angle, 1
            )
            image = cv2.warpAffine(
                image,
                M,
                (image_size[0], image_size[1]),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

            image_path = os.path.join(char_dir, f"{i + 1}.png")
            cv2.imwrite(image_path, image)


if __name__ == "__main__":
    create_dataset()
