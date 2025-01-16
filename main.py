import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO
import threading

MAX_TARGET_WIDTH = 600
MAX_TARGET_HEIGHT = 400


class LicensePlateReader:
    def __init__(self, root):
        self.root = root
        self.root.title("AO projekt - Czytnik tablic rejestracyjnych")
        self.root.geometry("1000x700")
        self.model_path = "model/best.pt"
        self.model = YOLO(self.model_path)

        self.create_gui()

    def create_gui(self):
        self.label = tk.Label(
            self.root, text="Czytnik tablic rejestracyjnych", font=("Arial", 20)
        )
        self.label.pack(pady=20)

        self.image_label = tk.Label(
            self.root, text="Załącz plik, aby rozpocząć", bg="gray"
        )
        self.image_label.pack(pady=10, fill="both", expand=True)

        self.upload_button = tk.Button(
            self.root, text="Załącz plik", command=self.upload_image, font=("Arial", 14)
        )
        self.upload_button.pack(pady=10)

        self.result_label = tk.Label(
            self.root, text="Numery tablic:", font=("Arial", 16)
        )
        self.result_label.pack(pady=10)

        self.result_text = tk.Text(self.root, height=3, width=50, font=("Arial", 14))
        self.result_text.pack(pady=10)

        self.clear_button = tk.Button(
            self.root, text="Wyczyść", command=self.clear_results, font=("Arial", 14)
        )
        self.clear_button.pack(pady=5)

    def find_plate_numbers(self, image_path):
        results = self.model(image_path)
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        cropped_plates = []

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

            cropped_plate = image.crop((x_min, y_min, x_max, y_max))
            cropped_plates.append(cropped_plate)

        for cropped_plate in cropped_plates:
            self.display_plate_window(cropped_plate)

        return image

    def display_plate_window(self, cropped_plate):
        new_window = tk.Toplevel(self.root)
        new_window.title("Wycięta tablica rejestracyjna")

        cropped_plate_tk = ImageTk.PhotoImage(cropped_plate)

        label = tk.Label(new_window, image=cropped_plate_tk)
        label.image = cropped_plate_tk
        label.pack(pady=10, padx=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if file_path:
            thread = threading.Thread(target=self.process_image, args=(file_path,))
            thread.start()

    def process_image(self, file_path):
        self.image_label.config(text="Przetwarzanie...", bg="yellow")
        image = self.find_plate_numbers(file_path)

        image.thumbnail((MAX_TARGET_WIDTH, MAX_TARGET_HEIGHT))

        self.image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image_tk, text="", bg="white")
        self.image_label.image = self.image_tk

        detected_text = self.detect_plate_numbers(image)

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, detected_text)

    def detect_plate_numbers(self, image):
        # nasz algorytm
        return "Numery tablic"

    def clear_results(self):
        self.image_label.config(image="", text="Załącz plik, aby rozpocząć", bg="gray")
        self.result_text.delete("1.0", tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateReader(root)
    root.mainloop()
