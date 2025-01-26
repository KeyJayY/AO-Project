import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO
import threading
from letter_regonizer import recognize_letters
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Używanie tylko CPU


MAX_TARGET_WIDTH = 600
MAX_TARGET_HEIGHT = 400


class LicensePlateReader:
    def __init__(self, root):
        """
        Inicjalizuje aplikację, wczytuje model YOLO i ustawia główne elementy GUI.
        """
        self.root = root
        self.root.title("AO projekt - Czytnik tablic rejestracyjnych")
        self.root.geometry("1000x700")
        self.model_path = "model/best.pt"
        self.model = YOLO(self.model_path)

        self.setup_gui()

    def setup_gui(self):
        """
        Tworzy elementy interfejsu użytkownika.
        """
        self.label = tk.Label(
            self.root, text="Czytnik tablic rejestracyjnych", font=("Arial", 20)
        )
        self.label.pack(pady=20)

        self.image_label = tk.Label(
            self.root, text="Załącz plik, aby rozpocząć", bg="gray"
        )
        self.image_label.pack(pady=10, fill="both", expand=True)

        self.upload_button = tk.Button(
            self.root, text="Załącz plik", command=self.upload_file, font=("Arial", 14)
        )
        self.upload_button.pack(pady=10)

        self.result_label = tk.Label(
            self.root, text="Numery tablic:", font=("Arial", 16)
        )
        self.result_label.pack(pady=10)

        self.result_text = tk.Text(
            self.root, height=3, width=50, font=("Arial", 14), state="disabled"
        )
        self.result_text.pack(pady=10)

        self.clear_button = tk.Button(
            self.root, text="Wyczyść", command=self.clear_results, font=("Arial", 14)
        )
        self.clear_button.pack(pady=5)

    def detect_plate_numbers(self, image_path):
        """
        Wykrywa tablice rejestracyjne oraz odczytuje ich numery na podstawie wprowadzonego obrazu.

        Args:
            image_path (str): Ścieżka do obrazu.

        Returns:
            tuple: odczytany tekst tablicy oraz współrzędne znaków.
        """
        results = self.model(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box = results[0].boxes[0]
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

        cropped_plate = image[y_min:y_max, x_min:x_max]
        cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)

        characters, letters_with_coords = recognize_letters(cropped_plate)
        characters = "".join(characters)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

        original_height, original_width = cropped_plate.shape[:2]
        scale_x = original_width / 128
        scale_y = original_height / 32

        for lx, ly, lw, lh in letters_with_coords:
            scaled_x_min = int(lx * scale_x)
            scaled_y_min = int(ly * scale_y)
            scaled_x_max = int((lx + lw) * scale_x)
            scaled_y_max = int((ly + lh) * scale_y)

            global_x_min = x_min + scaled_x_min
            global_y_min = y_min + scaled_y_min
            global_x_max = x_min + scaled_x_max
            global_y_max = y_min + scaled_y_max

            cv2.rectangle(
                image,
                (global_x_min, global_y_min),
                (global_x_max, global_y_max),
                (255, 0, 0),
                2,
            )

        return image, characters

    def upload_file(self):
        """
        Obsługuje wybór pliku przez użytkownika i uruchamia proces przetwarzania obrazu.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            thread = threading.Thread(target=self.process_file, args=(file_path,))
            thread.start()

    def process_file(self, file_path):
        """
        Przetwarza wybrany obraz, wykrywa tablice i aktualizuje GUI.

        Args:
            file_path (str): Ścieżka do wybranego obrazu.
        """
        self.image_label.config(text="Przetwarzanie...", bg="yellow")
        processed_image, characters = self.detect_plate_numbers(file_path)

        height, width, _ = processed_image.shape
        scaling_factor = min(MAX_TARGET_WIDTH / width, MAX_TARGET_HEIGHT / height, 1)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(processed_image, (new_width, new_height))

        resized_image_tk = self.convert_to_tkinter_image(resized_image)
        self.image_label.config(image=resized_image_tk, text="", bg="white")
        self.image_label.image = resized_image_tk

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, characters)
        self.result_text.config(state="disabled")

    def convert_to_tkinter_image(self, cv_image):
        """
        Konwertuje obraz OpenCV na format kompatybilny z Tkinter.

        Args:
            cv_image (numpy.ndarray): Obraz w formacie OpenCV.

        Returns:
            tk.PhotoImage: Obraz kompatybilny z Tkinter.
        """
        cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", cv_image_bgr)
        return tk.PhotoImage(data=buffer.tobytes())

    def clear_results(self):
        """
        Czyści wyniki i resetuje GUI.
        """
        self.image_label.config(image="", text="Załącz plik, aby rozpocząć", bg="gray")
        self.result_text.delete("1.0", tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateReader(root)
    root.mainloop()
