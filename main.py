import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class LicensePlateReader:
    def __init__(self, root):
        self.root = root
        self.root.title("AO projekt - Czytnik tablic rejestracyjnych")
        self.root.geometry("1000x700")

        self.create_gui()

    def create_gui(self):
        self.label = tk.Label(self.root, text="Czytnik tablic rejestracyjnych", font=("Arial", 20))
        self.label.pack(pady=20)

        self.image_label = tk.Label(self.root, text="Załącz plik, aby rozpocząć", bg="gray")
        self.image_label.pack(pady=10, fill="both", expand=True)

        self.upload_button = tk.Button(self.root, text="Załącz plik", command=self.upload_image, font=("Arial", 14))
        self.upload_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="Numery tablic:", font=("Arial", 16))
        self.result_label.pack(pady=10)

        self.result_text = tk.Text(self.root, height=3, width=50, font=("Arial", 14))
        self.result_text.pack(pady=10)

        self.clear_button = tk.Button(self.root, text="Wyczyść", command=self.clear_results, font=("Arial", 14))
        self.clear_button.pack(pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            image = Image.open(file_path)
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
