import cv2
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import load_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Używanie tylko CPU

# Wczytanie modeli YOLO i klasyfikatora liter
model_path = "model/best.pt"
model = YOLO(model_path)
model2 = load_model("model/letter_recognition_model.h5")


def find_plate(image_path):
    """
    Wykrywa tablicę rejestracyjną na podstawie obrazu.
    Zwraca wycięty fragment obrazu zawierający tablicę.
    """
    results = model(image_path)
    image = cv2.imread(image_path)
    box = results[0].boxes[0]
    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
    cropped_plate = image[y_min:y_max, x_min:x_max]
    return cropped_plate


def preprocess_image(image):
    """
    Przetwarza obraz, konwertując go na skalę szarości, zmieniając rozmiar i progowanie.
    Zwraca obraz binarny.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 32))
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def remove_borders(image):
    """
    Usuwa obramowanie obrazu poprzez użycie operacji flood fill.
    Zwraca obraz bez obramowania.
    """
    pad = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    h, w = pad.shape
    mask = np.zeros([h + 2, w + 2], np.uint8)
    img_floodfill = cv2.floodFill(pad, mask, (0, 0), 0, (5), (0), flags=8)[1]
    img_floodfill = img_floodfill[1 : h - 1, 1 : w - 1]
    return img_floodfill


def find_letters(image):
    """
    Segmentuje litery na obrazie na podstawie histogramu pionowego.
    Zwraca listę obrazów liter oraz ich współrzędne.
    """
    vertical_hist = np.sum(image, axis=0)
    threshold = 0.01 * np.max(vertical_hist)
    char_regions = []
    start = None
    for x, val in enumerate(vertical_hist):
        if val > threshold and start is None:
            start = x
        elif val <= threshold and start is not None:
            end = x
            if end - start > 2:
                char_regions.append((start, end))
            start = None
    if start is not None:
        char_regions.append((start, len(vertical_hist)))
    characters = []
    coordinates = []
    for start, end in char_regions:
        char_image = image[:, start:end]
        if 2 < char_image.shape[1] < image.shape[1] * 0.3:
            characters.append(char_image)
            coordinates.append((start, 0, end - start, image.shape[0]))
    return characters, coordinates


def predict_image(image, class_names):
    """
    Przewiduje klasę litery na podstawie obrazu przy użyciu modelu.
    Zwraca przewidywaną klasę.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y : y + h, x : x + w]
    else:
        cropped_image = image
    closed = cv2.morphologyEx(cropped_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    resized_image = cv2.resize(closed, (64, 64))
    gray_image = (
        resized_image
        if len(resized_image.shape) == 2
        else cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    )
    gray_image = np.expand_dims(gray_image, axis=-1).astype("float32") / 255.0
    gray_image = np.expand_dims(gray_image, axis=0)
    prediction = model2.predict(gray_image)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class


def recognize_letters(image):
    """
    Rozpoznaje litery na obrazie tablicy rejestracyjnej.
    Zwraca listę rozpoznanych liter oraz ich współrzędne.
    """
    processed_image = preprocess_image(image)
    processed_image = cv2.bitwise_not(processed_image)
    processed_image = remove_borders(processed_image)
    letters, coordinates = find_letters(processed_image)
    plates = []
    class_names = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
    for letter in letters:
        predicted_class = predict_image(letter, class_names)
        plates.append(predicted_class)
    return plates, coordinates
