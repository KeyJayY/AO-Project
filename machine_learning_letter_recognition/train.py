import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# trzeba wcześniej ppobrać dataset z https://www.kaggle.com/datasets/aladdinss/license-plate-digits-classification-dataset
# oraz stworzyć dataset z literami przy użyciu create_dataset.py
DATASET_DIR = "letters"
DATASET_DIR2 = "CNN_letter_dataset"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10


import cv2
import numpy as np

IMG_SIZE = (64, 64)


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        cropped_image = binary_image[y : y + h, x : x + w]
    else:
        cropped_image = binary_image

    closed = cv2.morphologyEx(cropped_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    resized_image = cv2.resize(closed, IMG_SIZE)

    return resized_image


def load_dataset(dataset_dir=DATASET_DIR, dataset_dir2=DATASET_DIR2):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))
    for label, class_name in enumerate(class_names):
        print(label)
        class_dir = os.path.join(dataset_dir, class_name)
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            if os.path.isfile(file_path):
                processed_image = preprocess_image(file_path)
                images.append(processed_image)
                labels.append(label)
    class_names2 = sorted(os.listdir(dataset_dir2))
    for label, class_name in enumerate(class_names2):
        print(label)
        class_dir = os.path.join(dataset_dir2, class_name)
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            if os.path.isfile(file_path):
                processed_image = preprocess_image(file_path)
                images.append(processed_image)
                labels.append(label)
    return np.array(images), np.array(labels), class_names


print("Wczytywanie danych...")
images, labels, class_names = load_dataset(DATASET_DIR)

images = images / 255.0
images = np.expand_dims(images, axis=-1)

labels = to_categorical(labels, num_classes=len(class_names))

X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Liczba klas: {len(class_names)}")
print(f"Rozmiar danych treningowych: {X_train.shape[0]}")
print(f"Rozmiar danych walidacyjnych: {X_val.shape[0]}")
print(f"Rozmiar danych testowych: {X_test.shape[0]}")


# 2. Tworzenie modelu
def build_model(input_shape, num_classes):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


model = build_model(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=len(class_names)
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Dokładność na zbiorze testowym: {test_accuracy:.2f}")

plt.plot(history.history["accuracy"], label="Dokładność treningowa")
plt.plot(history.history["val_accuracy"], label="Dokładność walidacyjna")
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.legend()
plt.savefig("accuracy.png")
plt.show()

model.save("letter_recognition_model.h5")
print("Model zapisano jako 'letter_recognition_model.h5'.")


def predict_image(model, image_path, class_names):
    image = preprocess_image(image_path)
    image = image / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class
