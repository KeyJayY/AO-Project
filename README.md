# AO-Project
Analiza Obrazów Projekt

# **1. Wprowadzenie**
Projekt ten zajmuje się rozpoznawaniem liter przy użyciu technik uczenia maszynowego. Wykorzystuje różne skrypty do generowania zestawu danych, trenowania modelu oraz testowania jego dokładności. Program bazuje na sieciach neuronowych oraz zbiorze danych wygenerowanym na podstawie różnych czcionek.

# **2. Struktura projektu**
Projekt składa się z następujących plików i katalogów:
- **`README.md`** – Dokumentacja opisująca projekt.
- **`skrypt.sh`** – Skrypt automatyzujący instalację zależności i uruchamianie programu.
- **`images/`** – Folder zawierający przykładowe obrazy do testowania modelu.
- **`machine_learning_letter_recognition/train.py`** – Skrypt do trenowania modelu rozpoznawania liter.
- **`machine_learning/script.py`** – Skrypt do trenowania modelu detekcji tablic rejestracyjnych.
- **`create_dataset.py`** – Skrypt generujący zestaw danych na podstawie różnych czcionek.
- **`test_model.py`** – Skrypt testujący model na nowych danych.
- **`main.py`** – Główny skrypt do uruchamiania aplikacji.
- **`letter_regonizer.py`** – Moduł odpowiedzialny za ładowanie modelu i przetwarzanie danych wejściowych.
- **`best.pt`** – Plik zapisujący najlepszy wytrenowany model detekcji tablic rejestracyjnych w formacie PyTorch.
- **`letter_recognition_model.h5`** – Model rozpoznawania znaków zapisany w formacie Keras.

# **3. Instalacja**
- Aby uruchomić projekt, należy wykonać następujące kroki:
   1. Uruchomić skrypt instalacyjny:
      ```bash
      ./skrypt.sh
      ```
      Skrypt ten automatycznie instaluje wymagane zależności, przygotowuje dane i uruchamia aplikację.
- Alternatywnie można samodzielnie zainstalować potrzebne biblioteki zamieszczone w pliku requirements.txt, po zainstalowaniu uruchomić można skrypt main.py

# **4. Generowanie zestawu danych**
Plik **`create_dataset.py`** odpowiada za generowanie obrazów liter na podstawie różnych czcionek. Proces ten obejmuje:
- Wybór czcionek z folderu `fonts/`
- Generowanie obrazów z literami od A do Z oraz cyfry od 0 do 9
- Zapisywanie danych w odpowiednim formacie do treningu sieci neuronowej.

# **5. Trenowanie modelu rozpoznawania znaków**
Plik **`train.py`** obsługuje proces trenowania modelu. Główne kroki obejmują:
- Wczytanie wygenerowanego zbioru danych.
- Budowanie modelu sieci neuronowej.
- Trenowanie modelu przy użyciu optymalizatora Adam.
- Zapisywanie najlepszego modelu na podstawie wyników walidacji.

# **6. Testowanie modelu**
Testowanie odbywa się za pomocą pliku **`test_model.py`**, który:
- Wczytuje zapisany model.
- Przetwarza obrazy testowe.
- Wyświetla dokładność oraz wyniki klasyfikacji.

# **7. Użycie modelu**
Plik **`main.py`** to główny punkt wejścia do aplikacji. Umożliwia wprowadzenie obrazu zawierającego litery, przetworzenie go za pomocą modelu i zwrócenie rozpoznanych znaków.

# **8. Wymagania systemowe**
Projekt wymaga:
- Python 3.8 lub wyższy
- Biblioteki instalowane poprzez skrypt `skrypt.sh`, m.in.:
  - TensorFlow
  - OpenCV
  - NumPy
  - Matplotlib

# **9. Automatyzacja**
Plik **`skrypt.sh`** służy do szybkiego uruchomienia projektu, wykonując kolejne kroki, takie jak instalacja zależności i uruchomienie głównego skryptu.

# **10. Podsumowanie**
Projekt umożliwia szybkie i dokładne rozpoznawanie liter na podstawie obrazów. Dzięki modularnej strukturze pozwala na łatwe rozszerzanie i dostosowywanie do różnych zastosowań.

