#!/bin/bash

# Lista wymaganych pakietów Pythona
REQUIRED_PACKAGES=(
    "opencv-python"
    "datasets"
    "ultralytics"
    "Pillow"
    "numpy"
    "tensorflow"
    "scikit-learn"
    "matplotlib"
)

# Funkcja sprawdzająca dostępność polecenia
command_exists() {
    command -v "$1" &> /dev/null
}

# Funkcja instalująca Pythona i pip w zależności od dystrybucji
install_python() {
    echo "Sprawdzanie systemu operacyjnego..."

    if command_exists apt; then
        echo "Wykryto system oparty na Debianie/Ubuntu"
        sudo apt update
        sudo apt install -y python3 python3-pip python3-tk
    elif command_exists yum; then
        echo "Wykryto system oparty na RedHat/CentOS"
        sudo yum install -y python3 python3-pip python3-tkinter
    elif command_exists dnf; then
        echo "Wykryto system Fedora"
        sudo dnf install -y python3 python3-pip python3-tkinter
    elif command_exists pacman; then
        echo "Wykryto system Arch Linux"
        sudo pacman -Sy --noconfirm python python-pip tk
    else
        echo "Nieobsługiwana dystrybucja systemu Linux"
        exit 1
    fi

    # Sprawdzenie poprawności instalacji
    if ! command_exists python3; then
        echo "Błąd: Python nie został poprawnie zainstalowany."
        exit 1
    fi

    if ! command_exists pip3; then
        echo "Błąd: pip nie został poprawnie zainstalowany."
        exit 1
    fi
}

# Funkcja instalująca pakiety Pythona do systemu globalnie
install_packages() {
    echo "Instalowanie brakujących pakietów Pythona globalnie w systemie..."

    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! python3 -c "import ${package%%-*}" &> /dev/null; then
            echo "Instalowanie: $package"
            sudo pip3 install "$package" --break-system-packages
        else
            echo "Pakiet $package jest już zainstalowany."
        fi
    done

    echo "Instalacja pakietów zakończona pomyślnie."
}

# Funkcja uruchamiająca skrypt Pythona
run_main_script() {
    if [ -f "main.py" ]; then
        echo "Uruchamianie skryptu main.py..."
        python3 main.py
    else
        echo "Błąd: Plik main.py nie istnieje!"
        exit 1
    fi
}

# Główna część skryptu
install_python
install_packages
run_main_script