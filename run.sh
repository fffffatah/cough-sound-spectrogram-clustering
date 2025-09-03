#!/bin/bash

echo "=== COUGH SOUND CLUSTERING ==="
echo "Please select an option:"
echo "1. Generate Spectrograms"
echo "2. Train"
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo "Running spectrogram generation..."
        python -u src/generate_spectrograms.py
        ;;
    2)
        echo "Running training..."
        python -u src/main.py
        ;;
    *)
        echo "Invalid choice. Please enter 1 or 2."
        exit 1
        ;;
esac
