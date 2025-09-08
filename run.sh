#!/bin/bash

MODEL_TYPE=${1:-custom}

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
        if [ "$MODEL_TYPE" = "custom" ]; then
            echo "Training - Custom Autoencoder with Contrastive Loss..."
            python -u src/main.py --model custom
        elif [ "$MODEL_TYPE" = "convvae" ]; then
            echo "Training - Convolutional Variational Autoencoder..."
            python -u src/main.py --model convvae
        else
            echo "Error: Invalid model type '$MODEL_TYPE'"
            echo "Valid options: custom, convvae"
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice. Please enter 1 or 2."
        exit 1
        ;;
esac
