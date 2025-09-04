# Cough Sound Spectrogram Clustering [![Train Model](https://github.com/fffffatah/cough-sound-spectrogram-clustering/actions/workflows/train-model.yml/badge.svg)](https://github.com/fffffatah/cough-sound-spectrogram-clustering/actions/workflows/train-model.yml)

An unsupervised learning system for clustering cough sound spectrograms using a custom Convolutional Autoencoder.
This project implements a complete pipeline from data loading, calculating mean and std to clustering evaluation,
specifically written for analyzing different types of cough sounds without requiring labeled data. This project was
built for partial completion of CSE715 at BRAC University.

## Instructions

1. Clone the Repository by running the following command:
   ```bash
   git clone https://github.com/fffffatah/cough-sound-spectrogram-clustering.git
   
2. Navigate to the project directory:
   ```bash
   cd cough-sound-spectrogram-clustering
   
3. Install required dependencies using '**pip**' listed in '**requirements.txt**'
4. Download the Cough Sound Dataset from Kaggle and place it in the '_cough_dataset_' directory in project root. Make sure to extract the contents first.
5. Make the '**run.sh**' file executable and run it by using the following command:
   ```bash
   chmod +x run.sh && ./run.sh
    ```
After running the script, follow on screen instructions. If you don't want to re-generate the spectrograms, download from the following link to skip the first step after running the script.

Dataset Source: https://www.kaggle.com/datasets/orvile/coughvid-v3

Generated Spectrograms: https://mega.nz/file/4bg0wRZK#pXxCI6C-sFniXXB0yBAjINtPmyl8qo7SG_OsxdtSbDE

## Hardware Used

| Component       | Name                          |
|-----------------|-------------------------------|
| CPU             | Intel Core i7 13620H 13th Gen |
| RAM             | 32 GB                         |
| GPU             | NVIDIA RTX 4060               |
| Operating System| Ubuntu 24.04 (WSL)            |

Average Training Time: 6 minutes (10 epochs, batch size 32)

## Using GitHub Actions for Training
This project includes a GitHub Actions workflow that automates the training process. To use this, fork the repo and manually trigger the pipeline. 
But make sure you have access token configured in repository secrets with the name 'MY_TOKEN' to allow uploading the trained model and generated visualizations 
to release.

Average Training Time: 230 minutes (10 epochs, batch size 32)