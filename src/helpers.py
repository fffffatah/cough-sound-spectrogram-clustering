import os
import torch
import random
from torchvision import transforms
from PIL import Image


def calculate_dataset_stats(files, data_dir):
    # Calculate mean and std for RGB channels across the dataset, we will need this during Normalization
    # Basic transform to tensor only
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Accumulate pixel values for each channel
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0

    print(f"Calculating stats from {len(files)} images...")

    for i, filename in enumerate(files):
        if i % 500 == 0:
            print(f"Processed {i}/{len(files)} images")

        try:
            img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
            img_tensor = transform(img)  # [3, 128, 128]

            # Sum across spatial dimensions (H, W)
            pixel_sum += img_tensor.sum(dim=[1, 2])
            pixel_squared_sum += (img_tensor ** 2).sum(dim=[1, 2])
            num_pixels += img_tensor.shape[1] * img_tensor.shape[2]  # H * W

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Calculate mean and std
    mean = pixel_sum / num_pixels
    std = torch.sqrt((pixel_squared_sum / num_pixels) - (mean ** 2))

    print(f"Dataset statistics:")
    print(f"Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]", flush=True)
    print(f"Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    return mean.tolist(), std.tolist()


def load_images(data_dir, max_images=34394):
    # Load samples randomly
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    files = random.sample(all_files, min(max_images, len(all_files)))
    mean, std = calculate_dataset_stats(files, data_dir)

    # Using calculated mean and std for normalization
    # mean = [0.3540, 0.1148, 0.2336]
    # std = [0.3614, 0.1229, 0.2128]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # 3 Channel Normalization
    ])

    images = []
    print(f"Loading {len(files)} images...")

    for i, filename in enumerate(files):
        if i % 500 == 0:
            print(f"Loaded {i}/{len(files)}")

        try:
            img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
            images.append(transform(img))

        except Exception as e:
            print(f"Error: {e}")

    return torch.stack(images)