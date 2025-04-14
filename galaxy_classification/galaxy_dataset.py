import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class GalaxyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (numpy.ndarray): Array of images with shape (N, H, W, C).
            labels (numpy.ndarray or torch.Tensor): Array of integer labels.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image and label
        image = self.images[idx]
        label = self.labels[idx]

        # If a transform is provided, assume it requires a PIL Image.
        if self.transform:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        else:
            # Otherwise, convert the numpy image to a tensor and adjust dimensions.
            image = torch.tensor(image, dtype=torch.float32)
            if image.ndim == 3:
                image = image.permute(2, 0, 1)

        return image, label