import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import logging
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
from collections import Counter
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class WildlifeDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._get_paths_labels()
        self.map_label = {label: idx for idx, label in enumerate(sorted(list(set(self.labels))))}
        logging.info(f"Found {len(self.image_paths)} images across {len(self.map_label)} classes")
        logging.info(f"Classes: {list(self.map_label.keys())}")

    def _get_paths_labels(
        self, valid_ext={".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    ) -> Tuple[List[str], List[str]]:
        labels = []
        image_paths = []

        for label_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, label_name)

            if not os.path.isdir(class_dir):
                continue

            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                extension = os.path.splitext(file_name)[1].lower()
                if extension in valid_ext:
                    image_paths.append(file_path)
                    labels.append(label_name)

        return image_paths, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_retries = 3

        for attempt in range(max_retries):
            try:
                actual_index = (index + attempt) % len(self.image_paths)
                image_path = self.image_paths[actual_index]
                label = self.map_label[self.labels[actual_index]]

                # Load image
                img = Image.open(image_path).convert("RGB")

                # Apply transforms
                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)

                return img, torch.tensor(label, dtype=torch.long)

            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue

        # If all retries fail
        raise RuntimeError(
            f"Could not load any valid image starting from index {index}"
        )

    def get_class_distribution(self):
        """Utility method to see class distribution"""
        return Counter(self.labels)


def get_sample_weights(dataset):
    """Calculate weights for WeightedRandomSampler to handle class imbalance"""
    # Count occurrences of each class
    label_counts = {}
    for label in dataset.labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Calculate inverse weights
    class_weights = {}
    for label, count in label_counts.items():
        class_weights[label] = 1.0 / count

    # Assign weight to each sample
    sample_weights = []
    for label in dataset.labels:
        sample_weights.append(class_weights[label])

    return sample_weights


# Define transforms
train_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def create_dataloaders(train_dir, val_dir, batch_size=64, num_workers=4, use_weighted_sampling=True):
    """Create training and validation dataloaders"""

    # Create datasets
    train_dataset = WildlifeDataset(train_dir, transform=train_transforms)
    val_dataset = WildlifeDataset(val_dir, transform=val_transforms)

    # Print class distributions
    print("\nTraining set class distribution:")
    for class_name, count in train_dataset.get_class_distribution().items():
        print(f"  {class_name}: {count} images")

    # Create training dataloader with optional weighted sampling
    if use_weighted_sampling:
        sample_weights = get_sample_weights(train_dataset)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Note: can't use shuffle=True with sampler
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation data
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data structure
    print("Testing WildlifeDataset implementation...")

    # This would be your actual usage:
    # train_loader, val_loader = create_dataloaders(
    #     train_dir="wildlife_dataset/train",
    #     val_dir="wildlife_dataset/test",
    #     batch_size=32,
    #     use_weighted_sampling=True
    # )

    # # Test a batch
    # for batch_idx, (images, labels) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
    #     if batch_idx == 2:  # Just test a few batches
    #         break

    print("Implementation complete! Ready to use with your wildlife dataset.")
