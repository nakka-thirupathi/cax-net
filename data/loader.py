import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import Config


class XRayDataset(Dataset):
    """
    * PyTorch Dataset for X-ray images
    """
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        # grayscale to 3-channel RGB for pretrained models
        image = np.stack([image, image, image], axis=0)
        image = torch.FloatTensor(image)
        label = torch.LongTensor([self.labels[idx]])[0]
        return image, label


def find_data_folder(base_path):
    """Find the chest_xray folder in the dataset path"""
    chest_xray_path = os.path.join(base_path, 'chest_xray')
    return chest_xray_path if os.path.exists(chest_xray_path) else base_path


def load_images(data_dir, img_size=Config.IMG_SIZE, labels=Config.LABELS):
    """
    * Load and preprocess X-ray images into numpy array
    
    Args:
        data_dir: Directory containing image folders
        img_size: Target image size
        labels: List of class labels
    
    Returns:
        numpy array of [image, label] pairs
    """
    data = []
    for label in labels:
        label_path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        
        if not os.path.exists(label_path):
            print(f"Warning: Path {label_path} does not exist")
            continue
            
        for img in os.listdir(label_path):
            try:
                img_arr = cv2.imread(
                    os.path.join(label_path, img), 
                    cv2.IMREAD_GRAYSCALE
                )
                if img_arr is None:
                    continue
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    
    return np.array(data, dtype=object)


def prepare_data(dataset_path):
    """
    * Load and prepare train/test datasets
    
    Args:
        dataset_path: Root path to dataset
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    dataset_root = find_data_folder(dataset_path)
    
    print("Loading data...")
    train = load_images(os.path.join(dataset_root, "train"))
    test = load_images(os.path.join(dataset_root, "test"))
    val = load_images(os.path.join(dataset_root, "val"))
    
    # Combine train and validation
    train = np.concatenate([train, val], axis=0)
    
    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    
    # Separate features and labels, normalize
    x_train = np.array([i[0] for i in train]) / 255.0
    y_train = np.array([i[1] for i in train])
    
    x_test = np.array([i[0] for i in test]) / 255.0
    y_test = np.array([i[1] for i in test])
    
    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"Train - Pneumonia={np.sum(y_train==0)}, Normal={np.sum(y_train==1)}")
    print(f"Test  - Pneumonia={np.sum(y_test==0)}, Normal={np.sum(y_test==1)}")
    
    return x_train, y_train, x_test, y_test


def create_dataloaders(x_train, y_train, x_test, y_test, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    * Create PyTorch DataLoaders
    
    Args:
        x_train, y_train: Training data and labels
        x_test, y_test: Test data and labels
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = XRayDataset(x_train, y_train)
    test_dataset = XRayDataset(x_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, test_loader
