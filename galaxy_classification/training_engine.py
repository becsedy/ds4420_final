import os
import sys

# Append the parent directory (project root) to the sys.path for imports
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import csv
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from galaxy_dataset import GalaxyDataset
from models.simple_cnn import SimpleCNN
from models.powerful_cnn import PowerfulCNN
from models.efficient_net import create_efficientnet
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

def load_data(h5_file='../data/Galaxy10_DECals.h5'):
    with h5py.File(h5_file, 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])
    return images, labels

def create_splits(dataset, seed=42):
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    return train_dataset, val_dataset, test_dataset

def main(raw=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Instantiate the model and move to GPU.
    model = PowerfulCNN(num_classes=10).to(device)
    model_name = 'powerful_cnn'
    
    # Create directories if not already present
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load data
    images, labels = load_data()
    
    # Define transforms
    if raw:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        data_type = "raw"
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        data_type = "augmented"
    
    dataset = GalaxyDataset(images, labels, transform=transform)
    
    # Create data splits
    train_dataset, val_dataset, test_dataset = create_splits(dataset, seed=42)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    num_epochs = 55
    
    # Set up TensorBoard writer for logging.
    writer = SummaryWriter(log_dir=f"logs/experiment_{model_name}_{data_type}")
    
    best_val_acc = 0.0
    best_model_path = f"models/best_{model_name}_{data_type}.pth"

    # Write header for log file
    with open(f"logs/experiment_{model_name}_{data_type}.csv", mode='w', newline='') as f:
        c_writer = csv.writer(f)
        c_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    print("Starting Training!")
    
    for epoch in range(num_epochs):
        # Mark the starting point
        start_time = time.time()
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss_epoch = train_loss / train_total
        train_acc_epoch = train_correct / train_total
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Record time after the second section
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {end_time - start_time:.6f} seconds")
        
        # Log metrics
        writer.add_scalar("Loss/train", train_loss_epoch, epoch+1)
        writer.add_scalar("Loss/val", avg_val_loss, epoch+1)
        writer.add_scalar("Accuracy/train", train_acc_epoch, epoch+1)
        writer.add_scalar("Accuracy/val", val_acc, epoch+1)

        # Then, after each epoch:
        with open(f"logs/experiment_{model_name}_{data_type}.csv", mode='a', newline='') as f:
            c_writer = csv.writer(f)
            c_writer.writerow([epoch+1, train_loss_epoch, train_acc_epoch, avg_val_loss, val_acc])
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at {best_model_path} with Val Acc: {best_val_acc:.4f}")
    
    writer.close()
    print("Training complete.")

if __name__ == '__main__':
    # Change raw=True or raw=False as needed
    main(raw=False)