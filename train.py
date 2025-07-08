import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataloader import create_data_loaders
from model import EEGTransformer
from utils import set_seed, count_parameters

def plot_confusion_matrix(y_true, y_pred, classes, filename='confusion_matrix.png'):
    """Plot and save a confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")

def train_model(config):
    """Main training function"""
    # Set random seeds for reproducibility
    set_seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        test_split=config['test_split'],
        standard_channels=config['n_channels'],  
        time_length=config['n_samples']
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGTransformer(
        emb_size=config['emb_size'],
        depth=config['depth'],
        n_classes=2,
    ).to(device)
    
    # Print model info
    print(f"Model parameters: {count_parameters(model)}")
    print(f"Using device: {device}")
    
    # Optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    # print(1)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    # print(2)
    
    # Class weighting for imbalanced datasets
    # Count class occurrences in train dataset
    class_counts = np.zeros(2, dtype=np.int64)
    progress_bar = tqdm(total=len(train_loader), desc="统计类别分布", unit="batch")
    
    for _, labels in train_loader:
        labels = labels.to(device)  # 确保在正确的设备上
        class_counts[0] += torch.sum(labels == 0).item()
        class_counts[1] += torch.sum(labels == 1).item()
        
        # 更新进度条
        progress_bar.update(1)
        progress_bar.set_postfix({
            "Class 0": class_counts[0],
            "Class 1": class_counts[1]
        })
    
    progress_bar.close()
    
    print(f"\nClass distribution: Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
    
    # Calculate weights inversely proportional to class frequencies
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training variables
    best_val_f1 = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Start training
    print("\nStarting training...")
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update scheduler
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(config['output_dir'], 'best_model.pth'))
            print(f"Saved new best model with F1: {val_f1:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['num_epochs']} - {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'best_model.pth')))
    print("Loaded best model for testing")
    
    # Final evaluation on test set
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1 Score: {test_f1:.4f}")
    
    # Save training and validation metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1
    }
    torch.save(metrics, os.path.join(config['output_dir'], 'metrics.pth'))
    
    # Plot confusion matrix for test set
    print("\nTest set performance:")
    y_true, y_pred = predict_model(model, test_loader, device)
    print(classification_report(y_true, y_pred, target_names=['Other', 'Trueictal']))
    plot_confusion_matrix(y_true, y_pred, ['Other', 'Trueictal'], 
                         os.path.join(config['output_dir'], 'confusion_matrix.png'))
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'training_curves.png'))
    plt.close()
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(config['output_dir'], 'final_model.pth'))
    print("\nTraining completed.")

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 确保张量正确转换
            all_labels.append(labels.cpu())  # 先移动回CPU
            all_preds.append(predicted.cpu())
    
    # 统一转换为NumPy数组
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def predict_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 先收集CPU张量
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    
    # 转换为NumPy数组
    return (
        torch.cat(all_labels).numpy(), 
        torch.cat(all_preds).numpy()
    )

if __name__ == "__main__":
    # Configuration settings
    config = {
        'seed': 42,  # Random seed for reproducibility
        'data_dir': '../../data',  # Preprocessed data directory
        'output_dir': '../../a-testcode/EEG-Transformer',  # Directory for output files
        'batch_size': 8,
        'num_epochs': 5,
        'lr': 0.0001,  # Learning rate
        'weight_decay': 1e-4,  # L2 regularization
        'val_split': 0.15,  # Validation set ratio
        'test_split': 0.15,  # Test set ratio
        'emb_size': 40,  # Embedding dimension
        'depth': 4,  # Number of transformer layers
        'n_channels': 64,  # Number of EEG channels (adjust based on your data)
        'n_samples': 2048,  # EEG segment length in samples (4 seconds * 128 Hz = 512)
    }
    
    # Start training
    train_model(config)