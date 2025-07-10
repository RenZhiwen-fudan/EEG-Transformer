import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch.distributed as dist
import csv

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

def get_lr_scheduler(optimizer, config):
    """Return appropriate learning rate scheduler based on configuration"""
    scheduler_name = config['lr_scheduler']['name']
    scheduler_params = config['lr_scheduler']['params']
    
    if scheduler_name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            **scheduler_params
        )
    elif scheduler_name == 'StepLR':
        return optim.lr_scheduler.StepLR(
            optimizer, 
            **scheduler_params
        )
    elif scheduler_name == 'CosineAnnealing':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            **scheduler_params
        )
    elif scheduler_name == 'Exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            **scheduler_params
        )
    elif scheduler_name == 'Cyclic':
        return optim.lr_scheduler.CyclicLR(
            optimizer,
            **scheduler_params
        )
    else:
        print(f"Warning: Unknown scheduler '{scheduler_name}'. Using default ReduceLROnPlateau")
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )

def train_model(config):
    """Main training function"""
    # Set random seeds for reproducibility
    set_seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create CSV file for saving epoch metrics
    metrics_file = os.path.join(config['output_dir'], 'epoch_metrics.txt')
    with open(metrics_file, 'w', newline='') as f:
        header = [
            'Epoch', 'Train Loss', 'Val Loss', 'Test Loss', 
            'Train Acc', 'Val Acc', 'Test Acc', 
            'Test F1', 'Test AUCROC', 'Params LT (µs)', 
            'Epoch Time (s)', 'Learning Rate', 'Early Stopping'
        ]
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
    
    # Save config for future reference
    with open(os.path.join(config['output_dir'], 'config.txt'), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
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
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available")
    
    model = EEGTransformer(
        emb_size=config['emb_size'],
        depth=config['depth'],
        n_classes=2,
    )
    
    # Use DataParallel for multi-GPU training
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params}")
    print(f"Using device: {device}")
    
    # Optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, config)
    print(f"Using {config['lr_scheduler']['name']} scheduler with params: {config['lr_scheduler']['params']}")
    
    # Class weighting for imbalanced datasets
    class_counts = np.zeros(2, dtype=np.int64)
    progress_bar = tqdm(total=len(train_loader), desc="统计类别分布", unit="batch")
    
    for _, labels in train_loader:
        labels = labels.to(device)
        class_counts[0] += torch.sum(labels == 0).item()
        class_counts[1] += torch.sum(labels == 1).item()
        progress_bar.update(1)
        progress_bar.set_postfix({
            "Class 0": class_counts[0],
            "Class 1": class_counts[1]
        })
    
    progress_bar.close()
    
    print(f"\nClass distribution: Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
    
    # Calculate class weights
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training variables
    best_val_f1 = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    test_losses = []
    train_accs = []
    val_accs = []
    test_accs = []
    test_f1s = []
    test_aurocs = []
    params_latencies = []
    learning_rates = []
    
    # Early stopping variables
    early_stop_counter = 0
    min_delta = config.get('early_stopping', {}).get('min_delta', 0.001)
    patience = config.get('early_stopping', {}).get('patience', 5)
    early_stop_triggered = False
    
    # Start training
    print("\nStarting training...")
    print(f"Early stopping configured: patience={patience}, min_delta={min_delta:.4f}")
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
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
        
        # Validation - FIXED HERE: added val_aucroc variable
        val_loss, val_acc, val_f1, val_aucroc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update scheduler based on validation metric or step
        if config['lr_scheduler']['name'] == 'ReduceLROnPlateau':
            scheduler.step(val_f1)  # Use validation F1 for plateau scheduler
        else:
            scheduler.step()  # Step-based schedulers update after every epoch
        
        # Evaluate on test set after each epoch
        test_loss, test_acc, test_f1, test_aucroc = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_aurocs.append(test_aucroc)
        
        # Measure inference latency
        latency = measure_inference_latency(model, test_loader, device, num_inferences=100)
        params_latencies.append(latency)
        
        # Save best model based on validation F1
        if val_f1 > best_val_f1:
            # Check improvement for early stopping
            improvement = val_f1 - best_val_f1
            if improvement > min_delta:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                early_stop_counter = 0  # Reset counter on improvement
                
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'test_f1': test_f1,
                    'test_acc': test_acc,
                    'test_aucroc': test_aucroc
                }, os.path.join(config['output_dir'], 'best_model.pth'))
                    
                print(f"Saved new best model at epoch {epoch+1} with Val F1: {val_f1:.4f} (↑{improvement:.4f})")
            else:
                print(f"No significant improvement: {improvement:.4f} < min_delta ({min_delta:.4f})")
        else:
            # No improvement
            improvement = best_val_f1 - val_f1
            early_stop_counter += 1
            print(f"No improvement in validation F1. Counter: {early_stop_counter}/{patience}")
            
            # Check early stopping condition
            if early_stop_counter >= patience and not early_stop_triggered:
                early_stop_triggered = True
                print(f"⚠️ Early stopping triggered at epoch {epoch+1}! "
                      f"No improvement for {patience} consecutive epochs.")
        
        epoch_time = time.time() - epoch_start
        
        # Save epoch metrics to TXT file
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([
                epoch+1,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{test_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_acc:.6f}",
                f"{test_acc:.6f}",
                f"{test_f1:.6f}",
                f"{test_aucroc:.6f}",
                f"{latency:.6f}",
                f"{epoch_time:.2f}",
                f"{current_lr:.8f}",
                "True" if early_stop_triggered else "False"
            ])
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - {epoch_time:.1f}s")
        print(f"LR: {current_lr:.8f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test AUCROC: {test_aucroc:.4f} | Latency: {latency:.2f} µs")
        print("-" * 80)
        
        # Break if early stopping is triggered
        if early_stop_triggered:
            print("⏹ Training stopped early due to validation performance.")
            break
    
    # Load best model
    best_model_path = os.path.join(config['output_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded best model from epoch {checkpoint['epoch']} for final testing")
    else:
        print("No best model found, using final model")
        checkpoint = {'epoch': config['num_epochs']}
    
    # Final evaluation on test set
    test_loss, test_acc, test_f1, test_aucroc = evaluate_model(model, test_loader, criterion, device)
    latency = measure_inference_latency(model, test_loader, device)
    print(f"\nFinal Test Results (Best Model Epoch {checkpoint['epoch']}):")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1 Score: {test_f1:.4f} | AUCROC: {test_aucroc:.4f} | Latency: {latency:.2f} µs")
    
    # Save training and validation metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_accs': test_accs,
        'test_f1s': test_f1s,
        'test_aurocs': test_aurocs,
        'params_latencies': params_latencies,
        'learning_rates': learning_rates,
        'num_params': num_params,
        'num_gpus_used': num_gpus,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'early_stop_triggered': early_stop_triggered,
        'final_epoch': len(train_losses)  # Actual number of epochs trained
    }
    
    # Save metrics to text file
    metrics_txt_file = os.path.join(config['output_dir'], 'final_metrics.txt')
    with open(metrics_txt_file, 'w') as f:
        for key, value in metrics.items():
            if isinstance(value, list):
                f.write(f"{key}: {','.join(map(str, value))}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\nConfig Settings:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Plot confusion matrix for test set
    print("\nTest set performance (Best Model):")
    y_true, y_pred = predict_model(model, test_loader, device)
    print(classification_report(y_true, y_pred, target_names=['Other', 'Trueictal']))
    plot_confusion_matrix(y_true, y_pred, ['Other', 'Trueictal'], 
                         os.path.join(config['output_dir'], 'confusion_matrix.png'))
    
    # Plot training curves
    plt.figure(figsize=(18, 12))
    
    # Loss Curve
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(test_losses, label='Test Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    # Accuracy Curve
    plt.subplot(2, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.plot(test_accs, label='Test Acc', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    
    # Learning Rate Curve
    plt.subplot(2, 3, 3)
    plt.plot(learning_rates, label='Learning Rate', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.title('Learning Rate Schedule')
    
    # F1 Score Curve
    plt.subplot(2, 3, 4)
    plt.plot(test_f1s, label='Test F1', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Test F1 Score Curve')
    
    # AUCROC Curve
    plt.subplot(2, 3, 5)
    plt.plot(test_aurocs, label='Test AUCROC', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('AUCROC')
    plt.legend()
    plt.title('Test AUCROC Curve')
    
    # Latency Curve
    plt.subplot(2, 3, 6)
    plt.plot(params_latencies, label='Inference Latency', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Latency (µs)')
    plt.legend()
    plt.title('Inference Latency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'training_curves.png'))
    plt.close()
    
    # Combined plot: F1 and AUCROC
    plt.figure(figsize=(10, 6))
    plt.plot(test_f1s, label='Test F1', color='purple')
    plt.plot(test_aurocs, label='Test AUCROC', color='green')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label='Best Epoch')
    
    if early_stop_triggered:
        plt.axvline(x=len(test_f1s)-1, color='orange', linestyle='--', label='Early Stop')
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation and Test Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config['output_dir'], 'performance_curves.png'))
    plt.close()
    
    # Save the final model
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': len(train_losses),
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
        'test_f1': test_f1,
        'test_acc': test_acc,
        'test_aucroc': test_aucroc,
        'early_stop': early_stop_triggered
    }, os.path.join(config['output_dir'], 'final_model.pth'))
    
    print("\nTraining completed.")

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Collect predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), average='weighted')
    
    # Calculate AUC ROC
    try:
        aucroc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        aucroc = 0.5
    
    return avg_loss, accuracy, f1, aucroc

def predict_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    
    return (
        torch.cat(all_labels).numpy(), 
        torch.cat(all_preds).numpy()
    )

def measure_inference_latency(model, data_loader, device, num_inferences=100):
    """Measure average inference latency per sample in microseconds"""
    model.eval()
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= 10:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Actual measurement
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_inferences:
                break
            inputs = inputs.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            _ = model(inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            
            batch_size = inputs.size(0)
            latency_us = (end_time - start_time) * 1e6 / batch_size
            latencies.append(latency_us)
    
    return sum(latencies) / len(latencies)

if __name__ == "__main__":
    # Configuration settings with dynamic LR scheduler and early stopping
    config = {
        'seed': 42,  # Random seed for reproducibility
        'data_dir': '../../data',  # Preprocessed data directory
        'output_dir': '../../a-testcode/EEG-Transformer/results0710',  # Directory for output files
        'batch_size': 16,  # Batch size
        'num_epochs': 100,  # Total number of training epochs
        'lr': 0.001,  # Initial learning rate
        'weight_decay': 1e-4,  # L2 regularization
        'val_split': 0.15,  # Validation set ratio
        'test_split': 0.15,  # Test set ratio
        'emb_size': 40,  # Embedding dimension
        'depth': 4,  # Number of transformer layers
        'n_channels': 64,  # Number of EEG channels
        'n_samples': 2048,  # EEG segment length in samples
        
        # Early stopping configuration
        'early_stopping': {
            'patience': 5,      # Number of epochs to wait after last improvement
            'min_delta': 0.001  # Minimum change to qualify as an improvement
        },
        
        # Dynamic learning rate scheduler configuration
        'lr_scheduler': {
            'name': 'ReduceLROnPlateau',  # Options: 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealing', 'Exponential'
            'params': {
                'factor': 0.5,  # For ReduceLROnPlateau: factor by which the learning rate will be reduced
                'patience': 3,   # For ReduceLROnPlateau: number of epochs with no improvement
                'threshold': 0.001,  # For ReduceLROnPlateau: threshold for measuring the new optimum
                'min_lr': 1e-6,  # Minimum learning rate allowed
                
                # For StepLR:
                # 'step_size': 30,  # Period of learning rate decay
                # 'gamma': 0.1,     # Multiplicative factor of learning rate decay
                
                # For CosineAnnealing:
                # 'eta_min': 1e-6,  # Minimum learning rate
                
                # For Exponential:
                # 'gamma': 0.95,    # Multiplicative factor of learning rate decay
            }
        }
    }
    
    # Set CUDA visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Specify which GPUs to use
    
    # Check if distributed training is supported
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
    
    # Start training
    train_model(config)