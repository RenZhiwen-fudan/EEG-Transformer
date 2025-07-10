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

def train_model(config):
    """Main training function"""
    # Set random seeds for reproducibility
    set_seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create CSV file for saving epoch metrics
    metrics_file = os.path.join(config['output_dir'], 'epoch_metrics.txt')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'Epoch', 'Train Loss', 'Val Loss', 'Test Loss', 
            'Train Acc', 'Val Acc', 'Test Acc', 
            'Test F1', 'Test AUCROC', 'Params LT (µs)', 
            'Epoch Time (s)'
        ])
    
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
    
    # 使用DataParallel包装模型以支持多GPU
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
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
    test_losses = []  # 新增：存储每个epoch的测试集损失
    train_accs = []
    val_accs = []
    test_accs = []  # 新增：存储每个epoch的测试集准确率
    test_f1s = []   # 新增：存储每个epoch的测试集F1分数
    test_aurocs = [] # 新增：存储每个epoch的测试集AUCROC
    params_latencies = [] # 新增：存储每个epoch的参数延迟
    
    # Precalculate model parameters (will be constant)
    num_params = count_parameters(model)
    
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
        
        # Evaluate on test set after each epoch
        test_loss, test_acc, test_f1, test_aucroc = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_aurocs.append(test_aucroc)
        
        # Measure inference latency
        latency = measure_inference_latency(model, test_loader, device, num_inferences=100)
        params_latencies.append(latency)
        
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | "
              f"Test AUCROC: {test_aucroc:.4f} | Latency: {latency:.2f} µs")
        
        # Save best model based on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
            # 关键修改：保存模型时处理DataParallel
            if isinstance(model, nn.DataParallel):
                # 如果是多GPU模型，只保存module部分
                torch.save(model.module.state_dict(), os.path.join(config['output_dir'], 'best_model.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(config['output_dir'], 'best_model.pth'))
                
            print(f"Saved new best model with Val F1: {val_f1:.4f}")
        
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
                f"{epoch_time:.2f}"
            ])
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test AUCROC: {test_aucroc:.4f} | Latency: {latency:.2f} µs")
    
    # Load best model
    best_model_path = os.path.join(config['output_dir'], 'best_model.pth')
    if isinstance(model, nn.DataParallel):
        # 如果是多GPU模型，加载到module中
        model.module.load_state_dict(torch.load(best_model_path))
    else:
        model.load_state_dict(torch.load(best_model_path))
    
    print("Loaded best model for final testing")
    
    # Final evaluation on test set
    test_loss, test_acc, test_f1, test_aucroc = evaluate_model(model, test_loader, criterion, device)
    latency = measure_inference_latency(model, test_loader, device)
    print(f"\nFinal Test Results:")
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
        'num_params': num_params,
        'num_gpus_used': num_gpus  # 记录使用的GPU数量
    }
    
    # 保存为txt而不是pth
    metrics_txt_file = os.path.join(config['output_dir'], 'final_metrics.txt')
    with open(metrics_txt_file, 'w') as f:
        for key, value in metrics.items():
            if isinstance(value, list):
                f.write(f"{key}: {','.join(map(str, value))}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # Plot confusion matrix for test set
    print("\nTest set performance:")
    y_true, y_pred = predict_model(model, test_loader, device)
    print(classification_report(y_true, y_pred, target_names=['Other', 'Trueictal']))
    plot_confusion_matrix(y_true, y_pred, ['Other', 'Trueictal'], 
                         os.path.join(config['output_dir'], 'confusion_matrix.png'))
    
    # Plot training curves with test performance
    plt.figure(figsize=(15, 8))
    
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
    
    # F1 Score Curve
    plt.subplot(2, 3, 3)
    plt.plot(test_f1s, label='Test F1', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Test F1 Score Curve')
    
    # AUCROC Curve
    plt.subplot(2, 3, 4)
    plt.plot(test_aurocs, label='Test AUCROC', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('AUCROC')
    plt.legend()
    plt.title('Test AUCROC Curve')
    
    # Latency Curve
    plt.subplot(2, 3, 5)
    plt.plot(params_latencies, label='Inference Latency', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Latency (µs)')
    plt.legend()
    plt.title('Inference Latency')
    
    # Parameters
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.5, f"Model Parameters: {num_params}\nGPUs Used: {num_gpus}", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    plt.title('Model Information')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'training_curves.png'))
    plt.close()
    
    # Save the final model
    if isinstance(model, nn.DataParallel):
        # 如果是多GPU模型，只保存module部分
        torch.save(model.module.state_dict(), os.path.join(config['output_dir'], 'final_model.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(config['output_dir'], 'final_model.pth'))
    
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
            
            # 获取预测概率
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # 收集预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 确保张量正确转换
            all_labels.append(labels.cpu())  # 先移动回CPU
            all_probs.append(probs.cpu())
    
    # 统一转换为NumPy数组
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), average='weighted')
    
    # 计算AUC ROC
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
            
            # 先收集CPU张量
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    
    # 转换为NumPy数组
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
            if i >= 10:  # 预热10个batch
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # 实际测量
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_inferences:
                break
            inputs = inputs.to(device)
            
            # 确保异步CUDA操作完成
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            _ = model(inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            
            # 计算每个样本的延迟（转换为微秒）
            batch_size = inputs.size(0)
            latency_us = (end_time - start_time) * 1e6 / batch_size
            latencies.append(latency_us)
    
    # 如果有多个GPU，平均延迟
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        avg_latency = sum(latencies) / len(latencies)
        avg_tensor = torch.tensor(avg_latency).to(device)
        dist.reduce(avg_tensor, dst=0, op=dist.ReduceOp.SUM)
        avg_latency = avg_tensor.item() / dist.get_world_size()
    else:
        avg_latency = sum(latencies) / len(latencies)
    
    return avg_latency

if __name__ == "__main__":
    # Configuration settings
    config = {
        'seed': 42,  # Random seed for reproducibility
        'data_dir': '../../data',  # Preprocessed data directory
        'output_dir': '../../a-testcode/EEG-Transformer/results0710',  # Directory for output files
        'batch_size': 16,  # 增加批次大小以充分利用多GPU
        'num_epochs': 50,
        'lr': 0.0001,  # Learning rate
        'weight_decay': 1e-4,  # L2 regularization
        'val_split': 0.15,  # Validation set ratio
        'test_split': 0.15,  # Test set ratio
        'emb_size': 40,  # Embedding dimension
        'depth': 4,  # Number of transformer layers
        'n_channels': 64,  # Number of EEG channels (adjust based on your data)
        'n_samples': 2048,  # EEG segment length in samples (4 seconds * 128 Hz = 512)
    }
    
    # 设置CUDA环境变量，确保所有GPU可见
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定使用哪些GPU
    
    # 检查是否支持分布式训练
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
    
    # Start training
    train_model(config)