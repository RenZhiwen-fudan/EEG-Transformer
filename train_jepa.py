import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.distributed as dist
import csv
import random
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score
from torch.cuda.amp import autocast, GradScaler  # 添加混合精度支持

from dataloader import create_data_loaders
from jepa import EEGJEPA, apply_masks
from utils import set_seed, count_parameters

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

def generate_masks(num_patches, min_keep=4, max_keep=None, allow_overlap=True):
    """
    生成单个样本的掩码
    :param num_patches: 总patch数
    :return: masks_enc (context mask), masks_pred (target mask) 形状为 [num_patches]
    """
    if max_keep is None:
        max_keep = num_patches // 2
    
    # 随机确定上下文patch数量
    num_context = random.randint(min_keep, max_keep)
    
    # 创建所有patch的索引
    all_indices = np.arange(num_patches)
    
    # 选择上下文索引
    context_indices = np.random.choice(all_indices, size=num_context, replace=False)
    
    # 创建上下文掩码
    masks_enc = torch.zeros(num_patches, dtype=torch.bool)
    masks_enc[context_indices] = True
    
    # 确定目标索引
    if allow_overlap:
        # 允许重叠：目标可以包含上下文patch
        num_target = random.randint(min_keep, max_keep)
        target_indices = np.random.choice(all_indices, size=num_target, replace=False)
    else:
        # 不允许重叠：目标与上下文不相交
        non_context = np.setdiff1d(all_indices, context_indices)
        num_target = min(len(non_context), max_keep)
        target_indices = np.random.choice(non_context, size=num_target, replace=False)
    
    # 创建目标掩码
    masks_pred = torch.zeros(num_patches, dtype=torch.bool)
    masks_pred[target_indices] = True
    
    return masks_enc, masks_pred

class EEGClassifier(nn.Module):
    """EEG Classification Model with JEPA backbone"""
    def __init__(self, jepa_model, num_classes=2):
        super().__init__()
        self.jepa = jepa_model
        
        # 获取嵌入维度 - 修复版本
        if hasattr(self.jepa, 'module'):
            # 处理 DataParallel 包装
            encoder = self.jepa.module.encoder
        else:
            encoder = self.jepa.encoder
        
        # 检查编码器是否有 embed_dim 属性
        if hasattr(encoder, 'embed_dim'):
            embed_dim = encoder.embed_dim
        # 如果没有，尝试从配置中获取
        elif hasattr(encoder, 'config') and 'embed_dim' in encoder.config:
            embed_dim = encoder.config['embed_dim']
        # 如果还没有，尝试从第一个线性层推断
        elif hasattr(encoder, 'patch_embed') and hasattr(encoder.patch_embed, 'proj'):
            embed_dim = encoder.patch_embed.proj.out_channels
        else:
            # 最后的手段：使用默认值或抛出错误
            embed_dim = 128  # 使用合理的默认值
            print(f"Warning: Could not determine embed_dim, using default {embed_dim}")
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # 正确处理 DataParallel 包装
        if hasattr(self.jepa, 'module'):
            # 处理 DataParallel 包装
            features = self.jepa.module.encoder(x)
        else:
            # 单 GPU 情况
            features = self.jepa.encoder(x)
        
        # Global average pooling
        features = features.mean(dim=1)
        
        # Classification
        return self.classifier(features)

def train_jepa(config):
    """Main training function for EEG-JEPA model"""
    # Set random seeds for reproducibility
    set_seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create CSV file for saving epoch metrics
    metrics_file = os.path.join(config['output_dir'], 'jepa_metrics.txt')
    with open(metrics_file, 'w', newline='') as f:
        header = [
            'Epoch', 'Train Loss', 'Val Loss', 'Test Loss', 
            'Params LT (µs)', 'Epoch Time (s)', 'Learning Rate', 'Early Stopping'
        ]
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
    
    # Save config for future reference
    with open(os.path.join(config['output_dir'], 'jepa_config.txt'), 'w') as f:
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
    
    model = EEGJEPA(
        encoder_config={
            'in_chans': config['jepa']['in_chans'],
            'embed_dim': config['jepa']['embed_dim'],
            'depth': config['jepa']['encoder_depth'],
            'num_heads': config['jepa']['encoder_heads'],
            'mlp_ratio': config['jepa']['mlp_ratio'],
            'patch_size': config['jepa']['patch_size'],
            'qkv_bias': config['jepa']['qkv_bias'],
            'drop_rate': config['jepa']['drop_rate'],
            'attn_drop_rate': config['jepa']['attn_drop_rate'],
            'drop_path_rate': config['jepa']['drop_path_rate']
        },
        predictor_config={
            'embed_dim': config['jepa']['embed_dim'],
            'predictor_embed_dim': config['jepa']['predictor_embed_dim'],
            'depth': config['jepa']['predictor_depth'],
            'num_heads': config['jepa']['predictor_heads'],
            'mlp_ratio': config['jepa']['mlp_ratio'],
            'qkv_bias': config['jepa']['qkv_bias'],
            'drop_rate': config['jepa']['drop_rate'],
            'attn_drop_rate': config['jepa']['attn_drop_rate'],
            'drop_path_rate': config['jepa']['drop_path_rate']
        }
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
    
    # Loss function for JEPA (regression loss)
    criterion = nn.SmoothL1Loss()
    
    # Training variables
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    test_losses = []
    params_latencies = []
    learning_rates = []
    
    # Early stopping variables
    early_stop_counter = 0
    min_d极 = config.get('early_stopping', {}).get('min_delta', 0.001)
    patience = config.get('early_stopping', {}).get('patience', 5)
    early_stop_triggered = False
    
    # Calculate number of patches
    num_patches = config['n_channels'] * (config['n_samples'] // config['patch_size'])
    print(f"Number of patches per sample: {num_patches}")
    
    # 添加梯度累积参数
    accumulation_steps = config.get('accumulation_steps', 1)
    print极(f"Using gradient accumulation with {accumulation_steps} steps")
    
    # 添加混合精度训练
    use_amp = config.get('use_amp', True)
    scaler = GradScaler(enabled=use_amp)
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")
    
    # 断点恢复设置
    checkpoint_file = os.path.join(config['output_dir'], 'jepa_checkpoint.pth')
    start_epoch = 0
    
    # 检查是否有检查点文件
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        
        # 加载模型状态
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态（如果存在）
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载训练状态
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        test_losses = checkpoint['test_losses']
        learning_rates = checkpoint['learning_rates']
        best_val_loss = checkpoint['best_val_loss']
        best_epoch = checkpoint['best_epoch']
        early_stop_counter = checkpoint['early_stop_counter']
        early_stop_triggered = checkpoint['early_stop_triggered']
        
        print(f"Loaded checkpoint from epoch {start_epoch}. Resuming training...")
    else:
        print("No checkpoint found. Starting new training.")
    
    # Start training
    print("\nStarting JEPA pre-training...")
    print(f"Early stopping configured: patience={patience}, min_delta={min_delta:.4f}")
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Training loop with gradient accumulation
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for i, (inputs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")):
            inputs = inputs.to(device)
            
            # 为每个样本生成独立的掩码
            batch_size = inputs.size(0)
            masks_enc_list = []
            masks_pred_list = []
            
            for i in range(batch_size):
                masks_enc, masks_pred = generate_masks(
                    num_patches,
                    min_keep=config['mask']['min_keep'],
                    max_keep=config['mask']['max_keep'],
                    allow_overlap=config['mask']['allow_overlap']
                )
                masks_enc_list.append(masks_enc)
                masks_pred_list.append(masks_pred)
            
            # 将掩码堆叠成张量
            masks_enc = torch.stack(masks_enc_list).to(device)
            masks_pred = torch.stack(masks_pred_list).to(device)
            
            # 使用混合精度训练
            with autocast(enabled=use_amp):
                # Forward pass
                pred, target = model(inputs, masks_enc, masks_pred)
                
                # Compute loss
                loss = criterion(pred, target)
                loss = loss / accumulation_steps  # 归一化损失
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 累积损失
            running_loss += loss.item() * accumulation_steps
            
            # 每accumulation_steps步更新一次参数
            if (i + 1) % accumulation_steps == 0:
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # 处理剩余的梯度（如果总步数不是accumulation_steps的整数倍）
        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Training metrics
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = evaluate_jepa(model, val_loader, criterion, device, num_patches, config['mask'])
        val_losses.append(val_loss)
        
        # Update scheduler based on validation loss
        if config['lr_scheduler']['name'] == 'ReduceLROnPlateau':
            scheduler.step(val_loss)  # Use validation loss for plateau scheduler
        else:
            scheduler.step()  # Step-based schedulers update after every epoch
        
        # Evaluate on test set after each epoch
        test_loss = evaluate_jepa(model, test_loader, criterion, device, num_patches, config['mask'])
        test_losses.append(test_loss)
        
        # Measure inference latency
        latency = measure_inference_latency(model, test_loader, device, num_patches, config['mask'])
        params_latencies.append(latency)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            # Check improvement for early stopping
            improvement = best_val_loss - val_loss
            if improvement > min_delta:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                early_stop_counter = 0  # Reset counter on improvement
                
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'test_loss': test_loss
                }, os.path.join(config['output_dir'], 'best_jepa_model.pth'))
                    
                print(f"Saved new best JEPA model at epoch {epoch+1} with Val Loss: {val_loss:.4f} (↓{improvement:.4f})")
            else:
                print(f"No significant improvement: {improvement:.4f} < min_delta ({min_delta:.4f})")
        else:
            # No improvement
            improvement = val_loss - best_val_loss
            early_stop_counter += 1
            print(f"No improvement in validation loss. Counter: {early_stop_counter}/{patience}")
            
            # Check early stopping condition
            if early_stop_counter >= patience and not early_stop_triggered:
                early_stop_triggered = True
                print(f"⚠️ Early stopping triggered at epoch {epoch+1}! "
                      f"No improvement for {patience} consecutive epochs.")
        
        epoch_time = time.time() - epoch_start
        
        # Save epoch metrics to TXT file
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\极')
            writer.writerow([
                epoch+1,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{test_loss:.6f}",
                f"{latency:.6f}",
                f"{epoch_time:.2f}",
                f"{current_lr:.8f}",
                "True" if early_stop_triggered else "False"
            ])
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - {epoch_time:.1f}s")
        print(f"LR: {current_lr:.8f} | Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f} | Latency: {latency:.2f} µs")
        print("-" * 80)
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'learning_rates': learning_rates,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'early_stop_counter': early_stop_counter,
            'early_stop_triggered': early_stop_triggered
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Saved checkpoint for epoch {epoch+1}")
        
        # Break if early stopping is triggered
        if early_stop_triggered:
            print("⏹ JEPA pre-training stopped early due to validation performance.")
            break
    
    # 训练完成后删除检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Deleted checkpoint file: {checkpoint_file}")
    
    # Load best model
    best_model_path = os.path.join(config['output_dir'], 'best_jepa_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded best JEPA model from epoch {checkpoint['epoch']}")
    else:
        print("No best JEPA model found, using final model")
        checkpoint = {'epoch': config['num_epochs']}
    
    # Save the final JEPA model
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': len(train_losses),
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'test_loss': test_loss,
        'early_stop': early_stop_triggered
    }, os.path.join(config['output_dir'], 'final_jepa_model.pth'))
    
    print("\nJEPA pre-training completed.")
    
    # Return the pre-trained model for classification fine-tuning
    return model

def evaluate_jepa(model, data_loader, criterion, device, num_patches, mask_config):
    """Evaluate JEPA model on a dataset"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # 为每个样本生成独立的掩码
            masks_enc_list = []
            masks_pred_list = []
            for i in range(batch_size):
                masks_enc, masks_pred = generate_masks(
                    num_patches,
                    min_keep=mask_config['min_keep'],
                    max_keep=mask_config['max_keep'],
                    allow_overlap=mask_config['allow_overlap']
                )
                masks_enc_list.append(masks_enc)
                masks_pred_list.append(masks_pred)
            
            # 堆叠掩码
            masks_enc = torch.stack(masks_enc_list).to(device)
            masks_pred = torch.stack(masks_pred_list).to(device)
            
            # Forward pass
            pred, target = model(inputs, masks_enc, masks_pred)
            
            # Compute loss
            loss = criterion(pred, target)
            running_loss += loss.item()
    
    return running_loss / len(data_loader)

def measure_inference_latency(model, data_loader, device, num_patches, mask_config, num_inferences=100):
    """Measure average inference latency per sample in microseconds"""
    model.eval()
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= 10:
                break
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # 为每个样本生成掩码
            masks_enc_list = []
            masks_pred_list = []
            for i in range(batch_size):
                masks_enc, masks_pred = generate_masks(
                    num_patches,
                    min_keep=mask_config['min_keep'],
                    max_keep=mask_config['max_keep'],
                    allow_overlap=mask_config['allow_overlap']
                )
                masks_enc_list.append(masks_enc)
                masks_pred_list.append(masks_pred)
            
            masks_enc = torch.stack(masks_enc_list).to(device)
            masks_pred = torch.stack(masks_pred_list).极(device)
            _ = model(inputs, masks_enc, masks_pred)
    
    # Actual measurement
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_inferences:
                break
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # 为每个样本生成掩码
            masks_enc_list = []
            masks_pred_list = []
            for i in range(batch_size):
                masks_enc, masks_pred = generate_masks(
                    num_patches,
                    min_keep=mask_config['min_keep'],
                    max_keep=mask_config['max_keep'],
                    allow_overlap=mask_config['allow_overlap']
                )
                masks_enc_list.append(masks_enc)
                masks_pred_list.append(masks_pred)
            
            masks_enc = torch.stack(masks_enc_list).to(device)
            masks_pred = torch.stack(masks_pred_list).to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            _ = model(inputs, masks_enc, masks_pred)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            
            latency_us = (end_time - start_time) * 1e6 / batch_size
            latencies.append(latency_us)
    
    return sum(latencies) / len(latencies)

def train_classifier(config, jepa_model):
    """Fine-tune classifier on pre-trained JEPA model"""
    # Set random seeds for reproducibility
    set_seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create CSV file for saving epoch metrics
    metrics_file = os.path.join(config['output_dir'], 'classifier_metrics.txt')
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
    with open(os.path.join(config['output_dir'], 'classifier_config.txt'), 'w') as f:
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
    
    # Initialize classifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available")
    
    model = EEGClassifier(jepa_model)
    
    # Use DataParallel for multi-GPU training
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Classifier parameters: {num_params}")
    print(f"Using device: {device}")
    
    # Optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['classifier_lr'],
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
    
    # 添加梯度累积参数
    accumulation_steps = config.get('accumulation_steps', 1)
    print(f"Using gradient accumulation with {accumulation_steps} steps")
    
    # 添加混合精度训练
    use_amp = config.get('use_amp', True)
    scaler = GradScaler(enabled=use_amp)
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")
    
    # 断点恢复设置
    checkpoint_file = os.path.join(config['output_dir'], 'classifier_checkpoint.pth')
    start_epoch = 0
    
    # 检查是否有检查点文件
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        
        # 加载模型状态
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['极odel_state_dict'])
        
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态（如果存在）
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载训练状态
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        test_losses = checkpoint['test_losses']
        train_accs = checkpoint['train_accs']
        val_accs = checkpoint['val_accs']
        test_accs = checkpoint['test_accs']
        test_f1s = checkpoint['test_f1s']
        test_aurocs = checkpoint['test_aurocs']
        learning_rates = checkpoint['learning_rates']
        best_val_f1 = checkpoint['best_val_f1']
        best_epoch = checkpoint['best_epoch']
        early_stop_counter = checkpoint['early_stop_counter']
        early_stop_triggered = checkpoint['early_stop_triggered']
        
        print(f"Loaded checkpoint from epoch {start_epoch}. Resuming training...")
    else:
        print("No checkpoint found. Starting new training.")
    
    # Start training
    print("\nStarting classifier fine-tuning...")
    print(f"Early stopping configured: patience={patience}, min_delta={min_delta:.4f}")
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Training loop with gradient accumulation
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 使用混合精度训练
            with autocast(enabled=use_amp):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # 归一化损失
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 累积损失和准确率
            running_loss += loss.item() * accumulation_steps
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每accumulation_steps步更新一次参数
            if (i + 1) % accumulation_steps == 0:
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # 处理剩余的梯度（如果总步数不是accumulation_steps的整数倍）
        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        val_loss, val_acc, val_f1, val_aucroc = evaluate_classifier(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update scheduler based on validation metric
        if config['lr_scheduler']['name'] == 'ReduceLROnPlateau':
            scheduler.step(val_f1)  # Use validation F1 for plateau scheduler
        else:
            scheduler.step()  # Step-based schedulers update after every epoch
        
        # Evaluate on test set after each epoch
        test_loss, test_acc, test_f1, test_aucroc = evaluate_classifier(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_aurocs.append(test_aucroc)
        
        # Measure inference latency
        latency = measure_classifier_latency(model, test_loader, device)
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
                }, os.path.join(config['output_dir'], 'best_classifier.pth'))
                    
                print(f"Saved new best classifier at epoch {epoch+1} with Val F1: {val_f1:.4f} (↑{improvement:.4f})")
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
                f"{latency:.6极}",
                f"{epoch_time:.2f}",
                f"{current_lr:.8f}",
                "True" if early_stop_triggered else "False"
            ])
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - {epoch_time:.1f}s")
        print(f"LR: {current_lr:.8f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test AUCROC: {test_aucroc:.4f} | Latency: {latency:.2f} µs")
        print("-" * 80)
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_accs': test_accs,
            'test_f1s': test_f1s,
            'test_aurocs': test_aurocs,
            'learning_rates': learning_rates,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'early_stop_counter': early_stop_counter,
            'early_stop_triggered': early_stop_triggered
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Saved checkpoint for epoch {epoch+1}")
        
        # Break if early stopping is triggered
        if early_stop_triggered:
            print("⏹ Classifier training stopped early due to validation performance.")
            break
    
    # 训练完成后删除检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Deleted checkpoint file: {checkpoint_file}")
    
    # Load best model
    best_model_path = os.path.join(config['output_dir'], 'best_classifier.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded best classifier from epoch {checkpoint['epoch']} for final testing")
    else:
        print("No best classifier found, using final model")
        checkpoint = {'epoch': config['num_epochs']}
    
    # Final evaluation on test set
    test_loss, test_acc, test_f1, test_aucroc = evaluate_classifier(model, test_loader, criterion, device)
    latency = measure_classifier_latency(model, test_loader, device)
    
    # Get predictions for confusion matrix
    y_true, y_pred = predict_classifier(model, test_loader, device)
    
    print(f"\nFinal Test Results (Best Classifier Epoch {checkpoint['epoch']}):")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1 Score: {test_f1:.4f} | AUCROC: {test_aucroc:.4f} | Latency: {latency:.2f} µs")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['non-ictal', 'trueictal']))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, ['non-ictal', 'trueictal'], 
                         os.path.join(config['output_dir'], 'confusion_matrix.png'))
    
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
    metrics_txt_file = os.path.join(config['output_dir'], 'classifier_final_metrics.txt')
    with open(metrics_txt_file, 'w') as f:
        for key, value in metrics.items():
            if isinstance(value, list):
                f.write(f"{key}: {','.join(map(str, value))}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\nConfig Settings:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
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
    plt.savefig(os.path.join(config['output_dir'], 'classifier_training_curves.png'))
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
    plt.savefig(os.path.join(config['output_dir'], 'classifier_performance_curves.png'))
    plt.close()
    
    # Save the final classifier
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
    }, os.path.join(config['output_dir'], 'final_classifier.pth'))
    
    print("\nClassifier training completed.")

def evaluate_classifier(model, data_loader, criterion, device):
    """Evaluate classifier model on a dataset"""
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

def predict_classifier(model, data_loader, device):
    """Get predictions from classifier model"""
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

def measure_classifier_latency(model, data_loader, device, num_inferences=100):
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
    plt.save极(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")

def load_pretrained_jepa(config):
    """Load pre-trained JEPA model from file"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = EEGJEPA(
        encoder_config={
            'in_chans': config['jepa']['in_chans'],
            'embed_dim': config['jepa']['embed_dim'],
            'depth': config['jepa']['encoder_depth'],
            'num_heads': config['jepa']['encoder_heads'],
            'mlp_ratio': config['jepa']['mlp_ratio'],
            'patch_size': config['jepa']['patch_size'],
            'qkv_bias': config['jepa']['qkv_bias'],
            'drop_rate': config['jepa']['drop_rate'],
            'attn_drop_rate': config['jepa']['attn_drop_rate'],
            'drop_path_rate': config['jepa']['drop_path_rate']
        },
        predictor_config={
            'embed_dim': config['jepa']['embed_dim'],
            'predictor_embed_dim': config['jepa']['predictor_embed_dim'],
            'depth': config['jepa']['predictor_depth'],
            'num_heads': config['jepa']['predictor_heads'],
            'mlp_ratio': config['jepa']['mlp_ratio'],
            'qkv_bias': config['jepa']['qkv_bias'],
            'drop_rate': config['jepa']['drop_rate'],
            'attn_drop_rate': config['jepa']['attn_drop_rate'],
            'drop_path_rate': config['jepa']['drop_path_rate']
        }
    )
    
    # Load model state
    model_path = os.path.join(config['output_dir'], 'final_jepa_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle DataParallel wrapping
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs with DataParallel")
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        # Load state dict
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded pre-trained JEPA model from {model_path}")
        return model
    else:
        print(f"No pre-trained JEPA model found at {model_path}")
        return None

if __name__ == "__main__":
    # Configuration settings for two-stage training
    config = {
        'seed': 42,  # Random seed for reproducibility
        'data_dir': '../../data',  # Preprocessed data directory
        'output_dir': '../../a-testcode/EEG-Transformer/results_jepa0711-2',  # Directory for output files
        'batch_size': 8,  # Batch size
        'num_epochs': 100,  # Total number of training epochs for each stage
        'lr': 0.001,  # Initial learning rate for JEPA pre-training
        'classifier_lr': 0.0001,  # Learning rate for classifier fine-tuning
        'weight_decay': 1e-4,  # L2 regularization
        'val_split': 0.15,  # Validation set ratio
        'test_split': 0.15,  # Test set ratio
        'n_channels': 64,  # Number of EEG channels
        'n_samples': 2048,  # EEG segment length in samples
        'patch_size': 32,  # Patch size for EEG segmentation
        'accumulation_steps': 4,  # 梯度累积步数
        'use_amp': True, # 使用混合精度训练
        
        # JEPA model configuration
        'jepa': {
            'in_chans': 1,  # Input channels (EEG data)
            'embed_dim': 128,  # Encoder embedding dimension
            'encoder_depth': 4, #  Number of encoder layers
            'encoder_heads': 8,  # Number of attention heads in encoder
            'predictor_embed_dim': 128,  # Predictor embedding dimension
            'predictor_depth': 4,  # Number of predictor layers
            'predictor_heads': 4,  # Number of attention heads in predictor
            'mlp_ratio': 4,  # MLP expansion ratio
            'patch_size': 32,  # Patch size for EEG segmentation
            'qkv_bias': True,  # Use bias in QKV projections
            'drop_rate': 0.1,  # Dropout rate
            'attn_drop_rate': 0.1,  # Attention dropout rate
            'drop_path_rate': 0.1  # Stochastic depth rate
        },
        
        # Mask configuration for JEPA pre-training
        'mask': {
            'min_keep': 10,  # Minimum number of patches to keep in context
            'max_keep': 100,  # Maximum number of patches to keep in context
            'allow_overlap': True,  # Allow overlap between context and target
        },
        
        # Early stopping configuration
        'early_stopping': {
            'patience': 5,      # Number of epochs to wait after last improvement
            'min_delta': 0.001  # Minimum change to qualify as an improvement
        },
        
        # Learning rate scheduler configuration
        'lr_scheduler': {
            'name': 'ReduceLROnPlateau',  # Options: 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealing'
            'params': {
                'factor': 0.5,  # For ReduceLROnPlateau: factor by which the learning rate will be reduced
                'patience': 3,   # For ReduceLROnPlateau: number of epochs with no improvement
                'threshold': 0.001,  # For ReduceLROnPlateau: threshold for measuring the new optimum
                'min_lr': 1e-6,  # Minimum learning rate allowed
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
    
    # 检查是否存在预训练的JEPA模型
    jepa_model = load_pretrained_jepa(config)
    
    if jepa_model is None:
        # Stage 1: JEPA pre-training
        print("\n" + "="*50)
        print("Starting Stage 1: JEPA Pre-training")
        print("="*50)
        jepa_model = train_jepa(config)
    else:
        print("\n" + "="*50)
        print("Found pre-trained JEPA model. Skipping pre-training.")
        print("="*50)
    
    # Stage 2: Classifier fine-tuning
    print("\n" + "="*50)
    print("Starting Stage 2: Classifier Fine-tuning")
    print("="*50)
    train_classifier(config, jepa_model)
    
    print("\nTwo-stage training completed successfully!")