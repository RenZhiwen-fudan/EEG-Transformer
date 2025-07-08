import torch
import numpy as np
import os
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataloader import EEGDataset
from model import EEGTransformer
from utils import set_seed

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

def test_model(model_path, data_dir, config):
    """Test a trained model"""
    set_seed(42)  # For reproducibility
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = EEGDataset(data_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = EEGTransformer(
        emb_size=config['emb_size'],
        depth=config['depth'],
        n_classes=2,
        n_channels=config['n_channels'],
        n_samples=config['n_samples']
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Get predictions
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Calculate metrics
    acc = np.mean(np.array(all_labels) == np.array(all_preds))
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Print results
    print(f"\nTest Results (All Data):")
    print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['Other', 'Trueictal']))
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, ['Other', 'Trueictal'], 'all_data_confusion_matrix.png')

if __name__ == "__main__":
    # Configuration (should match training config)
    config = {
        'emb_size': 40,
        'depth': 4,
        'n_channels': 16,  # Update with your EEG channel count
        'n_samples': 512,   # Update with your segment length
    }
    
    # Path to trained model
    model_path = "./results/best_model.pth"  # Or your model path
    
    # Test the model
    test_model(model_path, '../data', config)