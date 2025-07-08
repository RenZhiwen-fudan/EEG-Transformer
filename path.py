import os
config = {
        'seed': 42,  # Random seed for reproducibility
        'data_dir': '../../a-testcode/EEG-Transformer',  # Preprocessed data directory
        'output_dir': './results',  # Directory for output files
        'batch_size': 32,
        'num_epochs': 50,
        'lr': 0.0001,  # Learning rate
        'weight_decay': 1e-4,  # L2 regularization
        'val_split': 0.15,  # Validation set ratio
        'test_split': 0.15,  # Test set ratio
        'emb_size': 40,  # Embedding dimension
        'depth': 4,  # Number of transformer layers
        'n_channels': 16,  # Number of EEG channels (adjust based on your data)
        'n_samples': 512,  # EEG segment length in samples (4 seconds * 128 Hz = 512)
    }
print("当前数据目录：", config['data_dir'])
print("目录是否存在：", os.path.exists(config['data_dir']))
print("目录内容示例：", os.listdir(config['data_dir']))