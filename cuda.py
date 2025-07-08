import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"计算能力: {torch.cuda.get_device_capability(0)}")