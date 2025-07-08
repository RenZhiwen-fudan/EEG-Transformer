import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from scipy.interpolate import interp1d

class EEGDataset(Dataset):
    def __init__(self, data_dir, target_class="trueictal", transform=None, 
                 standard_channels=64, time_length=2048, 
                 target_sampling_rate=128):
        """
        加载预处理的EEG数据进行发作期检测
        :param data_dir: 包含预处理NPZ文件的目录
        :param target_class: 目标类别（默认trueictal）
        :param transform: 可选的数据转换
        :param standard_channels: 标准通道数（64）
        :param time_length: 时间序列长度（2048）
        :param target_sampling_rate: 目标采样率（128Hz）
        """
        self.data_paths = []
        self.labels = []
        self.standard_channels = standard_channels
        self.time_length = time_length
        self.target_sampling_rate = target_sampling_rate
        self.transform = transform
        
        # 根据图片信息组织的数据结构
        data_types = ["hup_ictal_win4x"] # "hup_interictal_win4"
        
        # 统计总文件数量
        total_files = 0
        for data_type in data_types:
            data_type_dir = os.path.join(data_dir, data_type)
            if not os.path.exists(data_type_dir):
                print(f"目录未找到: {data_type_dir}")
                continue
                
            # 遍历所有病人文件夹（如sub-HUP060）
            for patient_id in os.listdir(data_type_dir):
                patient_dir = os.path.join(data_type_dir, patient_id)
                if not os.path.isdir(patient_dir):
                    continue
                    
                # 添加所有NPZ文件
                file_count = len([f for f in os.listdir(patient_dir) if f.endswith(".npz")])
                total_files += file_count
                for fname in os.listdir(patient_dir):
                    if fname.endswith(".npz"):
                        file_path = os.path.join(patient_dir, fname)
                        self.data_paths.append(file_path)
        
        print(f"需处理文件总数: {total_files}")
        
        # 使用tqdm进度条加载标签
        progress_bar = tqdm(total=len(self.data_paths), desc="加载EEG标签", unit="文件")
        
        # 为每个文件确定标签
        for file_path in self.data_paths:
            fname = os.path.basename(file_path)
            
            # 从文件名确定标签（根据图片3中的命名规则）
            if "trueictal" in fname:
                label = "trueictal"
            elif "preictal" in fname:
                label = "preictal"
            elif "postictal" in fname:
                label = "postictal"
            elif "interictal" in fname:
                label = "interictal"
            else:
                # 默认设为interictal
                label = "interictal"
            
            # 转换为二分类标签
            binary_label = 1 if label == target_class else 0
            self.labels.append(binary_label)
            
            # 更新进度条
            progress_bar.update(1)
        
        progress_bar.close()
        print(f"已为{len(self.data_paths)}个EEG文件加载标签")
    
    def __len__(self):
        return len(self.data_paths)
    
    def _resample_to_target(self, eeg_data):
        """将EEG数据重采样到目标采样率（128Hz）"""
        original_sampling_rate = eeg_data.shape[1] / 16  # 假设原数据是4秒长度
        if original_sampling_rate == self.target_sampling_rate:
            return eeg_data
        
        resampled_length = int(eeg_data.shape[1] * self.target_sampling_rate / original_sampling_rate)
        resampled_data = np.zeros((eeg_data.shape[0], resampled_length))
        
        # 使用线性插值重采样
        original_timeline = np.linspace(0, 1, eeg_data.shape[1])
        new_timeline = np.linspace(0, 1, resampled_length)
        
        for channel_idx in range(eeg_data.shape[0]):
            interpolator = interp1d(original_timeline, eeg_data[channel_idx, :], kind='linear')
            resampled_data[channel_idx, :] = interpolator(new_timeline)
        
        return resampled_data
    
    def _adjust_channels(self, eeg_data):
        """调整通道数为64"""
        n_channels = eeg_data.shape[0]
        
        if n_channels == self.standard_channels:
            return eeg_data
        
        # 当通道数少于64时，使用插值补齐
        if n_channels < self.standard_channels:
            # 计算需要插入的通道数量
            new_channels = self.standard_channels - n_channels
            
            # 插值方法：使用临近通道的平均值创建新通道
            interpolated_data = np.zeros((self.standard_channels, eeg_data.shape[1]))
            
            # 复制已有的通道数据
            interpolated_data[:n_channels, :] = eeg_data
            
            # 为新增的通道创建数据（使用临近通道的平均值）
            for i in range(new_channels):
                # 选择最接近的新通道位置
                position = n_channels + i
                
                # 参考通道：前后各取两个通道（如果存在）
                ref_channels = []
                if position - 2 >= 0:
                    ref_channels.append(interpolated_data[position-2, :])
                if position - 1 >= 0:
                    ref_channels.append(interpolated_data[position-1, :])
                if position + 1 < self.standard_channels and (position + 1) < n_channels + i:
                    ref_channels.append(interpolated_data[position+1, :])
                if position + 2 < self.standard_channels and (position + 2) < n_channels + i:
                    ref_channels.append(interpolated_data[position+2, :])
                
                # 如果没有可用的参考通道，使用全局平均值
                if ref_channels:
                    interpolated_data[position, :] = np.mean(ref_channels, axis=0)
                else:
                    interpolated_data[position, :] = np.mean(eeg_data, axis=0)
            
            return interpolated_data
        
        # 当通道数多于64时，选择最相关或最优的64个通道
        if n_channels > self.standard_channels:
            # 策略1：根据信号能量选择
            channel_energies = np.sum(np.abs(eeg_data), axis=1)
            selected_indices = np.argsort(channel_energies)[-self.standard_channels:]
            
            # 策略2：均匀分布在各个脑区（如果需要更复杂的选择）
            # selected_indices = np.linspace(0, n_channels-1, self.standard_channels, dtype=np.int32)
            
            return eeg_data[selected_indices, :]
    
    def _crop_to_length(self, eeg_data):
        """调整时间序列长度为2048"""
        if eeg_data.shape[1] > self.time_length:
            # 随机裁剪
            start_idx = np.random.randint(0, eeg_data.shape[1] - self.time_length)
            return eeg_data[:, start_idx:start_idx + self.time_length]
        
        if eeg_data.shape[1] < self.time_length:
            # 零填充
            padding = np.zeros((eeg_data.shape[0], self.time_length - eeg_data.shape[1]))
            return np.concatenate([eeg_data, padding], axis=1)
        
        return eeg_data
    
    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        label = self.labels[idx]
        
        try:
            # 加载NPZ文件
            data_dict = np.load(file_path)
            eeg_data = data_dict["eeg_segment"]  # 原始形状: (n_channels, n_samples)
            
            # 确保时间序列长度合理
            if eeg_data.shape[1] < 128:
                raise ValueError(f"时间序列过短: {file_path}")
            
            # 重采样到128Hz
            eeg_data = self._resample_to_target(eeg_data)
            
            # 调整通道数为64
            eeg_data = self._adjust_channels(eeg_data)
            
            # 调整时间序列长度为2048 (16秒 * 128Hz = 2048)
            eeg_data = self._crop_to_length(eeg_data)
            
            # 转换为PyTorch格式 (1, 64, 2048) = (通道, 时间序列)
            eeg_data = np.expand_dims(eeg_data, axis=0)
            
            # 应用数据增强（如果定义）
            if self.transform:
                eeg_data = self.transform(eeg_data)
            
            return torch.tensor(eeg_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
            # 返回零张量避免训练中断
            dummy_data = torch.zeros((1, self.standard_channels, self.time_length), dtype=torch.float32)
            return dummy_data, torch.tensor(0, dtype=torch.long)

def create_data_loaders(data_dir, batch_size=32, val_split=0.1, test_split=0.1, 
                       standard_channels=64, time_length=2048, num_workers=1):
    """
    创建训练、验证和测试数据加载器
    :param data_dir: 预处理数据目录
    :param batch_size: 批量大小
    :param val_split: 验证集比例
    :param test_split: 测试集比例
    :param standard_channels: 标准通道数（64）
    :param time_length: 时间序列长度（2048）
    :param num_workers: DataLoader工作进程数
    :return: train_loader, val_loader, test_loader
    """
    print("创建数据集...")
    dataset = EEGDataset(data_dir, standard_channels=standard_channels, time_length=time_length)
    
    # 检查类别分布
    class_counts = np.bincount(dataset.labels)
    print(f"类别分布: Class 0 (其他): {class_counts[0]}, Class 1 (Trueictal): {class_counts[1]}")
    
    # 计算划分大小
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    
    # 分割数据集
    print("划分数据集...")
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器（使用多个工作进程加速）
    print("创建数据加载器...")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader