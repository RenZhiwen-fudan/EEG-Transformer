import os
import mne
from tqdm import tqdm
from collections import Counter
import numpy as np

def print_edf_channels(dataset_dir):
    """
    遍历数据集目录，打印每个EDF文件的通道数，并统计公共出现的通道
    
    :param dataset_dir: 包含EDF文件的根目录
    """
    # 收集所有EDF文件路径
    edf_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith('.edf'):
                edf_files.append(os.path.join(root, file))
    
    print(f"在 {dataset_dir} 中找到 {len(edf_files)} 个EDF文件")
    
    if not edf_files:
        print("未找到任何EDF文件")
        return
    
    # 初始化通道统计数据结构
    channel_counts = {}              # 每种通道数量的文件数
    all_channels_counter = Counter()  # 所有通道的计数器
    channel_presence_counter = Counter()  # 通道出现频率计数器
    channel_sets = []                 # 每个文件的通道集合
    
    # 通道类型统计
    channel_types_instances = Counter()  # 每种类型的实例数
    channel_types_files = Counter()      # 每种类型出现在多少文件中
    type_mapping = {}                   # 通道名称到类型的映射
    unique_channels = set()              # 所有唯一的通道名称
    
    # 遍历所有EDF文件并获取通道信息
    for file_path in tqdm(edf_files, desc="处理EDF文件"):
        try:
            # 读取EDF文件头信息（不加载实际数据）
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            
            # 获取通道名称并转换为集合
            ch_names = raw.ch_names
            num_channels = len(ch_names)
            
            # 更新通道数量统计
            channel_counts[num_channels] = channel_counts.get(num_channels, 0) + 1
            
            # 更新通道出现次数计数器
            all_channels_counter.update(ch_names)
            
            # 将当前文件的通道名称添加到集合列表
            channel_set = set(ch_names)
            channel_sets.append(channel_set)
            
            # 添加到唯一通道集合
            unique_channels.update(ch_names)
            
            # 为通道存在性计数
            channel_presence_counter.update(channel_set)
            
            # 使用MNE获取通道类型
            file_channel_types = {}
            for ch in ch_names:
                try:
                    # 获取通道类型
                    ch_type = mne.io.pick.channel_type(raw.info, ch)
                    
                    # 如果是未知类型，尝试猜测
                    if ch_type == "misc":
                        ch_type = mne.io.pick._picks_by_type(raw.info, None)[0]
                    
                    # 存储映射关系
                    type_mapping[ch] = ch_type
                    
                    # 更新实例计数
                    channel_types_instances[ch_type] += 1
                    
                    # 标记此类型已在本文件中出现
                    file_channel_types[ch_type] = True
                except Exception as e:
                    print(f"无法确定通道 '{ch}' 的类型: {str(e)}")
                    channel_types_instances["unknown"] += 1
                    file_channel_types["unknown"] = True
            
            # 更新文件级类型计数
            for ch_type in file_channel_types:
                channel_types_files[ch_type] += 1
            
            # 打印文件信息
            print(f"\n文件: {os.path.basename(file_path)}")
            print(f"路径: {file_path}")
            print(f"通道数: {num_channels}")
            print(f"通道名称: {', '.join(ch_names[:5])}{'...' if len(ch_names) > 5 else ''}")
            
        except Exception as e:
            print(f"\n处理文件 {file_path} 时出错: {str(e)}")
    
    # ===== 1. 通道数量统计 =====
    print("\n===== 通道数量统计 =====")
    for count, freq in sorted(channel_counts.items()):
        percentage = freq / len(edf_files) * 100
        print(f"{count} 通道的文件数: {freq} ({percentage:.1f}%)")
    
    # 打印最常见的通道配置
    if channel_counts:
        most_common = max(channel_counts.items(), key=lambda x: x[1])
        print(f"\n最常见的通道数: {most_common[0]} (出现 {most_common[1]} 次)")
    
    # ===== 2. 公共通道统计 =====
    # 计算所有文件中都出现的公共通道
    common_channels = set.intersection(*channel_sets) if channel_sets else set()
    print("\n===== 公共通道统计 =====")
    print(f"出现在所有 {len(edf_files)} 个文件中的公共通道 ({len(common_channels)} 个):")
    if common_channels:
        common_channels_info = []
        for ch in sorted(common_channels):
            ch_type = type_mapping.get(ch, "unknown")
            common_channels_info.append(f"{ch} ({ch_type})")
        print(", ".join(common_channels_info))
    else:
        print("无")
    
    # ===== 3. 通道出现频率统计 =====
    print("\n===== 通道出现频率 =====")
    total_files = len(edf_files)
    
    # 打印出现频率最高的通道
    top_10_common = channel_presence_counter.most_common(10)
    print("\n最常见的10个通道:")
    for i, (channel, count) in enumerate(top_10_common):
        ch_type = type_mapping.get(channel, "unknown")
        percentage = count / total_files * 100
        print(f"{i+1}. {channel} ({ch_type}): {count} 文件 ({percentage:.1f}%)")
    
    # 打印出现频率最低的通道
    print("\n最不常见的10个通道:")
    sorted_least_common = sorted(channel_presence_counter.items(), key=lambda x: x[1])
    for i, (channel, count) in enumerate(sorted_least_common[:10]):
        ch_type = type_mapping.get(channel, "unknown")
        percentage = count / total_files * 100
        print(f"{i+1}. {channel} ({ch_type}): {count} 文件 ({percentage:.1f}%)")
    
    # ===== 4. 所有通道名称（按字母顺序） =====
    print("\n===== 所有出现的通道名称 (按字母顺序) =====")
    print(f"总共 {len(unique_channels)} 个唯一的通道名称:")
    
    # 按字母顺序排序
    sorted_channel_names = sorted(unique_channels)
    
    # 分组打印，每行10个
    print("\n通道名称列表:")
    for i in range(0, len(sorted_channel_names), 10):
        print(", ".join(sorted_channel_names[i:i+10]))
    
    # # ===== 5. 所有通道及其频率 =====
    # print("\n===== 所有通道及其频率 (按字母排序) =====")
    # for channel, count in sorted(channel_presence_counter.items()):
    #     ch_type = type_mapping.get(channel, "unknown")
    #     percentage = count / total_files * 100
    #     print(f"{channel} ({ch_type}): {count} 文件 ({percentage:.1f}%)")
    
    # # ===== 6. 使用MNE的通道类型统计 =====
    # print("\n===== MNE通道类型统计 =====")
    # print("\n通道类型在数据集中的普及率:")
    # total_channels_instances = sum(channel_types_instances.values())
    
    # for ch_type in sorted(channel_types_files.keys()):
    #     file_count = channel_types_files[ch_type]
    #     percentage = file_count / total_files * 100
    #     print(f"{ch_type.upper()}: 存在于 {file_count} 个文件中 ({percentage:.1f}% 的文件)")
    
    # print("\n通道类型在总通道中的比例:")
    # for ch_type in sorted(channel_types_instances.keys()):
    #     instance_count = channel_types_instances[ch_type]
    #     percentage = instance_count / total_channels_instances * 100
    #     print(f"{ch_type.upper()}: {instance_count} 个实例 ({percentage:.1f}% 的总通道)")
    
    # # 打印最常见的10种通道类型
    # print("\n最常见的10种通道类型:")
    # top_10_types = channel_types_instances.most_common(10)
    # for i, (ch_type, count) in enumerate(top_10_types):
    #     percentage = count / total_channels_instances * 100
    #     print(f"{i+1}. {ch_type.upper()}: {count} 个实例 ({percentage:.1f}%)")

if __name__ == "__main__":
    # 设置你的数据集目录
    dataset_directory = "../rzw_23210720062/Brain_hup_data"
    
    print_edf_channels(dataset_directory)