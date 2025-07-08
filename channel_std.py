import os
import mne
import re
from tqdm import tqdm
from collections import Counter
import numpy as np

def standardize_channel_name(ch_name):
    """
    标准化通道名称，统一不同表示方式
    
    :param ch_name: 原始通道名称
    :return: 标准化后的通道名称
    """
    # 转换为大写并去除空格
    ch = ch_name.upper().replace(' ', '')
    
    # 1. 去除常见的后缀和前缀
    ch = re.sub(r'[-_]?REF$', '', ch)  # 去除 -Ref 后缀
    ch = re.sub(r'^EEG', '', ch)       # 去除 EEG 前缀
    ch = re.sub(r'[-_]?MON$', '', ch)  # 去除 -Mon 后缀
    ch = re.sub(r'[-_]?BIP$', '', ch)  # 去除 -Bip 后缀
    
    # 2. 替换常见别名和等效名称
    ch = ch.replace('T3', 'F7')
    ch = ch.replace('T4', 'F8')
    ch = ch.replace('T5', 'P7')
    ch = ch.replace('T6', 'P8')
    ch = ch.replace('A1', 'M1')
    ch = ch.replace('A2', 'M2')
    ch = ch.replace('EKG', 'ECG')  # EKG 和 ECG 是等效的
    
    # 3. 标准化网格电极命名
    if re.match(r'^G\d+$', ch):
        # 确保两位数编号 (G1 -> G01)
        match = re.match(r'^G(\d+)$', ch)
        if match:
            num = match.group(1).zfill(2)
            ch = f"G{num}"
    
    # 4. 标准化深部电极命名
    if re.match(r'^D\d+$', ch):
        # 确保两位数编号 (D1 -> D01)
        match = re.match(r'^D(\d+)$', ch)
        if match:
            num = match.group(1).zfill(2)
            ch = f"D{num}"
    
    # 5. 标准化海马电极命名
    if 'HIP' in ch:
        # 提取数字部分并标准化 (HIP1 -> HIP01)
        match = re.match(r'^HIPP?(\d+)$', ch)
        if match:
            num = match.group(1).zfill(2)
            ch = f"HIP{num}"
    
    # 6. 标准化杏仁核电极命名
    if 'AMY' in ch:
        # 提取数字部分并标准化 (AMY1 -> AMY01)
        match = re.match(r'^AMY(\d+)$', ch)
        if match:
            num = match.group(1).zfill(2)
            ch = f"AMY{num}"
    
    # 7. 标准化带脑半球前缀的电极
    if re.match(r'^[LR][A-Z]+\d+$', ch):
        # 分离脑半球和电极名称 (LG01 -> L_G01)
        match = re.match(r'^([LR])([A-Z]+)(\d+)$', ch)
        if match:
            hemisphere = match.group(1)
            electrode = match.group(2)
            number = match.group(3).zfill(2)  # 确保两位数编号
            ch = f"{hemisphere}_{electrode}{number}"
    
    # 8. 标准化带连字符的电极名称
    if '-' in ch:
        # 替换为下划线 (LGR14-1 -> LGR14_1)
        ch = ch.replace('-', '_')
    
    # 9. 特殊电极标准化
    if ch in ['EKG', 'ECG1', 'ECG2', 'EEGEKG-REF']:
        ch = 'ECG'
    elif ch in ['EMG1', 'EMG2', 'EMG-L', 'EMG-R']:
        ch = 'EMG'
    elif ch in ['EOG1', 'EOG2', 'EOG-L', 'EOG-R']:
        ch = 'EOG'
    
    # 10. 标准10-20系统电极
    std_electrodes = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                      'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
                      'FZ', 'CZ', 'PZ', 'OZ', 'T3', 'T4', 'T5', 'T6', 
                      'A1', 'A2', 'M1', 'M2']
    
    if ch in std_electrodes:
        return ch
    
    # 11. 处理复合电极名称 (如 LAF14-1)
    if re.match(r'^[LR][A-Z]+\d+_\d+$', ch):
        # 保留原样 (LAF14_1)
        return ch
    
    # 12. 去除多余的描述性文本
    ch = re.sub(r'CHANNEL\d+', '', ch)
    ch = re.sub(r'ELECTRODE\d+', '', ch)
    ch = re.sub(r'CONTACT\d+', '', ch)
    
    return ch

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
    original_to_standard = {}            # 原始名称到标准化名称的映射
    
    # 存储标准化前后的通道集合用于比较
    original_channel_sets = []           # 原始通道集合
    standardized_channel_sets = []        # 标准化后通道集合
    
    # 遍历所有EDF文件并获取通道信息
    for file_path in tqdm(edf_files, desc="处理EDF文件"):
        try:
            # 读取EDF文件头信息（不加载实际数据）
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            
            # 获取原始通道名称
            original_ch_names = raw.ch_names
            
            # 标准化通道名称
            standardized_ch_names = []
            for ch in original_ch_names:
                std_ch = standardize_channel_name(ch)
                standardized_ch_names.append(std_ch)
                original_to_standard[ch] = std_ch
            
            # 使用标准化后的通道名称
            ch_names = standardized_ch_names
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
            
            # 存储原始和标准化通道集合
            original_channel_sets.append(set(original_ch_names))
            standardized_channel_sets.append(set(standardized_ch_names))
            
            # 使用MNE获取通道类型
            file_channel_types = {}
            for ch in ch_names:
                try:
                    # 获取通道类型
                    ch_type = mne.io.pick.channel_type(raw.info, ch)
                    
                    # 如果是未知类型，尝试猜测
                    if ch_type == "misc":
                        # 尝试使用更精确的类型检测
                        if "ECG" in ch or "EKG" in ch:
                            ch_type = "ecg"
                        elif "EMG" in ch:
                            ch_type = "emg"
                        elif "EOG" in ch:
                            ch_type = "eog"
                        else:
                            # 尝试使用MNE的内部方法
                            try:
                                picks_by_type = mne.io.pick._picks_by_type(raw.info, None)
                                if picks_by_type:
                                    ch_type = picks_by_type[0][0]  # 使用第一个检测到的类型
                            except:
                                ch_type = "misc"
                    
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
            print(f"原始通道名称示例: {', '.join(original_ch_names[:3])}{'...' if len(original_ch_names) > 3 else ''}")
            print(f"标准化后通道名称示例: {', '.join(ch_names[:3])}{'...' if len(ch_names) > 3 else ''}")
            
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
    
    # ===== 2. 公共通道统计 (标准化前) =====
    # 计算所有文件中都出现的公共通道 (原始名称)
    original_common_channels = set.intersection(*original_channel_sets) if original_channel_sets else set()
    print("\n===== 原始名称公共通道统计 =====")
    print(f"出现在所有 {len(edf_files)} 个文件中的公共通道 (原始名称) ({len(original_common_channels)} 个):")
    if original_common_channels:
        print(", ".join(sorted(original_common_channels)[:10]) + ("..." if len(original_common_channels) > 10 else ""))
    else:
        print("无")
    
    # ===== 3. 公共通道统计 (标准化后) =====
    # 计算所有文件中都出现的公共通道 (标准化后名称)
    standardized_common_channels = set.intersection(*standardized_channel_sets) if standardized_channel_sets else set()
    print("\n===== 标准化后公共通道统计 =====")
    print(f"出现在所有 {len(edf_files)} 个文件中的公共通道 (标准化后名称) ({len(standardized_common_channels)} 个):")
    if standardized_common_channels:
        common_channels_info = []
        for ch in sorted(standardized_common_channels):
            ch_type = type_mapping.get(ch, "unknown")
            common_channels_info.append(f"{ch} ({ch_type})")
        print(", ".join(common_channels_info))
    else:
        print("无")
    
    # # 比较标准化前后的公共通道
    # print("\n===== 标准化前后公共通道比较 =====")
    # print(f"原始公共通道数量: {len(original_common_channels)}")
    # print(f"标准化后公共通道数量: {len(standardized_common_channels)}")
    
    # # 找出新增的公共通道
    # new_common_channels = standardized_common_channels - original_common_channels
    # print(f"\n标准化后新增的公共通道 ({len(new_common_channels)} 个):")
    # if new_common_channels:
    #     for ch in sorted(new_common_channels):
    #         print(f"  - {ch}")
    # else:
    #     print("无")
    
    # # 找出消失的公共通道
    # lost_common_channels = original_common_channels - standardized_common_channels
    # print(f"\n标准化后消失的公共通道 ({len(lost_common_channels)} 个):")
    # if lost_common_channels:
    #     for ch in sorted(lost_common_channels):
    #         print(f"  - {ch}")
    # else:
    #     print("无")
    
    # # ===== 4. 通道出现频率统计 =====
    # print("\n===== 通道出现频率 =====")
    # total_files = len(edf_files)
    
    # # 打印出现频率最高的通道
    # top_10_common = channel_presence_counter.most_common(10)
    # print("\n最常见的10个通道:")
    # for i, (channel, count) in enumerate(top_10_common):
    #     ch_type = type_mapping.get(channel, "unknown")
    #     percentage = count / total_files * 100
    #     print(f"{i+1}. {channel} ({ch_type}): {count} 文件 ({percentage:.1f}%)")
    
    # # 打印出现频率最低的通道
    # print("\n最不常见的10个通道:")
    # sorted_least_common = sorted(channel_presence_counter.items(), key=lambda x: x[1])
    # for i, (channel, count) in enumerate(sorted_least_common[:10]):
    #     ch_type = type_mapping.get(channel, "unknown")
    #     percentage = count / total_files * 100
    #     print(f"{i+1}. {channel} ({ch_type}): {count} 文件 ({percentage:.1f}%)")
    
    # # ===== 5. 所有通道名称（按字母顺序） =====
    # print("\n===== 所有出现的通道名称 (按字母顺序) =====")
    # print(f"总共 {len(unique_channels)} 个唯一的通道名称:")
    
    # # 按字母顺序排序
    # sorted_channel_names = sorted(unique_channels)
    
    # # 分组打印，每行10个
    # print("\n标准化通道名称列表:")
    # for i in range(0, len(sorted_channel_names), 10):
    #     print(", ".join(sorted_channel_names[i:i+10]))
    
    # # ===== 6. 所有通道及其频率 =====
    # print("\n===== 所有通道及其频率 (按字母排序) =====")
    # for channel, count in sorted(channel_presence_counter.items()):
    #     ch_type = type_mapping.get(channel, "unknown")
    #     percentage = count / total_files * 100
    #     print(f"{channel} ({ch_type}): {count} 文件 ({percentage:.1f}%)")
    
    # # ===== 7. 原始到标准化的映射 =====
    # print("\n===== 原始通道名称到标准化名称的映射 =====")
    # print("示例映射关系 (前20个):")
    # for i, (orig, std) in enumerate(list(original_to_standard.items())[:20]):
    #     print(f"{orig} -> {std}")
    
    # # ===== 8. 使用MNE的通道类型统计 =====
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
    
    # # ===== 9. 标准化效果统计 =====
    # print("\n===== 标准化效果统计 =====")
    # unique_original = len(original_to_standard)
    # unique_standardized = len(set(original_to_standard.values()))
    # reduction_percentage = (1 - unique_standardized / unique_original) * 100
    
    # print(f"原始唯一通道名称数量: {unique_original}")
    # print(f"标准化后唯一通道名称数量: {unique_standardized}")
    # print(f"名称减少比例: {reduction_percentage:.2f}%")
    
    # # 计算标准化映射关系
    # standard_to_original = {}
    # for orig, std in original_to_standard.items():
    #     if std not in standard_to_original:
    #         standard_to_original[std] = []
    #     standard_to_original[std].append(orig)
    
    # # 打印最常见的标准化名称映射
    # print("\n最常见的标准化名称映射 (前10个):")
    # sorted_std = sorted(standard_to_original.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    # for i, (std, orig_list) in enumerate(sorted_std):
    #     print(f"{i+1}. {std} 映射自 {len(orig_list)} 个原始名称:")
    #     print(f"   {', '.join(orig_list[:5])}{'...' if len(orig_list) > 5 else ''}")

if __name__ == "__main__":
    # 设置你的数据集目录
    dataset_directory = "../rzw_23210720062/Brain_hup_data"
    
    print_edf_channels(dataset_directory)