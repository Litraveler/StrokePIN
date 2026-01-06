import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import Sequence
from sklearn.metrics import roc_curve
import itertools
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from itertools import combinations
#！！！！！！！！！！！！！！！！！！！！优化超参数！！！！！！！！！！！！！！！！！！！！！
#！！！！！！！！！！！！！！！测试模型在不同数量数据集下的性能！！！！！！！！！！！！！！！！！！！！！
input_data_length_touch = 90
input_feature_length_touch = 8

input_data_length_git = 160
input_feature_length_git = 4
from tensorflow.keras.callbacks import Callback

# ======================== 数据准备与用户划分 ========================
def load_Tuser_ids(dirs, source):
    """从指定目录提取所有用户ID"""
    user_ids = set()
    for file_path in dirs:
        df = pd.read_csv(file_path)
        df_filtered = df[df['data_source'] == source]
        user_ids.update(df_filtered['UUID'].unique())
    return list(user_ids)

# 定义TDatas文件列表（新增需求的数据集）
tdatas_files = [
    'TDatas/Combmasked_normalized_filtered_touch_data_sit.csv',
    'TDatas/Combmasked_normalized_filtered_touch_data_walk.csv',
    'TDatas/Combmasked_normalized_filtered_sensor_data_sit.csv',
    'TDatas/Combmasked_normalized_filtered_sensor_data_walk.csv'
]


# 2. TDatas用户加载并按7:3划分（核心修改1）
all_Tdatas_users = load_Tuser_ids(tdatas_files,"T")
all_Ddatas_users = load_Tuser_ids(tdatas_files,"D")

tdatas_train_users, test_users = train_test_split(
    all_Tdatas_users, 
    test_size=0.3,  # 30%作为测试集
    random_state=42,  # 固定随机种子保证可复现
    shuffle=True
)

train_Tusers, val_Tusers = train_test_split(tdatas_train_users, test_size=0.3)
train_Dusers, val_Dusers = train_test_split(all_Ddatas_users, test_size=0.3)

train_users = train_Tusers + train_Dusers
val_users = val_Tusers + val_Dusers

# ======================== 获取所有CSV文件路径 ========================
# 训练集：原有datas + 完整TDatas（用于提取训练用户的4个时间段数据）
train_csv_files_touch = [
    'TDatas/Combmasked_normalized_filtered_touch_data_sit.csv',
    'TDatas/Combmasked_normalized_filtered_touch_data_walk.csv'
]
train_csv_files_git = [
    'TDatas/Combmasked_normalized_filtered_sensor_data_sit.csv',
    'TDatas/Combmasked_normalized_filtered_sensor_data_walk.csv'
]

# 测试集：保持原有TDatas路径，逻辑不变
test_csv_files_touch = [
    'TDatas/Combmasked_normalized_filtered_touch_data_walk.csv',
    'TDatas/Combmasked_normalized_filtered_touch_data_sit.csv'
]
test_csv_files_git = [
    'TDatas/Combmasked_normalized_filtered_sensor_data_walk.csv',
    'TDatas/Combmasked_normalized_filtered_sensor_data_sit.csv'
]


class TripletDataGeneratorForTouch(Sequence):
    """用于三元损失的动态数据生成器"""
    def __init__(self, file_paths, users, batch_size=32):
        self.file_paths = file_paths
        self.users = users
        self.batch_size = batch_size  # 可以根据需要调整
        self.triplet_queue = []
        self.findex = 0
        for file in self.file_paths:
            df = pd.read_csv(file)
            # 按 PIN 分组
            for pin, pin_group in df.groupby('PIN'):
                for user in self.users:
                    if user in pin_group['UUID'].unique():
                        triplet_pairs = self._generate_triplets_for_user(user, pin_group)
                        self.triplet_queue.extend(triplet_pairs)
        print("三元组样本数量为：")
        print(len(self.triplet_queue))

    def __len__(self):
        return int(len(self.triplet_queue) / self.batch_size)

    def _generate_triplets_for_user(self, user, pin_group):
        """为单个用户生成所有可能的三元组"""
        user_samples = pin_group[pin_group['UUID'] == user]
        # 按 touch_id 分组，每个 touch_id 对应一个样本
        grouped = user_samples.groupby('Sample ID')
        samples = [group for _, group in grouped]
        if len(samples) < 2:
            return []  # 如果样本数少于2，则无法生成三元组
        # 生成所有可能的锚点和正样本组合
        anchor_pos_pairs = list(itertools.combinations(samples, 2))
        triplets = []
        # 为每个组合获取负样本
        neg_users = [u for u in pin_group['UUID'].unique() if u != user and u in self.users]
        if not neg_users:
            return []
        for anchor_pos in anchor_pos_pairs:
            neg_user_idx = np.random.choice(len(neg_users))
            neg_user = neg_users[neg_user_idx]
            neg_samples = pin_group[pin_group['UUID'] == neg_user]
            grouped = neg_samples.groupby('Sample ID')
            neg_samples = [group for _, group in grouped]
            if not neg_samples:
                continue
            neg_idx = np.random.choice(len(neg_samples))  # 先随机选择索引
            neg = neg_samples[neg_idx]  # 然后通过索引获取样本
            a_features = anchor_pos[0][
                ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            p_features = anchor_pos[1][
                ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            n_features = neg[
                ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            triplets.append((a_features, p_features, n_features))
        return triplets

    def __getitem__(self, idx):
        if idx == 0:
            np.random.shuffle(self.triplet_queue)
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_triplets = self.triplet_queue[start_idx:end_idx]
        anchors, positives, negatives = zip(*batch_triplets)
        # 转换为 NumPy 数组
        anchors_array = np.array(anchors)
        positives_array = np.array(positives)
        negatives_array = np.array(negatives)
        return (anchors_array, positives_array, negatives_array), np.zeros(len(batch_triplets))

class TripletDataGeneratorForGit(Sequence):
    """用于三元损失的动态数据生成器"""
    def __init__(self, file_paths, users, batch_size=32):
        self.file_paths = file_paths
        self.users = users
        self.batch_size = batch_size  # 可以根据需要调整
        self.triplet_queue = []
        self.findex = 0
        for file in self.file_paths:
            df = pd.read_csv(file)
            # 按 PIN 分组
            for pin, pin_group in df.groupby('PIN'):
                for user in self.users:
                    if user in pin_group['UUID'].unique():
                        triplet_pairs = self._generate_triplets_for_user(user, pin_group)
                        self.triplet_queue.extend(triplet_pairs)

    def __len__(self):
        return int(len(self.triplet_queue ) / self.batch_size)

    def _generate_triplets_for_user(self, user, pin_group):
        """为单个用户生成所有可能的三元组"""
        user_samples = pin_group[pin_group['UUID'] == user]
        # 按 touch_id 分组，每个 touch_id 对应一个样本
        grouped = user_samples.groupby('Sample ID')
        samples = [group for _, group in grouped]
        if len(samples) < 2:
            return []  # 如果样本数少于2，则无法生成三元组
        # 生成所有可能的锚点和正样本组合
        anchor_pos_pairs = list(itertools.combinations(samples, 2))
        triplets = []
        # 为每个组合获取负样本
        neg_users = [u for u in pin_group['UUID'].unique() if u != user and u in self.users]
        if not neg_users:
            return []
        for anchor_pos in anchor_pos_pairs:
            neg_user_idx = np.random.choice(len(neg_users))
            neg_user = neg_users[neg_user_idx]
            neg_samples = pin_group[pin_group['UUID'] == neg_user]
            grouped = neg_samples.groupby('Sample ID')
            neg_samples = [group for _, group in grouped]
            if not neg_samples:
                continue
            neg_idx = np.random.choice(len(neg_samples))  # 先随机选择索引
            neg = neg_samples[neg_idx]  # 然后通过索引获取样本
            a_features = anchor_pos[0][
                ['Time', 'X', 'Y', 'Z']].values
            p_features = anchor_pos[1][
                ['Time', 'X', 'Y', 'Z']].values
            n_features = neg[
                ['Time', 'X', 'Y', 'Z']].values
            triplets.append((a_features, p_features, n_features))
        return triplets

    def __getitem__(self, idx):
        if idx == 0:
            np.random.shuffle(self.triplet_queue)
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_triplets = self.triplet_queue[start_idx:end_idx]
        anchors, positives, negatives = zip(*batch_triplets)
        # 转换为 NumPy 数组
        anchors_array = np.array(anchors)
        positives_array = np.array(positives)
        negatives_array = np.array(negatives)
        return (anchors_array, positives_array, negatives_array), np.zeros(len(batch_triplets))

class VerificationDataGenerator(Sequence):
    """用于认证网络的动态数据生成器"""
    def __init__(self, csv_touch_files, csv_git_files, users, batch_size=32):
        self.csv_touch_files = csv_touch_files
        self.csv_git_files = csv_git_files
        self.users = users
        self.batch_size = batch_size
        self.positive_pairs = []  # 存储正样本对
        self.negative_pairs = []  # 存储负样本对
        # 处理每个文件对
        for touch_file in self.csv_touch_files:
            if touch_file.endswith("_sit.csv"):
                git_file_suffix = "_sit.csv"
            elif touch_file.endswith("_walk.csv"):
                git_file_suffix = "_walk.csv"
            else:
                continue  # 跳过不匹配的文件

            git_file = next((f for f in self.csv_git_files if f.endswith(git_file_suffix)), None)
            if not git_file:
                continue  # 如果没有找到对应的 git_file，跳过

            touch_df = pd.read_csv(touch_file)
            git_df = pd.read_csv(git_file)

            for PIN in touch_df['PIN'].unique():
                touch_pin_group = touch_df[touch_df['PIN'] == PIN]
                git_pin_group = git_df[git_df['PIN'] == PIN]

                for user in self.users:
                    # 筛选 touch 和 git 中属于该用户的样本
                    touch_user_samples = touch_pin_group[touch_pin_group['UUID'] == user]
                    git_user_samples = git_pin_group[git_pin_group['UUID'] == user]

                    # 如果 touch 或 git 中没有该用户的样本，跳过
                    if touch_user_samples.empty or git_user_samples.empty:
                        continue

                    # 生成正样本对（核心修改2：4个时间段两两组合）
                    self._generate_positive_pairs(touch_user_samples, git_user_samples)

                    # 生成负样本对（保持原有逻辑不变）
                    self._generate_negative_pairs(touch_user_samples, git_user_samples, user, touch_pin_group,
                                                  git_pin_group)
        print("正负对长度")
        print(len(self.positive_pairs))
        print(len(self.negative_pairs))

    def _generate_positive_pairs(self, touch_user_samples, git_user_samples):
        """为单个用户生成正样本对（修改：4个时间段两两组合，覆盖同时间段/跨时间段）"""
        # 步骤1：获取有效样本（touch和git共有Sample ID）
        common_sample_ids = set(touch_user_samples['Sample ID']).intersection(git_user_samples['Sample ID'])
        if len(common_sample_ids) == 0:
            return []
        
        # 步骤2：提取4个时间段（固定为1-4）
        valid_time_periods = [1, 2, 3, 4]
        # 筛选当前用户有数据的时间段
        user_time_periods = [tp for tp in valid_time_periods if tp in touch_user_samples['TimePeriod'].unique()]
        if len(user_time_periods) < 1:
            return []
        
        # 步骤3：生成时间段两两组合（包括同时间段内、跨时间段，无重复组合）
        # 若需要包含（t1,t2）和（t2,t1），使用 itertools.product；此处用 combinations_with_replacement 避免重复
        time_period_pairs = list(itertools.combinations_with_replacement(user_time_periods, 2))
        
        # 步骤4：遍历时间段组合，生成样本对
        for tp1, tp2 in time_period_pairs:
            # 筛选时间段tp1的有效样本
            touch_tp1 = touch_user_samples[touch_user_samples['TimePeriod'] == tp1]
            git_tp1 = git_user_samples[git_user_samples['TimePeriod'] == tp1]
            sample_ids_tp1 = set(touch_tp1['Sample ID']).intersection(git_tp1['Sample ID'])
            
            # 筛选时间段tp2的有效样本
            touch_tp2 = touch_user_samples[touch_user_samples['TimePeriod'] == tp2]
            git_tp2 = git_user_samples[git_user_samples['TimePeriod'] == tp2]
            sample_ids_tp2 = set(touch_tp2['Sample ID']).intersection(git_tp2['Sample ID'])
            
            if not sample_ids_tp1 or not sample_ids_tp2:
                continue
            
            # 步骤5：遍历两个时间段的样本，两两组合（排除同一样本）
            for sid1 in sample_ids_tp1:
                for sid2 in sample_ids_tp2:
                    # 跳过同时间段内的同一样本（跨时间段可保留同sid，若有）
                    if tp1 == tp2 and sid1 == sid2:
                        continue
                    
                    # 提取touch样本特征
                    touch_feat1 = touch_tp1[touch_tp1['Sample ID'] == sid1][
                        ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']
                    ].values
                    touch_feat2 = touch_tp2[touch_tp2['Sample ID'] == sid2][
                        ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']
                    ].values
                    
                    # 提取git样本特征
                    git_feat1 = git_tp1[git_tp1['Sample ID'] == sid1][
                        ['Time', 'X', 'Y', 'Z']
                    ].values
                    git_feat2 = git_tp2[git_tp2['Sample ID'] == sid2][
                        ['Time', 'X', 'Y', 'Z']
                    ].values
                    
                    # 验证特征非空，添加到正样本对
                    if touch_feat1.size > 0 and touch_feat2.size > 0 and git_feat1.size > 0 and git_feat2.size > 0:
                        self.positive_pairs.append((touch_feat1, git_feat1, touch_feat2, git_feat2, 1))

    def _generate_negative_pairs(self, touch_user_samples, git_user_samples, user, touch_pin_group, git_pin_group):
        """为单个用户生成负样本对（保持原有逻辑不变）"""
        # 找到 touch 和 git 中共有相同的 Sample ID
        common_sample_ids = set(touch_user_samples['Sample ID']).intersection(git_user_samples['Sample ID'])

        other_users = list(set(touch_pin_group['UUID']).intersection(git_pin_group['UUID']) - {user})
        if len(other_users) < 2:
            return  # 如果没有足够的其他用户，跳过
        for sample_id in common_sample_ids:
            other_users_idx = np.random.choice(len(other_users), 2)  # 先随机选择索引
            # 从 touch 和 git 中获取该 Sample ID 的样本
            touch_samples = touch_user_samples[touch_user_samples['Sample ID'] == sample_id]
            git_samples = git_user_samples[git_user_samples['Sample ID'] == sample_id]

            touch_features1 = touch_samples[
                ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values

            git_features1 = git_samples[['Time', 'X', 'Y', 'Z']].values


            for other_user_idx in other_users_idx:
                other_user = other_users[other_user_idx]
                # 找到其他用户的 touch 和 git 中具有相同 Sample ID 的样本
                other_touch_samples = touch_pin_group[(touch_pin_group['UUID'] == other_user)]
                other_git_samples = git_pin_group[(git_pin_group['UUID'] == other_user)]

                # 找到 touch 和 git 中具有相同 Sample ID 的样本
                common_sample_ids_other_user = set(other_touch_samples['Sample ID']).intersection(other_git_samples['Sample ID'])

                if not common_sample_ids_other_user:
                    continue  # 如果没有共同的 Sample ID，跳过

                # 随机选择一个 Sample ID
                selected_sample_id = np.random.choice(list(common_sample_ids_other_user))

                # 获取其他用户的 touch 和 git 样本
                neg_touch_sample = other_touch_samples[other_touch_samples['Sample ID'] == selected_sample_id]
                neg_git_sample = other_git_samples[other_git_samples['Sample ID'] == selected_sample_id]

                # 提取其他用户的 touch 和 git 样本的特征
                touch_features2 = neg_touch_sample[
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                git_features2 = neg_git_sample[['Time', 'X', 'Y', 'Z']].values

                # 将提取的特征添加到负样本列表
                self.negative_pairs.append((touch_features1, git_features1, touch_features2, git_features2, 0))

    def __len__(self):
        return int((len(self.positive_pairs) + len(self.negative_pairs)) / self.batch_size)

    def on_epoch_end(self):
        # 在每个 epoch 结束时打乱数据
        np.random.shuffle(self.positive_pairs)
        np.random.shuffle(self.negative_pairs)

    def __getitem__(self, idx):
        # 在每个批次开始时打乱数据
        if idx == 0:
            self.on_epoch_end()

        # 计算批次大小
        half_batch_size = self.batch_size // 2  # 正负样本各占一半

        # 获取当前批次的正负样本
        positive_start = idx * half_batch_size
        positive_end = positive_start + half_batch_size
        negative_start = idx * half_batch_size
        negative_end = negative_start + half_batch_size

        # 获取当前批次的正负样本（处理边界情况）
        positive_batch = self.positive_pairs[positive_start:positive_end]
        negative_batch = self.negative_pairs[negative_start:negative_end]

        batch_pairs = []
        for positive_pair in positive_batch:
            batch_pairs.append(positive_pair)
        for negative_pair in negative_batch:
            batch_pairs.append(negative_pair)
        # 打乱批次顺序
        np.random.shuffle(batch_pairs)
        anchors_touch, anchors_git, contrastive_touch, contrastive_git, labels = zip(*batch_pairs)
        anchors_touch_array = np.array(anchors_touch)
        anchors_git_array = np.array(anchors_git)
        contrastive_touch_array = np.array(contrastive_touch)
        contrastive_git_array = np.array(contrastive_git)
        labels_array = np.array(labels)

        # 返回批次数据
        return (anchors_touch_array,anchors_git_array,contrastive_touch_array,contrastive_git_array), labels_array


class VerificationDataGeneratorForVal(Sequence):
    """用于认证网络的动态数据生成器（验证集，逻辑不变）"""
    def __init__(self, touch_file, git_file, PIN, user, users, batch_size=32):
        self.touch_file = touch_file
        self.git_file = git_file
        self.PIN = PIN
        self.user = user
        self.users = users
        self.batch_size = batch_size
        self.positive_pairs = []  # 存储正样本对
        self.negative_pairs = []  # 存储负样本对

        # 处理每个文件对
        touch_df = pd.read_csv(touch_file)
        git_df = pd.read_csv(git_file)

        touch_pin_group = touch_df[touch_df['PIN'] == PIN]
        git_pin_group = git_df[git_df['PIN'] == PIN]
        # 处理该 PIN 组的数据
        # 筛选 touch 和 git 中属于该用户的样本
        touch_user_samples = touch_pin_group[touch_pin_group['UUID'] == self.user]
        git_user_samples = git_pin_group[git_pin_group['UUID'] == self.user]

        # 生成正样本对
        self._generate_positive_pairs(touch_user_samples, git_user_samples)

        # 生成负样本对（从其他用户的样本中随机选择）
        self._generate_negative_pairs(touch_user_samples, git_user_samples, self.user, touch_pin_group,
                                      git_pin_group)


    def _generate_positive_pairs(self, touch_user_samples, git_user_samples):
        """为单个用户生成正样本对（验证集保持原有逻辑）"""
        # 找到 touch 和 git 中共有相同的 Sample ID
        common_sample_ids = set(touch_user_samples['Sample ID']).intersection(git_user_samples['Sample ID'])
        if len(common_sample_ids) == 0:
            return []
        # 遍历所有可能的 Sample ID 组合
        for sample_id1, sample_id2 in combinations(common_sample_ids, 2):
            # 从 touch 中获取第一个 Sample ID 的样本
            touch_samples1 = touch_user_samples[touch_user_samples['Sample ID'] == sample_id1]
            # 从 touch 中获取第二个 Sample ID 的样本
            touch_samples2 = touch_user_samples[touch_user_samples['Sample ID'] == sample_id2]

            # 从 git 中获取第一个 Sample ID 的样本
            git_samples1 = git_user_samples[git_user_samples['Sample ID'] == sample_id1]
            # 从 git 中获取第二个 Sample ID 的样本
            git_samples2 = git_user_samples[git_user_samples['Sample ID'] == sample_id2]

            # 提取 touch 样本的特征
            touch_features1 = touch_samples1[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            touch_features2 = touch_samples2[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values

            # 提取 git 样本的特征
            git_features1 = git_samples1[['Time', 'X', 'Y', 'Z']].values
            git_features2 = git_samples2[['Time', 'X', 'Y', 'Z']].values

            # 将提取的特征添加到正样本列表
            self.positive_pairs.append((touch_features1, git_features1, touch_features2, git_features2, 1))

    def _generate_negative_pairs(self, touch_user_samples, git_user_samples, user, touch_pin_group, git_pin_group):
        """为单个用户生成负样本对（验证集保持原有逻辑）"""
        # 找到 touch 和 git 中共有相同的 Sample ID
        common_sample_ids = set(touch_user_samples['Sample ID']).intersection(git_user_samples['Sample ID'])

        other_users = list(set(touch_pin_group['UUID']).intersection(git_pin_group['UUID']) - {user})
        if len(other_users) < 2:
            return  # 如果没有足够的其他用户，跳过

        for sample_id in common_sample_ids:
            other_users_idx = np.random.choice(len(other_users), 2)  # 先随机选择索引
            # 从 touch 和 git 中获取该 Sample ID 的样本
            touch_samples = touch_user_samples[touch_user_samples['Sample ID'] == sample_id]
            git_samples = git_user_samples[git_user_samples['Sample ID'] == sample_id]

            touch_features1 = touch_samples[
                ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values

            git_features1 = git_samples[['Time', 'X', 'Y', 'Z']].values

            for other_user_idx in other_users_idx:
                other_user = other_users[other_user_idx]
                # 找到其他用户的 touch 和 git 中具有相同 Sample ID 的样本
                other_touch_samples = touch_pin_group[(touch_pin_group['UUID'] == other_user)]
                other_git_samples = git_pin_group[(git_pin_group['UUID'] == other_user)]

                # 找到 touch 和 git 中具有相同 Sample ID 的样本
                common_sample_ids_other_user = set(other_touch_samples['Sample ID']).intersection(other_git_samples['Sample ID'])

                if not common_sample_ids_other_user:
                    continue  # 如果没有共同的 Sample ID，跳过

                # 随机选择一个 Sample ID
                selected_sample_id = np.random.choice(list(common_sample_ids_other_user))

                # 获取其他用户的 touch 和 git 样本
                neg_touch_sample = other_touch_samples[other_touch_samples['Sample ID'] == selected_sample_id]
                neg_git_sample = other_git_samples[other_git_samples['Sample ID'] == selected_sample_id]

                # 提取其他用户的 touch 和 git 样本的特征
                touch_features2 = neg_touch_sample[
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                git_features2 = neg_git_sample[['Time', 'X', 'Y', 'Z']].values

                # 将提取的特征添加到负样本列表
                self.negative_pairs.append((touch_features1, git_features1, touch_features2, git_features2, 0))

    def __len__(self):
        return int(min(len(self.positive_pairs), len(self.negative_pairs))*2 / self.batch_size)

    def on_epoch_end(self):
        # 在每个 epoch 结束时打乱数据
        np.random.shuffle(self.positive_pairs)
        np.random.shuffle(self.negative_pairs)

    def __getitem__(self, idx):
        # 在每个批次开始时打乱数据
        if idx == 0:
            self.on_epoch_end()

        # 计算批次大小
        half_batch_size = self.batch_size // 2  # 正负样本各占一半

        # 获取当前批次的正负样本
        positive_start = idx * half_batch_size
        positive_end = positive_start + half_batch_size
        negative_start = idx * half_batch_size
        negative_end = negative_start + half_batch_size

        # 获取当前批次的正负样本（处理边界情况）
        positive_batch = self.positive_pairs[positive_start:positive_end]
        negative_batch = self.negative_pairs[negative_start:negative_end]

        batch_pairs = []
        # 合并正负样本
        for positive_pair in positive_batch:
            batch_pairs.append(positive_pair)
        for negative_pair in negative_batch:
            batch_pairs.append(negative_pair)

        # 打乱批次顺序
        np.random.shuffle(batch_pairs)

        anchors_touch, anchors_git, contrastive_touch, contrastive_git, labels = zip(*batch_pairs)
        anchors_touch_array = np.array(anchors_touch)
        anchors_git_array = np.array(anchors_git)
        contrastive_touch_array = np.array(contrastive_touch)
        contrastive_git_array = np.array(contrastive_git)
        labels_array = np.array(labels)

        # 返回批次数据
        return (anchors_touch_array,anchors_git_array,contrastive_touch_array,contrastive_git_array), labels_array

# class VerificationDataGeneratorForTest(Sequence):
#     """用于认证网络的动态数据生成器（测试集，逻辑完全不变）"""
#     def __init__(self, touch_file, git_file, PIN, user, users, time=2, batch_size=32):
#         self.touch_file = touch_file
#         self.git_file = git_file
#         self.PIN = PIN
#         self.user = user
#         self.users = users
#         self.batch_size = batch_size
#         self.positive_pairs = []  # 存储正样本对
#         self.negative_pairs = []  # 存储负样本对
#         self.test_time = time  # 测试样本时间段（当前传入时间段）
#         self.anchor_time = 1  # 锚点样本时间段（time-1）
#         self.valid = True  # 新增：初始化valid属性为True

#         # 边界判断：锚点时间段不能小于1（TimePeriod仅1-4）
#         if self.anchor_time < 1:
#             self.valid = False  # 标记为无效
#             print(f"警告：锚点时间段无效，已跳过")
#             return

#         # 加载完整数据
#         touch_df = pd.read_csv(touch_file)
#         git_df = pd.read_csv(git_file)

#         # -------------------------- 核心修改1：拆分锚点时间段（time-1）和测试时间段（time）数据 --------------------------
#         # 1. 锚点数据（time-1）：用于正负样本的锚点
#         touch_anchor_df = touch_df[touch_df['TimePeriod'] == self.anchor_time]
#         git_anchor_df = git_df[git_df['TimePeriod'] == self.anchor_time]

#         # 2. 测试数据（time）：用于正样本的对比样本
#         touch_test_df = touch_df[touch_df['TimePeriod'] == self.test_time]
#         git_test_df = git_df[git_df['TimePeriod'] == self.test_time]

#         # 3. 筛选当前PIN下的所有数据（负样本需要其他用户的所有时间段）
#         touch_pin_all = touch_df[touch_df['PIN'] == PIN]
#         git_pin_all = git_df[git_df['PIN'] == PIN]

#         # 4. 筛选当前PIN下的锚点/测试数据
#         touch_anchor_pin = touch_anchor_df[touch_anchor_df['PIN'] == PIN]
#         git_anchor_pin = git_anchor_df[git_anchor_df['PIN'] == PIN]
#         touch_test_pin = touch_test_df[touch_test_df['PIN'] == PIN]
#         git_test_pin = git_test_df[git_test_df['PIN'] == PIN]

#         # 5. 筛选当前用户的锚点样本（time-1）和测试样本（time）（正样本专用）
#         self.touch_anchor_user = touch_anchor_pin[touch_anchor_pin['UUID'] == self.user]
#         self.git_anchor_user = git_anchor_pin[git_anchor_pin['UUID'] == self.user]
#         self.touch_test_user = touch_test_pin[touch_test_pin['UUID'] == self.user]
#         self.git_test_user = git_test_pin[git_test_pin['UUID'] == self.user]

#         # -------------------------- 生成正负样本 --------------------------
#         # 生成正样本（锚点：time-1，测试：time）
#         self._generate_positive_pairs()

#         # 生成负样本（锚点：time-1，对比：其他用户所有时间段）
#         self._generate_negative_pairs(touch_pin_all, git_pin_all)

#         if not self.positive_pairs and not self.negative_pairs:
#             self.valid = False
#             print(f"警告：用户{self.user}在PIN{PIN}的时间段{self.test_time}无任何有效样本对，已跳过")
#             return

#         # -------------------------- 核心修改3：强制正负样本数量相同 --------------------------
#         self._align_pos_neg_pair_counts()
#         if not self.valid:
#             return

#     def _generate_positive_pairs(self):
#         """为单个用户生成正样本对（需求1：锚点=time-1，测试=time）"""
#         # 获取当前用户锚点样本（time-1）和测试样本（time）的共同Sample ID（保证样本有效性）
#         anchor_sample_ids = set(self.touch_anchor_user['Sample ID']).intersection(self.git_anchor_user['Sample ID'])
#         test_sample_ids = set(self.touch_test_user['Sample ID']).intersection(self.git_test_user['Sample ID'])

#         if len(anchor_sample_ids) == 0 or len(test_sample_ids) == 0:
#             print("警告：当前用户锚点时间段或测试时间段无有效样本，无法生成正样本")
#             return

#         # 锚点样本（time-1）作为anchors，测试样本（time）作为contrastive（正样本对）
#         for anchor_sid in anchor_sample_ids:
#             for test_sid in test_sample_ids:
#                 # 提取锚点样本特征（time-1）
#                 touch_anchor = self.touch_anchor_user[self.touch_anchor_user['Sample ID'] == anchor_sid]
#                 git_anchor = self.git_anchor_user[self.git_anchor_user['Sample ID'] == anchor_sid]
#                 touch_anchor_feat = touch_anchor[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
#                 git_anchor_feat = git_anchor[['Time', 'X', 'Y', 'Z']].values

#                 # 提取测试样本特征（time）
#                 touch_test = self.touch_test_user[self.touch_test_user['Sample ID'] == test_sid]
#                 git_test = self.git_test_user[self.git_test_user['Sample ID'] == test_sid]
#                 touch_test_feat = touch_test[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
#                 git_test_feat = git_test[['Time', 'X', 'Y', 'Z']].values

#                 # 正样本标签为1，格式：(锚点touch, 锚点git, 测试touch, 测试git, 标签)
#                 self.positive_pairs.append((touch_anchor_feat, git_anchor_feat, touch_test_feat, git_test_feat, 1))

#     def _generate_negative_pairs(self, touch_pin_all, git_pin_all):
#         """为单个用户生成负样本对（需求2：锚点=time-1，对比=其他用户所有时间段）"""
#         # 1. 获取当前用户锚点样本（time-1）的有效Sample ID
#         anchor_sample_ids = set(self.touch_anchor_user['Sample ID']).intersection(self.git_anchor_user['Sample ID'])
#         if len(anchor_sample_ids) == 0:
#             print("警告：当前用户锚点时间段无有效样本，无法生成负样本")
#             return

#         # 2. 筛选当前PIN下除当前用户外的其他用户（保证负样本用户有效性）
#         other_users = list(set(touch_pin_all['UUID']).intersection(git_pin_all['UUID']) - {self.user})
#         if len(other_users) == 0:
#             print("警告：无其他用户数据，无法生成负样本")
#             return

#         # 3. 锚点样本（time-1）作为anchors，其他用户所有时间段样本作为contrastive（负样本对）
#         for anchor_sid in anchor_sample_ids:
#             # 提取当前锚点样本特征（time-1）
#             touch_anchor = self.touch_anchor_user[self.touch_anchor_user['Sample ID'] == anchor_sid]
#             git_anchor = self.git_anchor_user[self.git_anchor_user['Sample ID'] == anchor_sid]
#             touch_anchor_feat = touch_anchor[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
#             git_anchor_feat = git_anchor[['Time', 'X', 'Y', 'Z']].values

#             # 4. 随机选择5个其他用户（保持与原代码逻辑一致的负样本数量规模）
#             selected_other_users = np.random.choice(other_users, size=5, replace=False) if len(other_users) >= 2 else other_users

#             for other_user in selected_other_users:
#                 # 5. 提取其他用户的**所有时间段**数据（需求2核心：不再限制时间段）
#                 other_touch_all = touch_pin_all[touch_pin_all['UUID'] == other_user]
#                 other_git_all = git_pin_all[git_pin_all['UUID'] == other_user]

#                 # 6. 找到其他用户所有时间段的有效Sample ID（touch和git共有的）
#                 other_valid_sids = set(other_touch_all['Sample ID']).intersection(other_git_all['Sample ID'])
#                 if len(other_valid_sids) == 0:
#                     continue

#                 # 7. 随机选择一个其他用户的有效样本
#                 other_sid = np.random.choice(list(other_valid_sids))
#                 other_touch = other_touch_all[other_touch_all['Sample ID'] == other_sid]
#                 other_git = other_git_all[other_git_all['Sample ID'] == other_sid]

#                 # 8. 提取负样本对比特征
#                 touch_other_feat = other_touch[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
#                 git_other_feat = other_git[['Time', 'X', 'Y', 'Z']].values

#                 # 负样本标签为0
#                 self.negative_pairs.append((touch_anchor_feat, git_anchor_feat, touch_other_feat, git_other_feat, 0))

#     def _align_pos_neg_pair_counts(self):
#         min_pair_count = min(len(self.positive_pairs), len(self.negative_pairs))
#         if min_pair_count == 0:
#             self.valid = False
#             print(f"警告：用户{self.user}在PIN{PIN}的时间段{self.test_time}正负样本中至少有一个为空，无法对齐，已跳过")
#             return
    
#         self.positive_pairs = self.positive_pairs[:min_pair_count]
#         self.negative_pairs = self.negative_pairs[:min_pair_count]
#         print(f"正负样本数量已对齐，各{min_pair_count}对")

#     def __len__(self):
#         """计算批次数量：基于对齐后的样本数量，保证批次完整性"""
#         total_valid_pairs = len(self.positive_pairs) + len(self.negative_pairs)
#         return int(total_valid_pairs / self.batch_size)

#     def on_epoch_end(self):
#         """在每个epoch结束时打乱正负样本（保证训练随机性）"""
#         np.random.shuffle(self.positive_pairs)
#         np.random.shuffle(self.negative_pairs)

#     def __getitem__(self, idx):
#         """获取指定批次的数据，保证批次内正负样本各占一半"""
#         half_batch = self.batch_size // 2
#         batch_start = idx * half_batch

#         # 提取当前批次的正负样本（各半）
#         positive_batch = self.positive_pairs[batch_start:batch_start+half_batch]
#         negative_batch = self.negative_pairs[batch_start:batch_start+half_batch]

#         # 合并并打乱批次内样本
#         batch_data = positive_batch + negative_batch
#         np.random.shuffle(batch_data)

#         # 解包批次数据并转换为数组
#         anchors_touch, anchors_git, contrastive_touch, contrastive_git, labels = zip(*batch_data)
#         return (
#             np.array(anchors_touch),
#             np.array(anchors_git),
#             np.array(contrastive_touch),
#             np.array(contrastive_git)
#         ), np.array(labels)

class VerificationDataGeneratorForTest(Sequence):
    """用于认证网络的动态数据生成器（测试集，逻辑完全不变）"""
    def __init__(self, touch_file, git_file, PIN, user, users, time=2, batch_size=32):
        self.touch_file = touch_file
        self.git_file = git_file
        self.PIN = PIN
        self.user = user
        self.users = users
        self.batch_size = batch_size
        self.positive_pairs = []  # 存储正样本对
        self.negative_pairs = []  # 存储负样本对
        self.test_time = time  # 测试样本时间段（当前传入时间段）
        self.valid = True  # 新增：初始化valid属性为True

        # 加载完整数据
        touch_df = pd.read_csv(touch_file)
        git_df = pd.read_csv(git_file)

        # -------------------------- 核心修改1：拆分锚点时间段（time-1）和测试时间段（time）数据 --------------------------
        # 1. 锚点数据（time-1）：用于正负样本的锚点

        # 2. 测试数据（time）：用于正样本的对比样本
        touch_test_df = touch_df[touch_df['TimePeriod'] == self.test_time]
        git_test_df = git_df[git_df['TimePeriod'] == self.test_time]

        # 3. 筛选当前PIN下的所有数据（负样本需要其他用户的所有时间段）
        touch_pin_all = touch_df[touch_df['PIN'] == PIN]
        git_pin_all = git_df[git_df['PIN'] == PIN]

        # 4. 筛选当前PIN下的锚点/测试数据
      
        touch_test_pin = touch_test_df[touch_test_df['PIN'] == PIN]
        git_test_pin = git_test_df[git_test_df['PIN'] == PIN]

        # 5. 筛选当前用户的锚点样本（time-1）和测试样本（time）（正样本专用）
        self.touch_test_user = touch_test_pin[touch_test_pin['UUID'] == self.user]
        self.git_test_user = git_test_pin[git_test_pin['UUID'] == self.user]

        # -------------------------- 生成正负样本 --------------------------
        # 生成正样本（锚点：time-1，测试：time）
        self._generate_positive_pairs()

        # 生成负样本（锚点：time-1，对比：其他用户所有时间段）
        self._generate_negative_pairs(touch_pin_all, git_pin_all)

        if not self.positive_pairs and not self.negative_pairs:
            self.valid = False
            print(f"警告：用户{self.user}在PIN{PIN}的时间段{self.test_time}无任何有效样本对，已跳过")
            return

        # -------------------------- 核心修改3：强制正负样本数量相同 --------------------------
        self._align_pos_neg_pair_counts()
        if not self.valid:
            return

    def _generate_positive_pairs(self):
        """为单个用户生成正样本对（修改后：从有效样本中两两组合形成正样本对）"""
        # 获取当前用户锚点样本（time-1）和测试样本（time）的共同Sample ID（保证样本有效性）
        test_sample_ids = set(self.touch_test_user['Sample ID']).intersection(self.git_test_user['Sample ID'])

        if len(test_sample_ids) == 0:
            print("警告：当前用户锚点时间段或测试时间段无有效样本，无法生成正样本")
            return
    
        # 将集合转换为有序列表，方便后续两两组合遍历
        valid_sample_ids = list(test_sample_ids)
    
        # 锚点样本（time-1）作为anchors，测试样本（time）作为contrastive（正样本对）
        # 双层循环实现所有有效样本的两两组合（i≠j，避免自身与自身配对）
        for i in range(len(valid_sample_ids)):
            anchor_sid = valid_sample_ids[i]  # 确定锚点样本ID
            for j in range(len(valid_sample_ids)):
                if i == j:
                    continue  # 跳过自身配对，只保留不同样本间的两两组合
                test_sid = valid_sample_ids[j]  # 确定测试样本ID
            
                # 提取锚点样本特征（time-1）
                touch_anchor = self.touch_test_user[self.touch_test_user['Sample ID'] == anchor_sid]
                git_anchor = self.git_test_user[self.git_test_user['Sample ID'] == anchor_sid]
                touch_anchor_feat = touch_anchor[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                git_anchor_feat = git_anchor[['Time', 'X', 'Y', 'Z']].values

                # 提取测试样本特征（time）
                touch_test = self.touch_test_user[self.touch_test_user['Sample ID'] == test_sid]
                git_test = self.git_test_user[self.git_test_user['Sample ID'] == test_sid]
                touch_test_feat = touch_test[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                git_test_feat = git_test[['Time', 'X', 'Y', 'Z']].values

                # 正样本标签为1，格式：(锚点touch, 锚点git, 测试touch, 测试git, 标签)
                self.positive_pairs.append((touch_anchor_feat, git_anchor_feat, touch_test_feat, git_test_feat, 1))

    def _generate_negative_pairs(self, touch_pin_all, git_pin_all):
        """为单个用户生成负样本对（需求2：锚点=time-1，对比=其他用户所有时间段）"""
        # 1. 获取当前用户锚点样本（time-1）的有效Sample ID
        test_sample_ids = set(self.touch_test_user['Sample ID']).intersection(self.git_test_user['Sample ID'])

        if len(test_sample_ids) == 0:
            print("警告：当前用户锚点时间段无有效样本，无法生成负样本")
            return

        # 2. 筛选当前PIN下除当前用户外的其他用户（保证负样本用户有效性）
        other_users = list(set(touch_pin_all['UUID']).intersection(git_pin_all['UUID']) - {self.user})
        if len(other_users) == 0:
            print("警告：无其他用户数据，无法生成负样本")
            return

        # 3. 锚点样本（time-1）作为anchors，其他用户所有时间段样本作为contrastive（负样本对）
        for anchor_sid in test_sample_ids:
            # 提取当前锚点样本特征（time-1）
            touch_anchor = self.touch_test_user[self.touch_test_user['Sample ID'] == anchor_sid]
            git_anchor = self.git_test_user[self.git_test_user['Sample ID'] == anchor_sid]
            touch_anchor_feat = touch_anchor[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            git_anchor_feat = git_anchor[['Time', 'X', 'Y', 'Z']].values

            # 4. 随机选择5个其他用户（保持与原代码逻辑一致的负样本数量规模）
            selected_other_users = np.random.choice(other_users, size=5, replace=False) if len(other_users) >= 2 else other_users

            for other_user in selected_other_users:
                # 5. 提取其他用户的**所有时间段**数据（需求2核心：不再限制时间段）
                other_touch_all = touch_pin_all[touch_pin_all['UUID'] == other_user]
                other_git_all = git_pin_all[git_pin_all['UUID'] == other_user]

                # 6. 找到其他用户所有时间段的有效Sample ID（touch和git共有的）
                other_valid_sids = set(other_touch_all['Sample ID']).intersection(other_git_all['Sample ID'])
                if len(other_valid_sids) == 0:
                    continue

                # 7. 随机选择一个其他用户的有效样本
                other_sid = np.random.choice(list(other_valid_sids))
                other_touch = other_touch_all[other_touch_all['Sample ID'] == other_sid]
                other_git = other_git_all[other_git_all['Sample ID'] == other_sid]

                # 8. 提取负样本对比特征
                touch_other_feat = other_touch[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                git_other_feat = other_git[['Time', 'X', 'Y', 'Z']].values

                # 负样本标签为0
                self.negative_pairs.append((touch_anchor_feat, git_anchor_feat, touch_other_feat, git_other_feat, 0))

    def _align_pos_neg_pair_counts(self):
        min_pair_count = min(len(self.positive_pairs), len(self.negative_pairs))
        if min_pair_count == 0:
            self.valid = False
            print(f"警告：用户{self.user}在PIN{PIN}的时间段{self.test_time}正负样本中至少有一个为空，无法对齐，已跳过")
            return
    
        self.positive_pairs = self.positive_pairs[:min_pair_count]
        self.negative_pairs = self.negative_pairs[:min_pair_count]
        print(f"正负样本数量已对齐，各{min_pair_count}对")

    def __len__(self):
        """计算批次数量：基于对齐后的样本数量，保证批次完整性"""
        total_valid_pairs = len(self.positive_pairs) + len(self.negative_pairs)
        return int(total_valid_pairs / self.batch_size)

    def on_epoch_end(self):
        """在每个epoch结束时打乱正负样本（保证训练随机性）"""
        np.random.shuffle(self.positive_pairs)
        np.random.shuffle(self.negative_pairs)

    def __getitem__(self, idx):
        """获取指定批次的数据，保证批次内正负样本各占一半"""
        half_batch = self.batch_size // 2
        batch_start = idx * half_batch

        # 提取当前批次的正负样本（各半）
        positive_batch = self.positive_pairs[batch_start:batch_start+half_batch]
        negative_batch = self.negative_pairs[batch_start:batch_start+half_batch]

        # 合并并打乱批次内样本
        batch_data = positive_batch + negative_batch
        np.random.shuffle(batch_data)

        # 解包批次数据并转换为数组
        anchors_touch, anchors_git, contrastive_touch, contrastive_git, labels = zip(*batch_data)
        return (
            np.array(anchors_touch),
            np.array(anchors_git),
            np.array(contrastive_touch),
            np.array(contrastive_git)
        ), np.array(labels)
    
# ======================== 击键特征提取器模型 ========================
def build_keystroke_cnn(input_shape=(input_data_length_touch, input_feature_length_touch), dropout=0.5):
    input_layer = layers.Input(shape=input_shape, name="base_network_input")
    # 对特征进行卷积处理
    x = layers.Conv1D(64, 7, padding='same', activation='relu', strides= 1)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu', strides= 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(192, 3, padding='same', activation='relu', strides= 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu', strides= 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    #输出
    output = layers.Dense(128, name="base_network_output")(x)
    return models.Model(inputs=input_layer, outputs=output, name="touch_base_network")

def build_git_cnn(input_shape=(input_data_length_git, input_feature_length_git), dropout=0.5):
    input_layer = layers.Input(shape=input_shape, name="base_network_input")
    x = layers.Conv1D(64, 7, padding='same', activation='relu', strides= 1)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu', strides= 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(192, 3, padding='same', activation='relu', strides= 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu', strides= 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(128, name="base_network_output")(x)
    return models.Model(inputs=input_layer, outputs=output, name="git_base_network")

def triplet_loss(y_pred, alpha=0.5):
    """三元组损失函数"""
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(0.0, positive_dist - negative_dist + alpha)
    return loss

def build_triplet_model(base_network, input_data_length, input_feature_length, alpha=0.5):
    """构建完整的三元组模型"""
    anchor_input = layers.Input(shape=(input_data_length, input_feature_length))
    positive_input = layers.Input(shape=(input_data_length, input_feature_length))
    negative_input = layers.Input(shape=(input_data_length, input_feature_length))
    anchor_embedding = base_network(anchor_input)
    positive_embedding = base_network(positive_input)
    negative_embedding = base_network(negative_input)
    # 直接计算损失值
    loss = triplet_loss([anchor_embedding, positive_embedding, negative_embedding], alpha=alpha)
    model = models.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=loss
    )
    model.add_loss(loss)  # 显式添加损失
    return model

def calculate_eer_and_threshold(fpr, tpr, thresholds):
    interp_tpr = interp1d(fpr, tpr, kind='linear', assume_sorted=True)
    eer_val = brentq(lambda x: 1. - x - interp_tpr(x), 0.0, 1.0)
    fnr = 1 - tpr
    fnr_at_eer = 1 - interp_tpr(eer_val)
    distances = np.abs(fpr - eer_val) + np.abs(fnr - fnr_at_eer)
    idx = np.argmin(distances)
    return eer_val, thresholds[idx]

def build_feature_fusion_network(input_dim = 256):
    """构建认证网络"""
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    raw_vec = layers.Dense(256, activation=None)(x)
    # L2 归一化，使每个样本的向量长度为 1
    output = tf.nn.l2_normalize(raw_vec, axis=-1)
    return models.Model(inputs=input_layer, outputs=output, name="feature_fusion_network")

# ======================== 认证网络 ========================
def build_verification_network_with_fusion(touch_base, git_base, feature_fusion_model):
    """构建带跨模态融合的认证网络"""
    # 输入层
    input_a_touch = layers.Input(shape=(input_data_length_touch, input_feature_length_touch))
    input_a_git = layers.Input(shape=(input_data_length_git, input_feature_length_git))
    input_b_touch = layers.Input(shape=(input_data_length_touch, input_feature_length_touch))
    input_b_git = layers.Input(shape=(input_data_length_git, input_feature_length_git))
    # 冻结特征提取器
    touch_base.trainable = False
    git_base.trainable = False
    # 处理样本A
    emb_a_touch = touch_base(input_a_touch)
    emb_a_git = git_base(input_a_git)
    # 处理样本B
    emb_b_touch = touch_base(input_b_touch)
    emb_b_git = git_base(input_b_git)

    emb_a_features = layers.Concatenate()([emb_a_touch, emb_a_git])
    emb_b_features = layers.Concatenate()([emb_b_touch, emb_b_git])

    fusion_features_a = feature_fusion_model(emb_a_features)
    fusion_features_b = feature_fusion_model(emb_b_features)
    diff = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([fusion_features_a, fusion_features_b])

    # 认证网络
    x = layers.Dense(256, activation='relu')(diff)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(
        inputs=[input_a_touch, input_a_git, input_b_touch, input_b_git],
        outputs=output,
        name="verification_network_with_fusion"
    )
# 添加学习率调度回调类
class LearningRateScheduler(Callback):
    def __init__(self, reduce_after_epoch=20, reduce_factor=0.1):
        super().__init__()
        self.reduce_after_epoch = reduce_after_epoch
        self.reduce_factor = reduce_factor
        self.reduced = False  # 标记是否已经降低过学习率

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.reduce_after_epoch and not self.reduced:
            current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            new_lr = current_lr * self.reduce_factor
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            self.reduced = True  # 标记为已降低
            print(f"\nEpoch {epoch + 1}: 学习率降低为 {new_lr:.6f}")
# ======================== 主流程 ========================
# ======================== 主流程 ========================
if __name__ == "__main__":
    for account in range(1, 2):
        total_epochs = 30  # 总训练轮数设为40，大于20
        reduce_after_epoch = 20  # 20轮后降低学习率
        siamese_param_grid = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'alpha': 0.01,
            'dropout': 0.5
        }
        verification_param_grid = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'dropout': 0.5
        }

        # # 构建触摸双胞胎网络
        # base_touch_network = build_keystroke_cnn(dropout=siamese_param_grid['dropout'])
        # touch_triplet_model = build_triplet_model(base_touch_network, input_data_length_touch,
        #                                           input_feature_length_touch, alpha=siamese_param_grid['alpha'])
        # touch_triplet_model.compile(optimizer=optimizers.Adam(learning_rate=siamese_param_grid['learning_rate']))
        # train_gen_touch = TripletDataGeneratorForTouch(train_csv_files_touch, train_users,
        #                                                batch_size=siamese_param_grid['batch_size'])
        # lr_scheduler = LearningRateScheduler(reduce_after_epoch=reduce_after_epoch)
        # touch_triplet_model.fit(train_gen_touch, epochs=total_epochs, callbacks=[lr_scheduler])
        # touch_triplet_model.save(f'RQ_model/touch_triplet_model.keras')
        
        # # 构建步态双胞胎网络
        # base_git_network = build_git_cnn(dropout=siamese_param_grid['dropout'])
        # git_triplet_model = build_triplet_model(base_git_network, input_data_length_git, input_feature_length_git,
        #                                         alpha=siamese_param_grid['alpha'])
        # git_triplet_model.compile(optimizer=optimizers.Adam(learning_rate=siamese_param_grid['learning_rate']))
        # train_gen_git = TripletDataGeneratorForGit(train_csv_files_git, train_users,
        #                                            batch_size=siamese_param_grid['batch_size'])
        # lr_scheduler = LearningRateScheduler(reduce_after_epoch=reduce_after_epoch)
        # git_triplet_model.fit(train_gen_git, epochs=total_epochs, callbacks=[lr_scheduler])
        # git_triplet_model.save(f'RQ_model/git_triplet_model.keras')
        
        # touch_base_network = touch_triplet_model.get_layer("touch_base_network")
        # git_base_network = git_triplet_model.get_layer("git_base_network")
        # feature_fusion_module = build_feature_fusion_network()

        # verification_model = build_verification_network_with_fusion(
        #     touch_base_network,
        #     git_base_network,
        #     feature_fusion_module
        # )
        # train_gen = VerificationDataGenerator(train_csv_files_touch, train_csv_files_git, train_users,
        #                                       batch_size=verification_param_grid['batch_size'])
        # verification_model.compile(optimizer=optimizers.Adam(learning_rate=verification_param_grid['learning_rate']),
        #                      loss='binary_crossentropy',
        #                      metrics=['accuracy'])
        # lr_scheduler = LearningRateScheduler(reduce_after_epoch=reduce_after_epoch)
        # verification_model.fit(train_gen, epochs=total_epochs, callbacks=[lr_scheduler])
        # #保存整体的模型
        # verification_model.save(f'RQ_model/full_model.keras')

        base_git_network = build_git_cnn(dropout=siamese_param_grid['dropout'])
        git_triplet_model = build_triplet_model(base_git_network, input_data_length_git, input_feature_length_git,
                                                alpha=siamese_param_grid['alpha'])
        git_triplet_model = tf.keras.models.load_model('RQ_model/git_triplet_model.keras',safe_mode=False)
        
        base_touch_network = build_keystroke_cnn(dropout=siamese_param_grid['dropout'])
        touch_triplet_model = build_triplet_model(base_touch_network, input_data_length_touch,
                                                  input_feature_length_touch, alpha=siamese_param_grid['alpha'])
        touch_triplet_model = tf.keras.models.load_model('RQ_model/touch_triplet_model.keras',safe_mode=False)

        touch_base_network = touch_triplet_model.get_layer("touch_base_network")
        git_base_network = git_triplet_model.get_layer("git_base_network")
        feature_fusion_module = build_feature_fusion_network()

        verification_model = build_verification_network_with_fusion(
            touch_base_network,
            git_base_network,
            feature_fusion_module
        )
        verification_model = tf.keras.models.load_model('RQ_model/full_model.keras',safe_mode=False)

        # 5. 测试每个CSV文件并保存结果（核心修改：PIN循环结束后立即保存）
        test_results = {}
        output_dir = 'RQ_results'  # 测试结果保存目录
        os.makedirs(output_dir, exist_ok=True)

        for touch_file in test_csv_files_touch:
            if touch_file.endswith("_sit.csv"):
                git_file_suffix = "_sit.csv"
                posture = "sit"
            elif touch_file.endswith("_walk.csv"):
                git_file_suffix = "_walk.csv"
                posture = "walk"
            else:
                continue  # 跳过不匹配的文件

            git_file = next((f for f in test_csv_files_git if f.endswith(git_file_suffix)), None)
            if not git_file:
                continue  # 如果没有找到对应的 git_file，跳过

            touch_df = pd.read_csv(touch_file)
            git_df = pd.read_csv(git_file)
            common_PINs = set(touch_df['PIN']).intersection(git_df['PIN'])

            for PIN in common_PINs:
                # completePINs = ["10140", "111111", "122914", "141231", "201499", "585520", "992914"]
                # completePINs2 = [10140, 111111, 122914, 141231, 201499, 585520, 992914]
                # if PIN in completePINs or PIN in completePINs2:
                #     continue
                # 初始化当前PIN的数据存储（避免跨PIN数据堆积）
                pin_export = {
                    "TimeStamp": [], "Type": [], "User": [], "Posture": [], "PIN": [],
                    "Fpr": [], "Tpr": [], "EER": [], "EER_Threshold": [], "Thresholds": [],
                    "Labels": [], "Predictions": []
                }

                # 验证集用户处理
                for user in val_users:
                    test_gen = VerificationDataGeneratorForVal(touch_file, git_file, PIN, user, val_users,
                                                                batch_size=2)
                    predictions = []
                    labels = []
                    for i in range(len(test_gen)):
                        (anchors_touch, anchors_git, contrastive_touch, contrastive_git), batch_labels = test_gen[i]
                        batch_predictions = verification_model.predict(
                            (anchors_touch, anchors_git, contrastive_touch, contrastive_git),
                            verbose=0)
                        predictions.extend(batch_predictions.flatten())
                        labels.extend(batch_labels)
                    # 确保labels不为空
                    if len(labels) == 0:
                        continue
                    fpr, tpr, thresholds = roc_curve(labels, predictions)
                    eer, eer_threshold = calculate_eer_and_threshold(fpr, tpr, thresholds)
                    
                    # 写入当前PIN的数据存储
                    pin_export["TimeStamp"].append(1)
                    pin_export["Type"].append('val')
                    pin_export["User"].append(user)
                    pin_export["Posture"].append(posture)
                    pin_export["PIN"].append(PIN)
                    pin_export["Fpr"].append(fpr)
                    pin_export["Tpr"].append(tpr)
                    pin_export["EER"].append(eer)
                    pin_export["EER_Threshold"].append(eer_threshold)
                    pin_export["Thresholds"].append(thresholds)
                    pin_export["Labels"].append(labels)
                    pin_export["Predictions"].append(predictions)

                # 测试集用户处理
                for user in test_users:
                    for timeStamp in range(2, 5):
                        test_gen = VerificationDataGeneratorForTest(touch_file, git_file, PIN, user, test_users, time = timeStamp, 
                                                                batch_size=2)
                        if not test_gen.valid:
                            continue  # 跳过无效的生成器
                        predictions = []
                        labels = []
                        for i in range(len(test_gen)):
                            (anchors_touch, anchors_git, contrastive_touch, contrastive_git), batch_labels = test_gen[i]
                            batch_predictions = verification_model.predict(
                                (anchors_touch, anchors_git, contrastive_touch, contrastive_git),
                                verbose=0)
                            predictions.extend(batch_predictions.flatten())
                            labels.extend(batch_labels)
                        # 确保labels不为空
                        if len(labels) == 0:
                            continue
                        fpr, tpr, thresholds = roc_curve(labels, predictions)
                        eer, eer_threshold = calculate_eer_and_threshold(fpr, tpr, thresholds)
                        
                        # 写入当前PIN的数据存储
                        pin_export["TimeStamp"].append(timeStamp)
                        pin_export["Type"].append('test')
                        pin_export["User"].append(user)
                        pin_export["Posture"].append(posture)
                        pin_export["PIN"].append(PIN)
                        pin_export["Fpr"].append(fpr)
                        pin_export["Tpr"].append(tpr)
                        pin_export["EER"].append(eer)
                        pin_export["EER_Threshold"].append(eer_threshold)
                        pin_export["Thresholds"].append(thresholds)
                        pin_export["Labels"].append(labels)
                        pin_export["Predictions"].append(predictions)

                # -------------------------- 核心修改：当前PIN循环结束后，立即保存数据 --------------------------
                # 1. 转换当前PIN数据为DataFrame
                pin_df = pd.DataFrame(pin_export)
                results_file_ftt = os.path.join(output_dir, f'0_test_results_{PIN}.csv')
                # 2. 保存策略：
                #    - 若文件不存在，直接写入（含表头）
                #    - 若文件已存在，追加写入（不含表头）
                if not os.path.exists(results_file_ftt):
                    pin_df.to_csv(results_file_ftt, index=False, header=True)
                else:
                    pin_df.to_csv(results_file_ftt, index=False, header=False, mode='a')
                # 4. 清空当前PIN数据存储，释放内存
                del pin_export
                print(f"已完成PIN {PIN} 的数据处理并保存至 {results_file_ftt}")
