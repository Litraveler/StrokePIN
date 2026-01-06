import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics
from tensorflow.keras.utils import Sequence
from sklearn.metrics import roc_curve
import itertools
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from itertools import combinations
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

#！！！！！！！！！！！！！！！！！！！！优化超参数！！！！！！！！！！！！！！！！！！！！！
#！！！！！！！！！！！！！！！测试模型在不同数量数据集下的性能！！！！！！！！！！！！！！！！！！！！！
input_data_length_touch = 90
input_feature_length_touch = 8

input_data_length_git = 160
input_feature_length_git = 4

# ======================== 数据准备与用户划分 ========================
def load_user_ids(dirs):
    """从指定目录提取所有用户ID"""
    user_ids = set()
    for file_path in dirs:
        df = pd.read_csv(file_path)
        user_ids.update(df['UUID'].unique())
    return list(user_ids)

# 提取用户ID并划分训练/测试用户
all_users = load_user_ids(['datas/masked_normalized_filtered_touch_data_sit.csv', 'datas/masked_normalized_filtered_touch_data_walk.csv', 'datas/masked_normalized_filtered_sensor_data_sit.csv', 'datas/masked_normalized_filtered_sensor_data_walk.csv'])
# 1) 先把 20% 分出来做测试集
train_val, test_users = train_test_split(all_users, test_size=0.20, random_state=42)

# 2) 再把剩余 80% 中的 25%（0.25×0.8 = 0.2）做验证集，剩下 75%（0.75×0.8 = 0.6）做训练集
train_users, val_users = train_test_split(train_val, test_size=0.25, random_state=42)

# ======================== 获取所有CSV文件路径 ========================
csv_files_touch = ['datas/masked_normalized_filtered_touch_data_sit.csv', 'datas/masked_normalized_filtered_touch_data_walk.csv']
csv_files_git = ['datas/masked_normalized_filtered_sensor_data_sit.csv', 'datas/masked_normalized_filtered_sensor_data_walk.csv']

class VerificationDataGenerator(Sequence):
    """用于认证网络的动态数据生成器"""
    def __init__(self, csv_files, users, label, batch_size=32):
        self.csv_files = csv_files
        self.users = users
        self.label = label
        self.batch_size = batch_size
        self.positive_pairs = []  # 存储正样本对
        self.negative_pairs = []  # 存储负样本对
        # 处理每个文件对
        for file in self.csv_files:
            df = pd.read_csv(file)
            for PIN in df['PIN'].unique():
                pin_group = df[df['PIN'] == PIN]
                # 处理该 PIN 组的数据
                for user in self.users:
                    # 筛选 touch 和 git 中属于该用户的样本
                    user_samples = pin_group[pin_group['UUID'] == user]

                    # 如果 touch 或 git 中没有该用户的样本，跳过
                    if user_samples.empty:
                        continue
                    # 生成正样本对
                    self._generate_positive_pairs(user_samples, self.label)

                    # 生成负样本对（从其他用户的样本中随机选择）
                    self._generate_negative_pairs(user_samples, user, pin_group, self.label)

    def _generate_positive_pairs(self, user_samples, label):
        """为单个用户生成正样本对"""
        # 找到 touch 和 git 中共有相同的 Sample ID
        sample_ids = set(user_samples['Sample ID'])
        if len(sample_ids) == 0:
            return []
        # 遍历所有可能的 Sample ID 组合
        for sample_id1, sample_id2 in combinations(sample_ids, 2):
            # 从 touch 中获取第一个 Sample ID 的样本
            samples1 = user_samples[user_samples['Sample ID'] == sample_id1]
            # 从 touch 中获取第二个 Sample ID 的样本
            samples2 = user_samples[user_samples['Sample ID'] == sample_id2]
            if label == "touch":
                # 提取 touch 样本的特征
                features1 = samples1[
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                features2 = samples2[
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            else:
                # 提取 git 样本的特征
                features1 = samples1[['Time', 'X', 'Y', 'Z']].values
                features2 = samples2[['Time', 'X', 'Y', 'Z']].values

            # 将提取的特征添加到正样本列表
            self.positive_pairs.append((features1, features2, 1))

    def _generate_negative_pairs(self, user_samples, user, pin_group, label):
        """为单个用户生成负样本对"""
        # 找到 touch 和 git 中共有相同的 Sample ID
        sample_ids = set(user_samples['Sample ID'])

        other_users = list(set(pin_group['UUID']).intersection(pin_group['UUID']) - {user})
        if len(other_users) < 2:
            return  # 如果没有足够的其他用户，跳过
        for sample_id in sample_ids:
            other_users_idx = np.random.choice(len(other_users), 2)  # 先随机选择索引
            # 从 touch 和 git 中获取该 Sample ID 的样本
            samples = user_samples[user_samples['Sample ID'] == sample_id]
            if label == "touch":
                features1 = samples[
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            else:
                features1 = samples[['Time', 'X', 'Y', 'Z']].values

            for other_user_idx in other_users_idx:
                other_user = other_users[other_user_idx]
                # 找到其他用户的 touch 和 git 中具有相同 Sample ID 的样本
                other_samples = pin_group[(pin_group['UUID'] == other_user)]
                # 找到 touch 和 git 中具有相同 Sample ID 的样本
                ids_other_user = set(other_samples['Sample ID'])
                if not ids_other_user:
                    continue  # 如果没有共同的 Sample ID，跳过
                # 随机选择一个 Sample ID
                selected_sample_id = np.random.choice(list(ids_other_user))

                # 获取其他用户的 touch 和 git 样本
                neg_sample = other_samples[other_samples['Sample ID'] == selected_sample_id]
                # 随机选择一个样本
                # 提取其他用户的 touch 和 git 样本的特征
                if label == "touch":
                    features2 = neg_sample[
                        ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                else:
                    features2 = neg_sample[['Time', 'X', 'Y', 'Z']].values
                # 将提取的特征添加到负样本列表
                self.negative_pairs.append((features1, features2, 0))

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

        # 合并正负样本

        batch_pairs = []
        for positive_pair in positive_batch:
            batch_pairs.append(positive_pair)
        for negative_pair in negative_batch:
            batch_pairs.append(negative_pair)

        # 打乱批次顺序
        np.random.shuffle(batch_pairs)
        anchors, contrastive, labels = zip(*batch_pairs)
        anchors_array = np.array(anchors)
        contrastive_array = np.array(contrastive)
        labels_array = np.array(labels)
        # 返回批次数据
        return (anchors_array, contrastive_array), labels_array

class VerificationDataGeneratorForTest(Sequence):
    """用于认证网络的动态数据生成器"""
    def __init__(self, file, PIN, users, label, batch_size=32):
        self.files = file
        self.PIN = PIN
        self.users = users
        self.label = label
        self.batch_size = batch_size
        self.positive_pairs = []  # 存储正样本对
        self.negative_pairs = []  # 存储负样本对

        # 处理每个文件对
        df = pd.read_csv(file)
        pin_group = df[df['PIN'] == PIN]
        # 处理该 PIN 组的数据
        for user in self.users:
            # 筛选 touch 和 git 中属于该用户的样本
            user_samples = pin_group[pin_group['UUID'] == user]
            # 如果 touch 或 git 中没有该用户的样本，跳过
            if user_samples.empty:
                continue
            # 生成正样本对
            self._generate_positive_pairs(user_samples, self.label)
            # 生成负样本对（从其他用户的样本中随机选择）
            self._generate_negative_pairs(user_samples, user, pin_group, self.label)

    def _generate_positive_pairs(self, user_samples, label):
        """为单个用户生成正样本对"""
        # 找到 touch 和 git 中共有相同的 Sample ID
        sample_ids = set(user_samples['Sample ID'])
        if len(sample_ids) == 0:
            return []
        # 遍历所有可能的 Sample ID 组合
        for sample_id1, sample_id2 in combinations(sample_ids, 2):
            # 从 touch 中获取第一个 Sample ID 的样本
            samples1 = user_samples[user_samples['Sample ID'] == sample_id1]
            # 从 touch 中获取第二个 Sample ID 的样本
            samples2 = user_samples[user_samples['Sample ID'] == sample_id2]
            if label == "touch":
                # 提取 touch 样本的特征
                features1 = samples1[
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                features2 = samples2[
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            else:
                # 提取 git 样本的特征
                features1 = samples1[['Time', 'X', 'Y', 'Z']].values
                features2 = samples2[['Time', 'X', 'Y', 'Z']].values

            # 将提取的特征添加到正样本列表
            self.positive_pairs.append((features1, features2, 1))

    def _generate_negative_pairs(self, user_samples, user, pin_group, label):
        """为单个用户生成负样本对"""
        # 找到 touch 和 git 中共有相同的 Sample ID
        sample_ids = set(user_samples['Sample ID'])

        other_users = list(set(pin_group['UUID']).intersection(pin_group['UUID']) - {user})
        if len(other_users) < 2:
            return  # 如果没有足够的其他用户，跳过
        for sample_id in sample_ids:
            other_users_idx = np.random.choice(len(other_users), 2)  # 先随机选择索引
            # 从 touch 和 git 中获取该 Sample ID 的样本
            samples = user_samples[user_samples['Sample ID'] == sample_id]
            if label == "touch":
                features1 = samples[
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            else:
                features1 = samples[['Time', 'X', 'Y', 'Z']].values

            for other_user_idx in other_users_idx:
                other_user = other_users[other_user_idx]
                # 找到其他用户的 touch 和 git 中具有相同 Sample ID 的样本
                other_samples = pin_group[(pin_group['UUID'] == other_user)]
                # 找到 touch 和 git 中具有相同 Sample ID 的样本
                ids_other_user = set(other_samples['Sample ID'])
                if not ids_other_user:
                    continue  # 如果没有共同的 Sample ID，跳过
                # 随机选择一个 Sample ID
                selected_sample_id = np.random.choice(list(ids_other_user))

                # 获取其他用户的 touch 和 git 样本
                neg_sample = other_samples[other_samples['Sample ID'] == selected_sample_id]
                # 随机选择一个样本
                # 提取其他用户的 touch 和 git 样本的特征
                if label == "touch":
                    features2 = neg_sample[
                        ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                else:
                    features2 = neg_sample[['Time', 'X', 'Y', 'Z']].values
                # 将提取的特征添加到负样本列表
                self.negative_pairs.append((features1, features2, 0))

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

        # 合并正负样本

        batch_pairs = []
        for positive_pair in positive_batch:
            batch_pairs.append(positive_pair)
        for negative_pair in negative_batch:
            batch_pairs.append(negative_pair)

        # 打乱批次顺序
        np.random.shuffle(batch_pairs)
        anchors, contrastive, labels = zip(*batch_pairs)
        anchors_array = np.array(anchors)
        contrastive_array = np.array(contrastive)
        labels_array = np.array(labels)
        # 返回批次数据
        return (anchors_array, contrastive_array), labels_array

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
    x = layers.Dense(512, activation='relu')(x)
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
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(128, name="base_network_output")(x)
    return models.Model(inputs=input_layer, outputs=output, name="git_base_network")

def triplet_loss(y_pred, alpha=0.5):
    """三元组损失函数"""
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(0.0, alpha + positive_dist - negative_dist)
    return tf.reduce_mean(loss)

def build_triplet_model(base_network, input_data_length, input_feature_length, alpha=0.5,):
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

# ======================== 认证网络 ========================
def build_verification_network(input_dim, dropout_rate):
    """构建认证网络"""
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x) # 添加 Dropout 层，丢弃率设为 0.5
    output = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=input_layer, outputs=output, name="verification_base_network")

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
if __name__ == "__main__":
    #测试不同模态的表现
    for account in range(1, 11):
        total_epochs = 30  # 总训练轮数设为40，大于20
        reduce_after_epoch = 20  # 20轮后降低学习率
        siamese_param_grid = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'alpha': 0.01,
            'dropout': 0.5
        }
        label = "with_feature_fusion"
        with custom_object_scope({'triplet_loss': triplet_loss}):
            touch_triplet_model = load_model(f'E1_model/touch_triplet_model_with_feature_fusion_{account}.keras', compile=True)
            git_triplet_model = load_model(f'E1_model/git_triplet_model_with_feature_fusion_{account}.keras', compile=True)
        touch_base_network = touch_triplet_model.get_layer("touch_base_network")
        git_base_network = git_triplet_model.get_layer("git_base_network")

        verification_param_grid = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'dropout': 0.5
        }
        # 4. 训练认证网络
        input_a_touch = layers.Input(shape=(input_data_length_touch, input_feature_length_touch))
        input_a_git = layers.Input(shape=(input_data_length_git, input_feature_length_git))
        input_b_touch = layers.Input(shape=(input_data_length_touch, input_feature_length_touch))
        input_b_git = layers.Input(shape=(input_data_length_git, input_feature_length_git))

        # 使用预训练的特征提取器（固定参数）
        embedding_a_touch = touch_base_network(input_a_touch)
        embedding_a_git = git_base_network(input_a_git)
        embedding_b_touch = touch_base_network(input_b_touch)
        embedding_b_git = git_base_network(input_b_git)

        touch_base_network.trainable = False  # 固定特征提取器参数
        git_base_network.trainable = False
        # 计算差值并拼接
        touch_diff = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([embedding_a_touch, embedding_b_touch])
        git_diff = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([embedding_a_git, embedding_b_git])
        # 认证网络
        touch_verification_model = build_verification_network(touch_diff.shape[-1],
                                                              verification_param_grid['dropout'])
        touch_output = touch_verification_model(touch_diff)
        git_verification_model = build_verification_network(git_diff.shape[-1],
                                                            verification_param_grid['dropout'])
        git_output = git_verification_model(git_diff)
        touch_full_model = models.Model(inputs=[input_a_touch, input_b_touch], outputs=touch_output)
        git_full_model = models.Model(inputs=[input_a_git, input_b_git], outputs=git_output)
        touch_full_model.compile(optimizer=optimizers.Adam(learning_rate=verification_param_grid['learning_rate']),
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'])
        git_full_model.compile(optimizer=optimizers.Adam(learning_rate=verification_param_grid['learning_rate']),
                               loss='binary_crossentropy',
                               metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(reduce_after_epoch=reduce_after_epoch)
        touch_train_gen = VerificationDataGenerator(csv_files_touch, train_users, "touch",
                                                    batch_size=verification_param_grid['batch_size'])
        git_train_gen = VerificationDataGenerator(csv_files_git, train_users, "git",
                                                  batch_size=verification_param_grid['batch_size'])
        touch_full_model.fit(touch_train_gen, callbacks=[lr_scheduler])
        git_full_model.fit(git_train_gen, epochs = total_epochs, callbacks=[lr_scheduler])
        # 5. 测试每个CSV文件并保存结果
        touch_test_results = {}
        git_test_results = {}
        touch_export = {"Type": [], "posture": [], "PIN": [], "Fpr": [], "Tpr": [], "EER": [], "Thresholds": [], "Labels": [], "Predictions": []}
        git_export = {"Type": [], "posture": [], "PIN": [], "Fpr": [], "Tpr": [], "EER": [], "Thresholds": [], "Labels": [], "Predictions": []}
        output_dir = 'E3_results'  # 测试结果保存目录
        os.makedirs(output_dir, exist_ok=True)
        for touch_file in csv_files_touch:
            if touch_file.endswith("_sit.csv"):
                git_file_suffix = "_sit.csv"
                posture = "sit"
            elif touch_file.endswith("_walk.csv"):
                git_file_suffix = "_walk.csv"
                posture = "walk"
            else:
                continue  # 跳过不匹配的文件
            git_file = next((f for f in csv_files_git if f.endswith(git_file_suffix)), None)
            if not git_file:
                continue  # 如果没有找到对应的 git_file，跳过

            touch_df = pd.read_csv(touch_file)
            git_df = pd.read_csv(git_file)
            common_PINs = set(touch_df['PIN']).intersection(git_df['PIN'])
            for PIN in common_PINs:
                touch_test_gen = VerificationDataGeneratorForTest(touch_file, PIN, val_users, "touch", batch_size=32)
                git_test_gen = VerificationDataGeneratorForTest(git_file, PIN, val_users, "git", batch_size=32)

                touch_predictions = []
                touch_labels = []
                for i in range(len(touch_test_gen)):
                    (anchors, contrastive), batch_labels = touch_test_gen[i]
                    batch_predictions = touch_full_model.predict(
                        (anchors, contrastive), verbose=0)
                    touch_predictions.extend(batch_predictions.flatten())
                    touch_labels.extend(batch_labels)

                git_predictions = []
                git_labels = []
                for i in range(len(git_test_gen)):
                    (anchors, contrastive), batch_labels = git_test_gen[i]
                    batch_predictions = git_full_model.predict(
                        (anchors, contrastive), verbose=0)
                    git_predictions.extend(batch_predictions.flatten())
                    git_labels.extend(batch_labels)

                git_fpr, git_tpr, git_thresholds = roc_curve(git_labels, git_predictions)

                git_eer = brentq(lambda x: 1. - x - interp1d(git_fpr, git_tpr)(x), 0.0, 1.0)
                git_export["Type"].append("val")
                git_export["posture"].append(posture)
                git_export["PIN"].append(PIN)
                git_export["Fpr"].append(git_fpr)
                git_export["Tpr"].append(git_tpr)
                git_export["EER"].append(git_eer)
                git_export["Thresholds"].append(git_thresholds)
                git_export["Labels"].append(git_labels)
                git_export["Predictions"].append(git_predictions)

                touch_fpr, touch_tpr, touch_thresholds = roc_curve(touch_labels, touch_predictions)

                touch_eer = brentq(lambda x: 1. - x - interp1d(touch_fpr, touch_tpr)(x), 0.0, 1.0)
                touch_export["Type"].append("val")
                touch_export["posture"].append(posture)
                touch_export["PIN"].append(PIN)
                touch_export["Fpr"].append(touch_fpr)
                touch_export["Tpr"].append(touch_tpr)
                touch_export["EER"].append(touch_eer)
                touch_export["Thresholds"].append(touch_thresholds)
                touch_export["Labels"].append(touch_labels)
                touch_export["Predictions"].append(touch_predictions)
            for PIN in common_PINs:
                touch_test_gen = VerificationDataGeneratorForTest(touch_file, PIN, test_users, "touch", batch_size=32)
                git_test_gen = VerificationDataGeneratorForTest(git_file, PIN, test_users, "git", batch_size=32)

                touch_predictions = []
                touch_labels = []
                for i in range(len(touch_test_gen)):
                    (anchors, contrastive), batch_labels = touch_test_gen[i]
                    batch_predictions = touch_full_model.predict(
                        (anchors, contrastive), verbose=0)
                    touch_predictions.extend(batch_predictions.flatten())
                    touch_labels.extend(batch_labels)

                git_predictions = []
                git_labels = []
                for i in range(len(git_test_gen)):
                    (anchors, contrastive), batch_labels = git_test_gen[i]
                    batch_predictions = git_full_model.predict(
                        (anchors, contrastive), verbose=0)
                    git_predictions.extend(batch_predictions.flatten())
                    git_labels.extend(batch_labels)

                git_fpr, git_tpr, git_thresholds = roc_curve(git_labels, git_predictions)

                git_eer = brentq(lambda x: 1. - x - interp1d(git_fpr, git_tpr)(x), 0.0, 1.0)
                git_export["Type"].append("test")
                git_export["posture"].append(posture)
                git_export["PIN"].append(PIN)
                git_export["Fpr"].append(git_fpr)
                git_export["Tpr"].append(git_tpr)
                git_export["EER"].append(git_eer)
                git_export["Thresholds"].append(git_thresholds)
                git_export["Labels"].append(git_labels)
                git_export["Predictions"].append(git_predictions)

                touch_fpr, touch_tpr, touch_thresholds = roc_curve(touch_labels, touch_predictions)

                touch_eer = brentq(lambda x: 1. - x - interp1d(touch_fpr, touch_tpr)(x), 0.0, 1.0)
                touch_export["Type"].append("test")
                touch_export["posture"].append(posture)
                touch_export["PIN"].append(PIN)
                touch_export["Fpr"].append(touch_fpr)
                touch_export["Tpr"].append(touch_tpr)
                touch_export["EER"].append(touch_eer)
                touch_export["Thresholds"].append(touch_thresholds)
                touch_export["Labels"].append(touch_labels)
                touch_export["Predictions"].append(touch_predictions)

        touch_results_file_ftt = os.path.join(output_dir, f'touch_test_results_{account}.csv')
        touch_df = pd.DataFrame(touch_export)
        touch_df.to_csv(touch_results_file_ftt, index=False)

        git_results_file_ftt = os.path.join(output_dir, f'git_test_results_{account}.csv')
        git_df = pd.DataFrame(git_export)
        git_df.to_csv(git_results_file_ftt, index=False)


















