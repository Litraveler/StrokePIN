import pandas as pd
import numpy as np

# 非时间序列数据
# 最大touch长度: 52
# 最大sensor长度: 122
# 最小touch长度: 13
# 最小sensor长度: 29
# 最大touch长度: 64
# 最大sensor长度: 119
# 最小touch长度: 14
# 最小sensor长度: 26

# 时间序列数据
# 最大touch长度: 49
# 最大sensor长度: 122
# 最小touch长度: 13
# 最小sensor长度: 58
# 最大touch长度: 53
# 最大sensor长度: 138
# 最小touch长度: 14
# 最小sensor长度: 46
# 定义文件路径
file_paths = {
    "normalized_touch_data_sit.csv": "TDatas/normalized_filtered_touch_data_sit.csv",
    "normalized_sensor_data_sit.csv": "TDatas/normalized_filtered_sensor_data_sit.csv",
    "normalized_touch_data_walk.csv": "TDatas/normalized_filtered_touch_data_walk.csv",
    "normalized_sensor_data_walk.csv": "TDatas/normalized_filtered_sensor_data_walk.csv"
}

# 定义目标长度
target_lengths = {
    "normalized_touch_data_sit.csv": 90,
    "normalized_sensor_data_sit.csv": 160,
    "normalized_touch_data_walk.csv": 90,
    "normalized_sensor_data_walk.csv": 160
}

# 定义需要扩充的列
touch_columns_to_pad = ['ACTION_TYPE', 'Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']
sensor_columns_to_pad = ['Time', 'SensorType', 'X', 'Y', 'Z']


# 定义处理函数
def process_files(touch_file, sensor_file, target_touch_length, target_sensor_length):
    # 读取文件
    touch_df = pd.read_csv(touch_file)
    sensor_df = pd.read_csv(sensor_file)

    common_sample_ids = touch_df['Sample ID'][touch_df['Sample ID'].isin(sensor_df['Sample ID'])].unique()

    # 筛选出交集中的样本
    touch_df = touch_df[touch_df['Sample ID'].isin(common_sample_ids)]
    sensor_df = sensor_df[sensor_df['Sample ID'].isin(common_sample_ids)]

    touch_lengths = touch_df[touch_df['Sample ID'].isin(common_sample_ids)].groupby('Sample ID').size()
    sensor_lengths = sensor_df[sensor_df['Sample ID'].isin(common_sample_ids)].groupby('Sample ID').size()
    
    # # 找出最大长度
    # max_touch_length = touch_lengths.max()
    # max_sensor_length = sensor_lengths.max()
    
    # # 输出最大长度
    # print(f"最大touch长度: {max_touch_length}")
    # print(f"最大sensor长度: {max_sensor_length}")
    
    # # 找出最小长度
    # min_touch_length = touch_lengths.min()
    # min_sensor_length = sensor_lengths.min()
    
    # # 输出最大长度
    # print(f"最小touch长度: {min_touch_length}")
    # print(f"最小sensor长度: {min_sensor_length}")


    # 初始化结果列表
    touch_result_data = []
    sensor_result_data = []

    # 遍历每个样本
    for sample_id in common_sample_ids:
        touch_group = touch_df[touch_df['Sample ID'] == sample_id]
        sensor_group = sensor_df[sensor_df['Sample ID'] == sample_id]

        # 扩充touch数据
        touch_sample_length = len(touch_group)
        if touch_sample_length < target_touch_length:
            touch_pad_data = {col: [0] * (target_touch_length - touch_sample_length) for col in touch_columns_to_pad}
            touch_pad_data['Sample ID'] = [sample_id] * (target_touch_length - touch_sample_length)
            touch_pad_data['PIN'] = [touch_group['PIN'].iloc[0]] * (target_touch_length - touch_sample_length)
            touch_pad_data['UUID'] = [touch_group['UUID'].iloc[0]] * (target_touch_length - touch_sample_length)
            touch_pad_data['is_original'] = [0] * (target_touch_length - touch_sample_length)
            touch_pad_data['Posture'] = [touch_group['Posture'].iloc[0]] * (target_touch_length - touch_sample_length)
            touch_pad_data['TimePeriod'] = [touch_group['TimePeriod'].iloc[0]] * (target_touch_length - touch_sample_length)
            touch_pad_df = pd.DataFrame(touch_pad_data)
            touch_group = pd.concat([touch_group.assign(is_original=1), touch_pad_df], ignore_index=True)

        # 扩充sensor数据
        sensor_sample_length = len(sensor_group)
        if sensor_sample_length < target_sensor_length:
            sensor_pad_data = {col: [0] * (target_sensor_length - sensor_sample_length) for col in
                               sensor_columns_to_pad}
            sensor_pad_data['Sample ID'] = [sample_id] * (target_sensor_length - sensor_sample_length)
            sensor_pad_data['PIN'] = [sensor_group['PIN'].iloc[0]] * (target_sensor_length - sensor_sample_length)
            sensor_pad_data['UUID'] = [sensor_group['UUID'].iloc[0]] * (target_sensor_length - sensor_sample_length)
            sensor_pad_data['is_original'] = [0] * (target_sensor_length - sensor_sample_length)
            sensor_pad_data['posture'] = [sensor_group['posture'].iloc[0]] * (target_sensor_length - sensor_sample_length)
            sensor_pad_data['TimePeriod'] = [sensor_group['TimePeriod'].iloc[0]] * (target_sensor_length - sensor_sample_length)
            sensor_pad_df = pd.DataFrame(sensor_pad_data)
            sensor_group = pd.concat([sensor_group.assign(is_original=1), sensor_pad_df], ignore_index=True)

        # 保存扩充后的样本
        touch_result_data.append(touch_group)
        sensor_result_data.append(sensor_group)

    # 合并所有样本
    touch_result_df = pd.concat(touch_result_data, ignore_index=True)
    sensor_result_df = pd.concat(sensor_result_data, ignore_index=True)

    # 保存到新的文件
    touch_output_file = f"TDatas/masked_{touch_file.split('/')[-1]}"
    sensor_output_file = f"TDatas/masked_{sensor_file.split('/')[-1]}"
    touch_result_df.to_csv(touch_output_file, index=False)
    sensor_result_df.to_csv(sensor_output_file, index=False)
    print(f"处理完成，结果已保存到 {touch_output_file} 和 {sensor_output_file}")


# 处理sit和walk数据
process_files(file_paths["normalized_touch_data_sit.csv"], file_paths["normalized_sensor_data_sit.csv"],
              target_lengths["normalized_touch_data_sit.csv"], target_lengths["normalized_sensor_data_sit.csv"])

process_files(file_paths["normalized_touch_data_walk.csv"], file_paths["normalized_sensor_data_walk.csv"],
              target_lengths["normalized_touch_data_walk.csv"], target_lengths["normalized_sensor_data_walk.csv"])