import pandas as pd
import numpy as np
import os

# 设置工作目录
os.chdir('datas')

# 定义文件名
touch_files = ["filtered_touch_data_sit.csv", "filtered_touch_data_walk.csv"]
sensor_files = ["filtered_sensor_data_sit.csv", "filtered_sensor_data_walk.csv"]

# 处理Touch数据
for file in touch_files:
    print(f"Processing {file}...")
    df = pd.read_csv(file)

    # 先对每个Sample ID内的数据按时间排序
    df = df.sort_values(by=['Sample ID', 'Time']).reset_index(drop=True)

    df['Time'] = df.groupby('Sample ID')['Time'].transform(lambda x: x - x.iloc[0])

    if df['Time'].nunique() > 1:
        min_t = df['Time'].min()
        max_t = df['Time'].max()
        df['Time'] = (df['Time'] - min_t) / (max_t - min_t)
    else:
        df['Time'] = 0.0  # 全相同则设为0

    # 全局归一化处理指定列
    norm_cols = ['X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']
    for col in norm_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0

    # 保存处理后的文件
    output_file = f"normalized_{file}"
    df.to_csv(output_file, index=False)
    print(f"Processed file saved as {output_file}")

# 处理Sensor数据
for file in sensor_files:
    print(f"Processing {file}...")
    df = pd.read_csv(file)

    # 先对每个Sample ID内的数据按时间排序
    df = df.sort_values(by=['Sample ID', 'Time']).reset_index(drop=True)

    df['Time'] = df.groupby('Sample ID')['Time'].transform(lambda x: x - x.iloc[0])

    # Step 2: 对整个文件的Time列进行归一化（不再分组）
    if df['Time'].nunique() > 1:
        min_t = df['Time'].min()
        max_t = df['Time'].max()
        df['Time'] = (df['Time'] - min_t) / (max_t - min_t)
    else:
        df['Time'] = 0.0  # 全相同则设为0

    # 全局归一化处理指定列
    sensor_order = ['Gravity', 'Accelerometer', 'Gyroscope']
    for sensor_type in sensor_order:
        sensor_df = df[df['SensorType'] == sensor_type]
        if not sensor_df.empty:
            for col in ['X', 'Y', 'Z']:
                min_val = sensor_df[col].min()
                max_val = sensor_df[col].max()
                if max_val != min_val:
                    df.loc[sensor_df.index, col] = (sensor_df[col] - min_val) / (max_val - min_val)
                else:
                    df.loc[sensor_df.index, col] = 0.0  # 全相同则设为0


    def sort_by_sensor_type(group):
        # 对每种SensorType按时间排序
        sorted_groups = []
        for sensor in sensor_order:
            sensor_group = group[group['SensorType'] == sensor]
            sensor_group = sensor_group.sort_values('Time')
            sorted_groups.append(sensor_group)
        return pd.concat(sorted_groups)


    df = df.groupby('Sample ID').apply(sort_by_sensor_type).reset_index(drop=True)

    # 保存处理后的文件
    output_file = f"normalized_{file}"
    df.to_csv(output_file, index=False)
    print(f"Processed file saved as {output_file}")