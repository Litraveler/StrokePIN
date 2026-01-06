import pandas as pd
import os
import numpy as np
# 定义文件路径
data_folder = "datas"
sensor_files = ["sensor_data_sit.csv", "sensor_data_walk.csv"]
touch_files = ["touch_data_sit.csv", "touch_data_walk.csv"]

# 定义时间间隔阈值（单位：纳秒）
time_threshold = 3 * 1e9  # 3秒 = 3 * 1e9纳秒

def process_files(sensor_file, touch_file):
    # 读取数据
    sensor_data = pd.read_csv(os.path.join(data_folder, sensor_file))
    touch_data = pd.read_csv(os.path.join(data_folder, touch_file))

    # =============== 记录原始样本数 ===============
    original_ids = sensor_data['Sample ID'].unique()
    n_original = len(original_ids)

    touch_data['Time'] = touch_data['Time'].astype(np.int64)
    sample_time_diff = touch_data.groupby('Sample ID')['Time'].agg(lambda x: x.max() - x.min())
    invalid_sample_ids = sample_time_diff[sample_time_diff > time_threshold].index

    sensor_data = sensor_data[~sensor_data['Sample ID'].isin(invalid_sample_ids)]
    touch_data = touch_data[~touch_data['Sample ID'].isin(invalid_sample_ids)]

    after_time_filter_ids = sensor_data['Sample ID'].unique()
    n_after_time = len(after_time_filter_ids)

    sample_lengths = sensor_data.groupby('Sample ID').size().reset_index(name='Length')
    sample_lengths = sample_lengths.sort_values(by='Length')

    num_samples = len(sample_lengths)
    num_to_remove = max(1, int(num_samples * 0.1))
    last_10_percent_sample_ids = sample_lengths.iloc[-num_to_remove:]['Sample ID']

    sensor_data = sensor_data[~sensor_data['Sample ID'].isin(last_10_percent_sample_ids)]
    touch_data = touch_data[~touch_data['Sample ID'].isin(last_10_percent_sample_ids)]

    after_len_filter_ids = sensor_data['Sample ID'].unique()
    n_final = len(after_len_filter_ids)

    # =============== 保存处理后的数据 ===============
    # 构造输出文件名（添加filtered_前缀）
    filtered_sensor_file = f"filtered_{sensor_file}"
    filtered_touch_file = f"filtered_{touch_file}"
    # 保存文件
    sensor_data.to_csv(os.path.join(data_folder, filtered_sensor_file), index=False)
    touch_data.to_csv(os.path.join(data_folder, filtered_touch_file), index=False)

    # =============== 打印百分比 ===============
    pct_time = (n_original - n_after_time) / n_original * 100
    pct_len  = (n_after_time - n_final) / n_original * 100
    pct_total = (n_original - n_final) / n_original * 100

    print(f"{sensor_file} & {touch_file}  处理结果:")
    print(f"  - 因时间间隔过长去除: {pct_time:.2f}% ({n_original - n_after_time}/{n_original})")
    print(f"  - 因长度后10%去除   : {pct_len:.2f}% ({n_after_time - n_final}/{n_original})")
    print(f"  - 总共去除比例      : {pct_total:.2f}% ({n_original - n_final}/{n_original})")
    print(f"  - 已保存处理后文件: {filtered_sensor_file}, {filtered_touch_file}")
    print("-" * 50)

# 处理每组文件
for sensor_file, touch_file in zip(sensor_files, touch_files):
    process_files(sensor_file, touch_file)

print("处理完成！")

# (swipeformerHuMI) E:\论文撰写记录\投稿\StrokePIN\Experiments\Keystroke Dynamic based Authentication>python Data_Cleaning.py
# E:\论文撰写记录\投稿\StrokePIN\Experiments\Keystroke Dynamic based Authentication\Data_Cleaning.py:22: RuntimeWarning: overflow encountered in scalar subtract
#   sample_time_diff = touch_data.groupby('Sample ID')['Time'].agg(lambda x: x.max() - x.min())
# sensor_data_sit.csv & touch_data_sit.csv  处理结果:
#   - 因时间间隔过长去除: 0.00% (0/7219)
#   - 因长度后10%去除   : 9.99% (721/7219)
#   - 总共去除比例      : 9.99% (721/7219)
#   - 已保存处理后文件: filtered_sensor_data_sit.csv, filtered_touch_data_sit.csv
# --------------------------------------------------
# E:\论文撰写记录\投稿\StrokePIN\Experiments\Keystroke Dynamic based Authentication\Data_Cleaning.py:22: RuntimeWarning: overflow encountered in scalar subtract
#   sample_time_diff = touch_data.groupby('Sample ID')['Time'].agg(lambda x: x.max() - x.min())
# sensor_data_walk.csv & touch_data_walk.csv  处理结果:
#   - 因时间间隔过长去除: 0.00% (0/7045)
#   - 因长度后10%去除   : 9.99% (704/7045)
#   - 总共去除比例      : 9.99% (704/7045)
#   - 已保存处理后文件: filtered_sensor_data_walk.csv, filtered_touch_data_walk.csv