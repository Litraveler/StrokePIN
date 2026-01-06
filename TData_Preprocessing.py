import pandas as pd
import os
import glob
import uuid

# 配置路径（修改为新数据集路径）
data_dir = r"E:\代码库 数据库\混合数据集\4个时间段数据集"
os.makedirs(data_dir, exist_ok=True)

postures = ["sit", "walk"]

def find_file_name(user_path, pattern):
    """查找匹配的文件路径"""
    file_pattern = os.path.join(user_path, pattern)
    matching_files = glob.glob(file_pattern)
    
    # 如果没有找到匹配的文件，尝试使用用户文件夹名称作为前缀进行匹配
    if not matching_files:
        user_folder = os.path.basename(user_path)
        alternative_pattern = os.path.join(user_path, f"{user_folder}_*{pattern.strip('*')}")
        matching_files = glob.glob(alternative_pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"未找到匹配的文件: {file_pattern}")
    
    return matching_files[0]

# 处理每个用户：获取顶级用户文件夹（19个用户）
user_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, f))]

# 输出路径设置
save_file_path = "TDatas"
os.makedirs(save_file_path, exist_ok=True)  # 确保输出目录存在

# 为不同姿势定义不同的输出文件路径
output_sensor_file_names = {
    "sit": "sensor_data_sit.csv",
    "walk": "sensor_data_walk.csv"
}
output_touch_file_names = {
    "sit": "touch_data_sit.csv",
    "walk": "touch_data_walk.csv"
}

output_sensor_paths = {
    "sit": os.path.join(save_file_path, output_sensor_file_names["sit"]),
    "walk": os.path.join(save_file_path, output_sensor_file_names["walk"])
}
output_touch_paths = {
    "sit": os.path.join(save_file_path, output_touch_file_names["sit"]),
    "walk": os.path.join(save_file_path, output_touch_file_names["walk"])
}

# 为不同姿势分别初始化DataFrame
filtered_sensor_data = {
    "sit": pd.DataFrame(),
    "walk": pd.DataFrame()
}
filtered_touch_data = {
    "sit": pd.DataFrame(),
    "walk": pd.DataFrame()
}

# 为每个用户生成唯一UUID（一个用户对应一个UUID）
user_id_map = {}
for user_folder in user_folders:
    while True:
        user_id = str(uuid.uuid4())
        if user_id not in user_id_map.values():
            user_id_map[user_folder] = user_id
            break

for user_folder in user_folders:
    user_id = user_id_map[user_folder]
    print(f"正在处理用户: {user_folder}，UUID: {user_id}")
    
    # 获取用户文件夹下的所有时间段文件夹，并按时间排序（假设文件夹名按日期从小到大排序）
    time_folders = [f for f in os.listdir(user_folder) 
                   if os.path.isdir(os.path.join(user_folder, f))]
    # 按文件夹名称排序（确保时间从早到晚）
    time_folders_sorted = sorted(time_folders)
    print(time_folders_sorted)
    
    # 遍历每个时间段文件夹（1-4）
    for time_period, time_folder in enumerate(time_folders_sorted, start=1):
        time_folder_path = os.path.join(user_folder, time_folder)
        print(f"  处理时间段 {time_period}：{time_folder_path}")
        
        # 检查该时间段文件夹下是否有CSV文件
        if not any(file.endswith('.csv') for file in os.listdir(time_folder_path)):
            print(f"  时间段 {time_period} 无CSV文件，跳过")
            continue
        
        for posture in postures:
            try:
                # 读取PIN输入记录数据（路径改为时间段文件夹）
                try:
                    record_file_path = find_file_name(time_folder_path, f'*_PINKeystroke_{posture}_editText_record*.csv')
                    print(f"  找到PIN输入记录文件: {record_file_path}")

                    # 确保CharSequence列被读取为字符串类型
                    pin_data = pd.read_csv(record_file_path, dtype={'CharSequence': str})

                except FileNotFoundError:
                    print(f"  用户 {user_folder} 时间段 {time_period} 没有PIN输入记录数据，跳过处理")
                    continue
                except Exception as e:
                    print(f"  读取用户 {user_folder} 时间段 {time_period} 的PIN输入记录数据时出错: {e}")
                    continue

                # 定义要收集的PIN码列表，共20个PIN码
                valid_pins_str = [
                    "194012", "201412", "400101", "141231", "194011", 
                    "201499", "111111", "147258", "123456", "585520", 
                    "121940", "122914", "010140", "123114", "111940", 
                    "992914", "121212","112233", "136136", "111222"
                ]
                
                # 找出完整正确输入PIN码的时间段（6次输入+1个空格）
                complete_pin_entries = []
                current_sequence = []
                current_start_time = None

                # 用于跟踪每个PIN码出现的次数
                pin_count = {}
                for i in range(len(pin_data)):
                    row = pin_data.iloc[i]

                    # 记录当前输入，确保存储原始格式
                    if not pd.isna(row['CharSequence']) and row['CharSequence'] != '':
                        if current_start_time is None:
                            current_start_time = row['Time']
                        # 直接存储原始值，不进行字符串转换
                        current_sequence.append(str(row['CharSequence']).replace('.0', ''))
                    else:
                        # 如果是空字符串，表示一次输入结束
                        # 检查是否是完整的6位PIN码输入，并且是我们要收集的PIN码
                        if len(current_sequence) == 6:
                            # 获取最后一个值
                            last_pin = current_sequence[-1]

                            # 确保PIN码格式正确（去除可能的小数点和零）
                            last_pin = str(last_pin).replace('.0', '')

                            # 检查是否在有效PIN码列表中
                            if last_pin in valid_pins_str:
                                # 检查是否已经收集了7次该PIN码
                                if last_pin not in pin_count:
                                    pin_count[last_pin] = 0
                                # 记录这次完整正确输入的时间范围
                                end_time = row['Time']  # 当前行的时间（输入结束）
                                if current_start_time is not None:
                                    pin_count[last_pin] += 1
                                    if pin_count[last_pin] != 1 and pin_count[last_pin] != 2 and pin_count[last_pin] < 8:
                                        complete_pin_entries.append((current_start_time, end_time, last_pin))

                        # 重置当前序列
                        current_sequence = []
                        current_start_time = None

                # 读取传感器数据和触摸数据（路径改为时间段文件夹）
                try:
                    sensor_file_path = find_file_name(time_folder_path, f'*PINKeystroke_{posture}_sensorData*.csv')
                    touch_file_path = find_file_name(time_folder_path, f'*PINKeystroke_{posture}_touchData*.csv')
                    print(f"  传感器文件: {sensor_file_path}")
                    print(f"  触摸文件: {touch_file_path}")

                    sensor_data = pd.read_csv(sensor_file_path)
                    touch_data = pd.read_csv(touch_file_path)

                    # 统计每次PIN码输入时间段内的触摸数据次数（一个Down+Move+Up算一次触摸）
                    touch_counts = []
                    complete_touch_data_periods = []  # 存储完整的触摸数据时间段

                    for idx, (start_time, end_time, pin_value) in enumerate(complete_pin_entries):
                        # 选择时间范围内的触摸数据
                        mask = (touch_data['Time'] >= start_time) & (touch_data['Time'] <= end_time)
                        period_touch_data = touch_data[mask].copy()

                        # 计算Down事件的数量，即触摸次数
                        down_events = period_touch_data[period_touch_data['ACTION_TYPE'].str.startswith('Down')]
                        up_events = period_touch_data[period_touch_data['ACTION_TYPE'].str.startswith('Up')]

                        new_start_time = down_events['Time'].min() if not down_events.empty else start_time
                        new_end_time = up_events['Time'].max() if not up_events.empty else end_time

                        if not up_events.empty and not down_events.empty and up_events['Time'].min() < down_events['Time'].min():
                            # 找到前一个down
                            down_mask = touch_data['ACTION_TYPE'].str.startswith('Down')
                            all_down_events = touch_data[down_mask]
                            before_down_events = all_down_events[all_down_events['Time'] < up_events['Time'].min()]
                            if not before_down_events.empty:
                                before_touch = before_down_events.iloc[-1]  # 取最后一条（最接近的）
                                new_start_time = before_touch['Time']
                                down_events = pd.concat([before_touch, down_events], ignore_index=True)
                        if not up_events.empty and not down_events.empty and up_events['Time'].max() < down_events['Time'].max():
                            # 找到后一个up
                            after_mask = touch_data['ACTION_TYPE'].str.startswith('Up')
                            all_up_events = touch_data[after_mask]
                            after_up_events = all_up_events[all_up_events['Time'] > down_events['Time'].max()]
                            if not after_up_events.empty:
                                after_touch = after_up_events.iloc[0]  # 取第一条（最接近的）
                                new_end_time = after_touch['Time']
                                up_events = pd.concat([after_touch, up_events], ignore_index=True)

                        touch_count = len(down_events)
                        # 如果触摸次数只有4次或5次，需要补充触摸
                        if touch_count < 6:
                            needed_touches = 6 - touch_count  # 需要补充的触摸次数
                            # 查找包含Down事件的触摸数据
                            down_mask = touch_data['ACTION_TYPE'].str.startswith('Down')
                            all_down_events = touch_data[down_mask]

                            # 补充需要的触摸次数
                            for _ in range(needed_touches):
                                # 找到当前时间段之前的Down事件
                                before_down_events = all_down_events[all_down_events['Time'] < new_start_time]
                                if not before_down_events.empty:
                                    before_touch = before_down_events.iloc[-1]  # 取最后一条（最接近的）
                                    before_time_diff = new_start_time - before_touch['Time']
                                    before_time = before_touch['Time']
                                else:
                                    before_time_diff = float('inf')
                                    before_time = None

                                # 找到当前时间段之后的Down事件
                                after_down_events = all_down_events[all_down_events['Time'] > new_end_time]
                                if not after_down_events.empty:
                                    after_touch = after_down_events.iloc[0]  # 取第一条（最接近的）
                                    after_time_diff = after_touch['Time'] - new_end_time
                                    after_time = after_touch['Time']
                                else:
                                    after_time_diff = float('inf')
                                    after_time = None

                                # 选择时间差较小的那个
                                if before_time_diff <= after_time_diff and before_time_diff != float('inf'):
                                    # 使用之前的Down事件
                                    # 找到这个Down事件相关的所有事件（包括后续的Move和Up）
                                    extra_start_time = before_time
                                    # 找到这个Down事件对应的Up事件
                                    up_events_after_down = touch_data[(touch_data['Time'] >= extra_start_time) &
                                                                      (touch_data['Time'] < new_start_time) &
                                                                      (touch_data['ACTION_TYPE'].str.startswith('Up'))]
                                    if not up_events_after_down.empty:
                                        extra_end_time = up_events_after_down.iloc[0]['Time']
                                    else:
                                        extra_end_time = new_start_time

                                    new_start_time = extra_start_time
                                elif after_time_diff != float('inf'):
                                    # 使用之后的Down事件
                                    extra_start_time = after_time
                                    # 找到这个Down事件对应的Up事件
                                    up_events_after_down = touch_data[(touch_data['Time'] >= extra_start_time) &
                                                                      (touch_data['ACTION_TYPE'].str.startswith('Up'))]
                                    if not up_events_after_down.empty:
                                        extra_end_time = up_events_after_down.iloc[0]['Time']
                                    else:
                                        extra_end_time = new_end_time

                                    new_end_time = extra_end_time
                                else:
                                    # 没有找到合适的触摸数据
                                    print(f"  未找到合适的Down事件进行补充")
                                    break  # 如果找不到合适的触摸数据，就退出循环
                                # 重新选择时间范围内的触摸数据
                                mask = (touch_data['Time'] >= new_start_time) & (touch_data['Time'] <= new_end_time)
                                period_touch_data = touch_data[mask].copy()

                                # 重新计算触摸次数
                                down_events = period_touch_data[period_touch_data['ACTION_TYPE'].str.startswith('Down')]
                                touch_count = len(down_events)
                            # 更新时间段
                            if touch_count == 6:
                                complete_touch_data_periods.append((new_start_time, new_end_time, pin_value, user_id))

                        elif touch_count == 6:
                            complete_touch_data_periods.append((new_start_time, new_end_time, pin_value, user_id))
                            touch_counts.append(touch_count)

                    for idx, (start_time, end_time, pin_value, user_id) in enumerate(complete_touch_data_periods):
                        valid_sensor_types = ['Gravity', 'Gyroscope', 'Accelerometer']
                        sample_id = str(uuid.uuid4())
                        # 传感器数据时间范围扩展
                        mask = (sensor_data['Time'] >= start_time - 250000000) & (
                                    sensor_data['Time'] <= end_time + 250000000)
                        period_sensor_data = sensor_data[mask].copy()
                        period_sensor_data["posture"] = posture
                        period_sensor_data['PIN'] = pin_value
                        period_sensor_data['Sample ID'] = sample_id
                        period_sensor_data['UUID'] = user_id
                        period_sensor_data['TimePeriod'] = time_period
                        # 筛选有效传感器类型
                        period_sensor_data = period_sensor_data[period_sensor_data['SensorType'].isin(valid_sensor_types)].copy()

                        # 触摸数据处理
                        mask = (touch_data['Time'] >= start_time) & (touch_data['Time'] <= end_time)
                        period_touch_data = touch_data[mask].copy()
                        period_touch_data['Posture'] = posture
                        period_touch_data['PIN'] = pin_value
                        period_touch_data['Sample ID'] = sample_id
                        period_touch_data['UUID'] = user_id
                        # 添加时间段维度（1-4）
                        period_touch_data['TimePeriod'] = time_period

                        # 按姿势合并到对应的数据框
                        filtered_touch_data[posture] = pd.concat([filtered_touch_data[posture], period_touch_data])
                        filtered_sensor_data[posture] = pd.concat([filtered_sensor_data[posture], period_sensor_data])
                except Exception as e:
                    print(f"  读取用户 {user_folder} 时间段 {time_period} 的传感器或触摸数据时出错: {e}")
                    continue
            except Exception as e:
                print(f"  处理用户 {user_folder} 时间段 {time_period} 时出现未预期的错误: {e}")
                continue

# 按姿势分别保存处理后的数据
for posture in postures:
    filtered_sensor_data[posture].to_csv(output_sensor_paths[posture], index=False)
    filtered_touch_data[posture].to_csv(output_touch_paths[posture], index=False)

print("所有用户处理完成！")



# ****************测试*******************
import pandas as pd
import os

# 1. 配置文件路径（与原代码保持一致）
save_file_path = "TDatas"
postures = ["sit", "walk"]

# 2. 定义输出文件名称（与原代码输出文件名对应）
output_sensor_file_names = {
    "sit": "sensor_data_sit.csv",
    "walk": "sensor_data_walk.csv"
}
output_touch_file_names = {
    "sit": "touch_data_sit.csv",
    "walk": "touch_data_walk.csv"
}

# 3. 封装样本长度测试函数
def test_sample_length(file_path, file_type, posture):
    """
    测试单个文件中每个Sample ID的样本长度
    :param file_path: 文件完整路径
    :param file_type: 文件类型（sensor/touch）
    :param posture: 姿势（sit/walk）
    """
    # 读取文件
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ 未找到文件：{file_path}")
        return
    except Exception as e:
        print(f"❌ 读取文件 {file_path} 失败：{str(e)}")
        return
    
    # 检查是否包含Sample ID列
    if "Sample ID" not in df.columns:
        print(f"⚠️ 文件 {file_path} 中不存在'Sample ID'列，无法统计样本长度")
        return
    
    # 按Sample ID分组，统计每个样本的行数（即样本长度）
    sample_lengths = df.groupby("Sample ID").size().reset_index(name="Sample_Length")
    
    # 计算统计信息
    total_samples = len(sample_lengths)
    if total_samples == 0:
        print(f"⚠️ 文件 {file_path} 中无有效样本数据")
        return
    
    min_length = sample_lengths["Sample_Length"].min()
    max_length = sample_lengths["Sample_Length"].max()
    mean_length = sample_lengths["Sample_Length"].mean().round(2)
    median_length = sample_lengths["Sample_Length"].median()
    
    # 打印结果
    print("=" * 60)
    print(f"📊 【{posture.upper()} - {file_type.upper()} 文件】样本长度统计")
    print(f"文件路径：{file_path}")
    print(f"总样本数：{total_samples}")
    print(f"样本长度范围：{min_length} ~ {max_length}")
    print(f"样本平均长度：{mean_length}")
    print(f"样本中位数长度：{median_length}")
    print("\n前10个样本的长度详情：")
    print(sample_lengths.head(10).to_string(index=False))
    print("=" * 60 + "\n")

# 4. 遍历所有文件，执行样本长度测试
if __name__ == "__main__":
    print("开始测试所有输出文件的样本长度...\n")
    
    # 确保输出目录存在
    if not os.path.exists(save_file_path):
        print(f"❌ 输出目录 {save_file_path} 不存在")
        exit(1)
    
    # 测试传感器文件
    for posture in postures:
        sensor_file_name = output_sensor_file_names[posture]
        sensor_file_path = os.path.join(save_file_path, sensor_file_name)
        test_sample_length(sensor_file_path, "sensor", posture)
    
    # 测试触摸文件
    for posture in postures:
        touch_file_name = output_touch_file_names[posture]
        touch_file_path = os.path.join(save_file_path, touch_file_name)
        test_sample_length(touch_file_path, "touch", posture)
    
    print("✅ 所有文件样本长度测试完成！")

# ============================================================
# 📊 【SIT - SENSOR 文件】样本长度统计
# 文件路径：TDatas\sensor_data_sit.csv
# 总样本数：7110
# 样本长度范围：58 ~ 554
# 样本平均长度：100.82
# 样本中位数长度：96.0

# 前10个样本的长度详情：
#                            Sample ID  Sample_Length
# 00059d6e-97a8-42dc-a5ba-a4f8228086c8            105
# 003244ba-4093-4daa-a49c-b3f4309aef19             94
# 003faecf-4c11-4f23-bbba-4e00edbe7336             95
# 004274fe-80a1-4caa-a67a-c527f61456a3             82
# 00546cc5-17b4-425b-84bc-38d01f6ff9fc             82
# 0067aa07-80fa-4e1f-b581-bb41f83e54f4             83
# 007de87e-1baf-44df-93b4-1fe0a7b5855b            139
# 008094dd-28cd-4a68-85b1-73bfd5d8816f             97
# 00862cfc-7675-44b1-bfe1-d85b450ebf12             94
# 009763fa-d8a9-440e-8348-278e71b81ed3            106
# ============================================================

# ============================================================
# 📊 【WALK - SENSOR 文件】样本长度统计
# 文件路径：TDatas\sensor_data_walk.csv
# 总样本数：6951
# 样本长度范围：46 ~ 618
# 样本平均长度：115.48
# 样本中位数长度：96.0

# 前10个样本的长度详情：
#                            Sample ID  Sample_Length
# 000ab98f-e955-4646-bd2c-2607854a2418            114
# 001990dd-a699-41e6-bbff-19e5cccc65a6             72
# 001e2208-f11e-4f9c-9f76-9bf183df14b9             93
# 0028f0a3-47f6-42ee-b7c4-ccb7fdbc4cc5             74
# 0029a670-1a5a-4a63-9576-8fa97c8e875c            116
# 00395514-7e16-4020-9051-987d0b14c529             96
# 003f3e88-a779-4e33-9b83-4d0223369801             77
# 00458934-8828-4001-9add-226faa012b0e            101
# 0047d1e2-b43a-4cdc-952f-dc7192f5cacd             81
# 0049e411-e2ed-450e-a0c8-12e6469b1b57             91
# ============================================================

# ============================================================
# 📊 【SIT - TOUCH 文件】样本长度统计
# 文件路径：TDatas\touch_data_sit.csv
# 总样本数：7110
# 样本长度范围：13 ~ 53
# 样本平均长度：24.28
# 样本中位数长度：24.0

# 前10个样本的长度详情：
#                            Sample ID  Sample_Length
# 00059d6e-97a8-42dc-a5ba-a4f8228086c8             25
# 003244ba-4093-4daa-a49c-b3f4309aef19             23
# 003faecf-4c11-4f23-bbba-4e00edbe7336             22
# 004274fe-80a1-4caa-a67a-c527f61456a3             22
# 00546cc5-17b4-425b-84bc-38d01f6ff9fc             22
# 0067aa07-80fa-4e1f-b581-bb41f83e54f4             24
# 007de87e-1baf-44df-93b4-1fe0a7b5855b             23
# 008094dd-28cd-4a68-85b1-73bfd5d8816f             22
# 00862cfc-7675-44b1-bfe1-d85b450ebf12             29
# 009763fa-d8a9-440e-8348-278e71b81ed3             21
# ============================================================

# ============================================================
# 📊 【WALK - TOUCH 文件】样本长度统计
# 文件路径：TDatas\touch_data_walk.csv
# 总样本数：6951
# 样本长度范围：14 ~ 53
# 样本平均长度：24.21
# 样本中位数长度：24.0

# 前10个样本的长度详情：
#                            Sample ID  Sample_Length
# 000ab98f-e955-4646-bd2c-2607854a2418             29
# 001990dd-a699-41e6-bbff-19e5cccc65a6             27
# 001e2208-f11e-4f9c-9f76-9bf183df14b9             23
# 0028f0a3-47f6-42ee-b7c4-ccb7fdbc4cc5             20
# 0029a670-1a5a-4a63-9576-8fa97c8e875c             27
# 00395514-7e16-4020-9051-987d0b14c529             24
# 003f3e88-a779-4e33-9b83-4d0223369801             29
# 00458934-8828-4001-9add-226faa012b0e             28
# 0047d1e2-b43a-4cdc-952f-dc7192f5cacd             20
# 0049e411-e2ed-450e-a0c8-12e6469b1b57             24
# ============================================================