import pandas as pd
import os
import glob
import uuid

# 配置路径
data_dir = '../Data'
os.makedirs(data_dir, exist_ok=True)

postures = ["sit","walk"]

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

# 处理每个用户
user_folders = []
# 遍历主目录及其所有子目录
for root, dirs, files in os.walk(data_dir):
    # 检查当前目录中的文件是否有 .csv 文件
    if any(file.endswith('.csv') for file in files):
        # 如果有 .csv 文件，将当前目录路径添加到列表中
        user_folders.append(root)
save_file_path = '../Data'
output_sensor_file_name = f"sensor_data.csv"
output_touch_file_name = f"touch_data.csv"
output_sensor_path = os.path.join(save_file_path, output_sensor_file_name)
output_touch_path = os.path.join(save_file_path, output_touch_file_name)
filtered_sensor_data = pd.DataFrame()
filtered_touch_data = pd.DataFrame()
save_file_path = "datas"


user_ids = []
for user_folder in user_folders:
    judge = False
    while judge is False:
        user_id = str(uuid.uuid4())
        if user_id not in user_ids:
            judge = True
            user_ids.append(user_id)

    print(f"正在处理用户: {user_folder}")
    for posture in postures:
        try:
            # 读取PIN输入记录数据
            try:
                # 使用更通用的匹配模式
                record_file_path = find_file_name(user_folder, f'*_PINKeystroke_{posture}_editText_record*.csv')
                print(f"找到PIN输入记录文件: {record_file_path}")

                # 确保CharSequence列被读取为字符串类型
                pin_data = pd.read_csv(record_file_path, dtype={'CharSequence': str})

            except FileNotFoundError:
                print(f"用户 {user_folder} 没有PIN输入记录数据，跳过处理")
                continue
            except Exception as e:
                print(f"读取用户 {user_folder} 的PIN输入记录数据时出错: {e}")
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
            # all_pins = set()
            # temp_sequence = []
            # for i in range(len(pin_data)):
            #     row = pin_data.iloc[i]
            #     if pd.isna(row['CharSequence']) or row['CharSequence'] == '':
            #         if len(temp_sequence) == 6:
            #             all_pins.add(str(temp_sequence[-1]).replace('.0', ''))
            #         temp_sequence = []
            #     else:
            #         temp_sequence.append(row['CharSequence'])
            # # 读取传感器数据和触摸数据
            try:
                sensor_file_path = find_file_name(user_folder, f'*PINKeystroke_{posture}_sensorData*.csv')
                touch_file_path = find_file_name(user_folder, f'*PINKeystroke_{posture}_touchData*.csv')
                print(sensor_file_path)
                print(touch_file_path)

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

                    new_start_time = down_events['Time'].min()
                    new_end_time = up_events['Time'].max()

                    if up_events['Time'].min() < down_events['Time'].min():
                        # 找到前一个down
                        down_mask = touch_data['ACTION_TYPE'].str.startswith('Down')
                        all_down_events = touch_data[down_mask]
                        before_down_events = all_down_events[all_down_events['Time'] < up_events['Time'].min()]
                        before_touch = before_down_events.iloc[-1]  # 取最后一条（最接近的）
                        new_start_time = before_touch['Time']
                        down_events = pd.concat([before_touch, down_events], ignore_index=True)
                    if up_events['Time'].max() < down_events['Time'].max():
                        # 找到后一个up
                        after_mask = touch_data['ACTION_TYPE'].str.startswith('Up')
                        all_up_events = touch_data[after_mask]
                        after_up_events = all_up_events[all_up_events['Time'] > down_events['Time'].max()]
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
                    mask = (sensor_data['Time'] >= start_time - 250000000) & (
                                sensor_data['Time'] <= end_time + 250000000)
                    period_sensor_data = sensor_data[mask].copy()
                    period_sensor_data["posture"] = posture
                    period_sensor_data['PIN'] = pin_value
                    period_sensor_data['Sample ID'] = sample_id
                    period_sensor_data['UUID'] = user_id
                    period_sensor_data[period_sensor_data['SensorType'].isin(valid_sensor_types)].copy()
                    mask = (touch_data['Time'] >= start_time) & (touch_data['Time'] <= end_time)
                    period_touch_data = touch_data[mask].copy()
                    period_touch_data['Posture'] = posture
                    period_touch_data['PIN'] = pin_value
                    period_touch_data['Sample ID'] = sample_id
                    period_touch_data['UUID'] = user_id

                    filtered_touch_data = pd.concat([filtered_touch_data, period_touch_data])
                    filtered_sensor_data = pd.concat([filtered_sensor_data, period_sensor_data])
            except Exception as e:
                print(f"读取用户 {user_folder} 的传感器或触摸数据时出错: {e}")
                continue
        except Exception as e:
            print(f"处理用户 {user_folder} 时出现未预期的错误: {e}")
            continue

filtered_sensor_data.to_csv(output_sensor_path, index=False)
filtered_touch_data.to_csv(output_touch_path, index=False)
print("所有用户处理完成！")