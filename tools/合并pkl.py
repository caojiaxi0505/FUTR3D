import pickle
import os

def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def merge_data(data_list):
    merged_data = {}
    for data in data_list:
        for key, value in data.items():
            if key not in merged_data:
                merged_data[key] = value
            else:
                if isinstance(value, list):
                    merged_data[key] += value
                else:
                    merged_data[key] = value
    return merged_data

def save_pkl(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


file1 = '/mnt/sdc/FUTR3D/data/nuscenes/nuscenes_infos_train.pkl'
file2 = '/mnt/sdc/FUTR3D/data/nuscenes/nuscenes_infos_val.pkl'
merged_file = '/mnt/sdc/FUTR3D/data/nuscenes/nuscenes_infos_trainval.pkl'
data1 = read_pkl(file1)
data2 = read_pkl(file2)
merged_data = merge_data([data1, data2])
save_pkl(merged_file, merged_data)