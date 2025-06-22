import cv2
import os
from nuscenes.nuscenes import NuScenes

# ===============================================================
#               配置部分 - 请在此处修改
# ===============================================================
# 1. nuScenes数据集的根目录路径
#    (即包含 "maps", "samples", "sweeps", "v1.0-trainval" 等子文件夹的目录)
NUSCENES_DATA_ROOT = '/mnt/sdc/FUTR3D/data/nuscenes'

# 2. 要使用的数据集版本 ('v1.0-mini' 对应迷你版, 'v1.0-trainval' 对应完整版)
NUSCENES_VERSION = 'v1.0-trainval' # 或者 'v1.0-mini'

# 3. 用于存放输出图片的文件夹路径
OUTPUT_SAVE_PATH = 'nuscenes_output_images'
# ===============================================================


def load_and_save_6_cam_views(nusc: NuScenes, save_path: str):
    """
    加载并保存在第一个场景(scene)的第一个样本(sample)的6个摄像头视图。

    参数:
        nusc (NuScenes): nuScenes API 的主对象。
        save_path (str): 保存图片的文件夹路径。
    """
    print("正在加载第一个场景和第一个样本...")

    # 1. 从数据集中获取第一个场景
    my_scene = nusc.scene[0]

    # 2. 获取该场景中第一个样本(sample)的token（可以理解为时间戳的唯一ID）
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)

    # 3. 如果指定的保存路径不存在，则创建该文件夹
    os.makedirs(save_path, exist_ok=True)
    print(f"\n图片将被保存到这个绝对路径下: '{os.path.abspath(save_path)}'")

    # nuScenes数据集中6个摄像头的标准通道名称
    camera_channels = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
    ]

    # 4. 遍历每一个摄像头通道，以获取并保存对应的图片
    for cam_channel in camera_channels:
        # 获取当前摄像头通道在此样本中的 'sample_data' token
        cam_data_token = my_sample['data'][cam_channel]
        
        # 获取该 'sample_data' 对应的完整文件路径
        image_filepath = nusc.get_sample_data_path(cam_data_token)

        # 使用OpenCV读取源图片文件
        image = cv2.imread(image_filepath)
        if image is None:
            print(f"错误：无法读取源图片文件: {image_filepath}")
            continue

        # 5. 构建输出文件的名称
        # 文件名格式: [样本token]_[摄像头通道名].png
        output_filename = f"{first_sample_token}_{cam_channel}.png"
        full_save_path = os.path.join(save_path, output_filename)

        # 6. 将图片写入到目标文件夹中
        cv2.imwrite(full_save_path, image)
        print(f"  -> 已保存图片: {output_filename}")

    print("\n操作完成！")


if __name__ == '__main__':
    # 初始化nuScenes API接口
    print(f"正在从路径初始化NuScenes: {NUSCENES_DATA_ROOT}")
    try:
        nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATA_ROOT, verbose=True)
        
        # 调用主函数，执行加载和保存操作
        load_and_save_6_cam_views(nusc, OUTPUT_SAVE_PATH)
        
    except Exception as e:
        print("\n" + "="*50)
        print("错误：NuScenes初始化失败。")
        print(f"错误详情: {e}")
        print("\n请仔细检查以下几点:")
        print(f"1. 路径 '{NUSCENES_DATA_ROOT}' 是否正确？")
        print("2. 该文件夹下是否包含 'maps', 'samples', 'v1.0-trainval' 等子文件夹？")
        print(f"3. 您指定的版本 '{NUSCENES_VERSION}' 是否已下载并存在于该路径中？")
        print("="*50)