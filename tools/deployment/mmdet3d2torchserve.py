# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import mmcv

try:
    from model_archiver.model_packaging import package_model
    from model_archiver.model_packaging_utils import ModelExportUtils
except ImportError:
    package_model = None

def mmdet3d2torchserve(config_file: str, checkpoint_file: str, output_folder: str, model_name: str, model_version: str = '1.0', force: bool = False):
    """将MMDetection3D模型(配置+检查点)转换为TorchServe的.mar格式。"""
    mmcv.mkdir_or_exist(output_folder)
    config = mmcv.Config.fromfile(config_file)
    with TemporaryDirectory() as tmpdir:
        config.dump(f'{tmpdir}/config.py')
        args = Namespace(
            model_file=f'{tmpdir}/config.py',
            serialized_file=checkpoint_file,
            handler=f'{Path(__file__).parent}/mmdet3d_handler.py',
            model_name=model_name or Path(checkpoint_file).stem,
            version=model_version,
            export_path=output_folder,
            force=force,
            requirements_file=None,
            extra_files=None,
            runtime='python',
            archive_format='default')
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest)

def parse_args():
    parser = ArgumentParser(description='将MMDetection模型转换为TorchServe的.mar格式')
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('checkpoint', type=str, help='检查点文件路径')
    parser.add_argument('--output-folder', type=str, required=True, help='输出.mar文件的文件夹路径')
    parser.add_argument('--model-name', type=str, default=None, help='模型名称,用于命名输出的.mar文件,若为None则使用检查点文件名')
    parser.add_argument('--model-version', type=str, default='1.0', help='模型版本号')
    parser.add_argument('-f', '--force', action='store_true', help='是否覆盖已存在的同名.mar文件')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if package_model is None:
        raise ImportError('需要安装torch-model-archiver,请执行: pip install torch-model-archiver')
    mmdet3d2torchserve(args.config, args.checkpoint, args.output_folder, args.model_name, args.model_version, args.force)
