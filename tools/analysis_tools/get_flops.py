# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from mmcv import Config, DictAction
from mmdet3d.models import build_model
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('请升级mmcv到>0.6.2')

def parse_args():
    parser = argparse.ArgumentParser(description='训练检测器')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--shape', type=int, nargs='+', default=[40000, 4], help='输入点云大小')
    parser.add_argument('--modality', type=str, default='point', choices=['point', 'image', 'multi'], help='输入数据模态')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置文件中的一些设置,以xxx=yyy格式的键值对将被合并到配置文件中。如果要覆盖的值是列表,应该像key="[a,b]"或key=a,b。也允许嵌套的列表/元组值,例如key="[(a,b),(c,d)]"。注意引号是必需的,并且不允许有空格')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.modality == 'point':
        assert len(args.shape) == 2, '无效的输入形状'
        input_shape = tuple(args.shape)
    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3, ) + tuple(args.shape)
        else:
            raise ValueError('无效的输入形状')
    elif args.modality == 'multi':
        raise NotImplementedError('FLOPs计数器目前不支持多模态输入的模型')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError('FLOPs计数器目前不支持{}'.format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\n输入形状: {input_shape}\nFlops: {flops}\n参数量: {params}\n{split_line}')
    print('!!!如果您在论文中使用这些结果请谨慎。您可能需要检查是否支持所有操作并验证flops计算是否正确。')

if __name__ == '__main__':
    main()
