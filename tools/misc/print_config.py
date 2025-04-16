# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from mmcv import Config, DictAction

def parse_args():
    parser = argparse.ArgumentParser(description='打印完整配置')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--options', nargs='+', action=DictAction, help='字典形式的参数')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(f'Config:\n{cfg.pretty_text}')

if __name__ == '__main__':
    main()
