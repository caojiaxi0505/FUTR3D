# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import subprocess
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='处理要发布的检查点')
    parser.add_argument('in_file', help='输入检查点文件名')
    parser.add_argument('out_file', help='输出检查点文件名')
    return parser.parse_args()

def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])

def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)

if __name__ == '__main__':
    main()
