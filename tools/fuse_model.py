import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fuse image and lidar model weights')
    parser.add_argument('--img', type=str, required=True, help='path to image model checkpoint')
    parser.add_argument('--lidar', type=str, required=True, help='path to lidar model checkpoint')
    parser.add_argument('--out', type=str, required=True, help='output path for fused checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    # Load image model checkpoint
    img_ckpt = torch.load(args.img)
    state_dict1 = img_ckpt['state_dict']
    # Load lidar model checkpoint
    pts_ckpt = torch.load(args.lidar)
    state_dict2 = pts_ckpt['state_dict']
    # Fuse the models
    state_dict1.update(state_dict2)
    # Save fused model
    torch.save({'state_dict': state_dict1}, args.out)
    print(f'Fused model saved to {args.out}')

if __name__ == '__main__':
    main()
