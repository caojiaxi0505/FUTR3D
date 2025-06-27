import argparse
import copy
import importlib
import os
import os.path as osp
import sys
import time
import torch
from mmcv import Config
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.models import build_model

def parse_args():
    parser = argparse.ArgumentParser(description='spconv weight converter')
    parser.add_argument('config', help='model cfg')
    parser.add_argument('--inckpt', required=True, default=None, help='ckpt to convert')
    parser.add_argument('--outckpt', required=True, default=None, help='converted ckpt')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = args.config
    inckpt = args.inckpt
    outckpt = args.outckpt
    print("*="*16+"*"+"\n")
    print(f">>> Cfg path: {cfg}\n")
    print(f">>> Input ckpt: {inckpt}\n")
    print(f">>> Output ckpt: {outckpt}\n")
    cfg = Config.fromfile(cfg)
    print("*="*16+"*"+"\n")
    print("Processing Plugin Dir...\n")
    plugin_paths = []
    cfg_dir = os.path.dirname(args.config)
    plugin_paths.append(cfg_dir.replace('/', '.'))
    for _module_path in plugin_paths:
        importlib.import_module(_module_path)
    print("*="*16+"*"+"\n")
    print("Building Model Structure (on CPU)...\n")
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).to('cpu')
    model.init_weights()
    print("*="*16+"*"+"\n")
    print(f"Loading ckpt from: {inckpt}\n")
    ckpt = torch.load(inckpt, map_location='cpu')
    state_dict_key = 'state_dict' if 'state_dict' in ckpt else ('model' if 'model' in ckpt else None)
    if state_dict_key:
        state_dict = ckpt[state_dict_key]
        print("*="*16+"*"+"\n")
        print(f"Found state_dict under key: '{state_dict_key}'\n")
    elif isinstance(ckpt, dict) and any('.weight' in k for k in ckpt.keys()):
        state_dict = ckpt
        print("*="*16+"*"+"\n")
        print(f"Checkpoint is treated as state_dict directly.\n")
    else:
        raise ValueError(f"Cannot find state_dict in checkpoint {inckpt}.\n")
    model_state_dict = model.state_dict()
    new_state_dict = {}
    permuted_count_1 = 0    # 使用第一种permute修复的权重
    permuted_count_2 = 0    # 使用第二种permute修复的权重
    forced_permute_count = 0    # 功能存疑
    skipped_count = 0   # 预训练权重包含的，而模型不包含的权重的数目
    copied_count = 0    # 预训练权重包含的，模型包含的，且两者形状相同的权重的数目
    warning_count = 0   # 使用两种permute都不能修复的权重
    print("*="*16+"*"+"\n")
    print("Starting state_dict conversion...\n")
    keys_to_process = list(state_dict.keys())
    print(f"Total keys in checkpoint: {len(keys_to_process)}\n")
    print(f"Total keys in model: {len(model_state_dict)}\n")
    for key in keys_to_process:
        tensor = state_dict[key]
        tensor_shape = tensor.shape
        # 预训练权重包含的，而模型不包含的
        if key not in model_state_dict:
            # new_state_dict[key] = tensor
            skipped_count += 1
            continue
        target_shape = model_state_dict[key].shape
        # 功能存疑
        if (tensor.ndim == 5 and tensor_shape == target_shape and \
            "pts_middle_encoder" in key and ".weight" in key):
            try:
                permuted_tensor_1 = tensor.permute((1, 2, 3, 4, 0))
                if permuted_tensor_1.shape != tensor_shape:
                    new_state_dict[key] = permuted_tensor_1
                    forced_permute_count += 1
                    continue
            except RuntimeError:
                pass
        # 顺序出现问题
        if tensor.ndim == 5 and tensor_shape != target_shape and set(tensor_shape) == set(target_shape):
            if tensor.permute((1, 2, 3, 4, 0)).shape == tensor.permute((4, 0, 1, 2, 3)):
                raise ValueError("非常严重的错误，性能差可能就是这里导致的")
            try:
                permuted_tensor_1 = tensor.permute((1, 2, 3, 4, 0))
                if permuted_tensor_1.shape == target_shape:
                    new_state_dict[key] = permuted_tensor_1
                    permuted_count_1 += 1
                    continue
            except RuntimeError:
                pass
            try:
                permuted_tensor_2 = tensor.permute((4, 0, 1, 2, 3))
                if permuted_tensor_2.shape == target_shape:
                    new_state_dict[key] = permuted_tensor_2
                    permuted_count_2 += 1
                    continue
            except RuntimeError:
                pass
        # 非spconv权重处理
        if tensor_shape == target_shape:
            new_state_dict[key] = tensor
            copied_count += 1
            continue
        # 使用两种permute都失败了
        print(f"[Warning]: Shape mismatch {key}: ckpt={tensor_shape}, model={target_shape}. No fix applied. Copying original.\n")
        # 直接不处理，放入输出权重中
        new_state_dict[key] = tensor
        warning_count += 1
    print("*="*16+"*"+"\n")
    print(f"Conversion Summary:\n")
    print(f"\tKeys FORCED permute type 1 {(1, 2, 3, 4, 0)}: {forced_permute_count}\n") 
    print(f"\tKeys permuted type 1 {(1, 2, 3, 4, 0)} (mismatch fix): {permuted_count_1}\n")
    print(f"\tKeys permuted type 2 {(4, 0, 1, 2, 3)} (mismatch fix): {permuted_count_2}\n")
    print(f"\tKeys copied (shape matched & not forced): {copied_count}\n") 
    print(f"\tKeys from ckpt not in model (kept in output): {skipped_count}\n")
    print(f"\tWarnings (mismatch not fixed): {warning_count}\n")
    total_permuted = forced_permute_count + permuted_count_1 + permuted_count_2
    print("*="*16+"*"+"\n")
    print(f"Preparing to save to {outckpt}\n")
    save_object = copy.deepcopy(ckpt)
    note = f'Weights permuted (Force:{forced_permute_count}, P1:{permuted_count_1}, P2:{permuted_count_2}, Warn:{warning_count}) on {time.asctime()}.\n'
    if state_dict_key:
        print(f"Replacing key '{state_dict_key}' in original checkpoint structure.\n")
        save_object[state_dict_key] = new_state_dict
        if 'meta' not in save_object or not isinstance(save_object['meta'], dict):
            save_object['meta'] = {}
            save_object['meta']['weight_conversion_note'] = note
            save_object['meta']['converter_mmdet3d_version'] = mmdet3d_version
    else:
        print("Original checkpoint was treated as state_dict directly.\n")
        save_object = {
            'state_dict': new_state_dict,
            'meta': {
                'weight_conversion_note': note,
                'converter_mmdet3d_version': mmdet3d_version
            }
        }
        print("Wrapped converted state_dict with 'state_dict' and 'meta' keys.\n")
    try:
        os.makedirs(osp.dirname(outckpt) or '.', exist_ok=True)
        torch.save(save_object, outckpt)
        print(f"SUCCESS: Saved converted checkpoint to: {outckpt}\n")
    except Exception as e:
        raise ValueError(f"ERROR saving checkpoint to {outckpt}: {e}\n")

    print("Validating by loading into model (strict=False)...")
    if warning_count == 0 and new_state_dict and (total_permuted + copied_count > 0):
        try:
            load_info = model.load_state_dict(new_state_dict, strict=False)
            print("Model loading validation finished.\n")
            missing_keys_count = len(load_info.missing_keys)
            unexpected_keys_count = len(load_info.unexpected_keys)
            if missing_keys_count > 0:
                print(f"\tValidation Missing keys: {missing_keys_count}")
                print(f"\t---> List of MISSING keys ({missing_keys_count}):")
                for k in load_info.missing_keys:
                    print(f"\t\t{k}")
            print(f"\tValidation Unexpected keys: {unexpected_keys_count} (Expected: {skipped_count} from keys not in model definition)\n") 
            if unexpected_keys_count > 0:
                print(f"  ---> List of UNEXPECTED keys ({unexpected_keys_count}):")
                # print(f"  ---> List of UNEXPECTED keys: {load_info.unexpected_keys}")
        except RuntimeError as e:
            print(f"\nVALIDATION ERROR: model.load_state_dict failed: {e}.\n >>> Runtime shape mismatch detected despite conversion! Check logs/warnings above!!!")
        except Exception as e:
            print(f"\nERROR during validation load: {e}. Check conversion logs/warnings above.")
    elif warning_count > 0:
        print(f"VALIDATION SKIPPED: {warning_count} warnings (unfixed mismatches) occurred during conversion. Check logs.")
    else:
        print("VALIDATION SKIPPED: new_state_dict is empty or no keys were successfully processed.")
    print("\n>>> Script finished.")
    sys.exit(0)

if __name__ == '__main__':
    main()