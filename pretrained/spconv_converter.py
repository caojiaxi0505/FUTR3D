import torch
import argparse
import copy 
import importlib
import mmcv
import os
import time
import torch
import warnings
import sys
import traceback
from mmcv import Config, DictAction
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.models import build_model
from os import path as osp


def parse_args():
    parser = argparse.ArgumentParser(description='spconv weight converter')
    parser.add_argument('config', help='model cfg')
    parser.add_argument('--inckpt', required=True, default=None, help='ckpt to convert')
    parser.add_argument('--outckpt', required=True, default=None, help='converted ckpt')
    args = parser.parse_args()
    return args

PERMUTE_1 = (1, 2, 3, 4, 0)
PERMUTE_2 = (4, 0, 1, 2, 3)

def main():
    args = parse_args()
    cfg = args.config
    inckpt = args.inckpt
    outckpt = args.outckpt
    print("\n")
    print("="*32+"\n")
    print(f">>> Cfg path: {cfg}\n")
    print(f">>> Input ckpt: {inckpt}\n")
    print(f">>> Output ckpt: {outckpt}\n")
    cfg = Config.fromfile(cfg)
    print(f">>> Processing plugin dir...")
    plugin_paths = []
    cfg_dir = os.path.dirname(cfg)
    plugin_paths.append(cfg_dir.replace('/', '.'))
    for _module_path in plugin_paths:
       plg_lib = importlib.import_module(_module_path)
    print(f">>> Building model structure...")
    try:
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).to('cpu')
        model.init_weights()
        print(">>> Model structure built (on CPU).")

    print(f">>> Loading checkpoint from: {inckpt}")
    try:
     ckpt = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
       print(f"ERROR loading checkpoint {checkpoint_path}: {e}")
       sys.exit(1)
       
    state_dict_key = 'state_dict' if 'state_dict' in ckpt else ('model' if 'model' in ckpt else None)
    if state_dict_key:
      state_dict = ckpt[state_dict_key]
      print(f">>> Found state_dict under key: '{state_dict_key}'")
    elif isinstance(ckpt, dict) and any('.weight' in k for k in ckpt.keys()):
       state_dict = ckpt
       state_dict_key = None 
       print(f">>> Checkpoint is treated as state_dict directly.")
    else:
        print(f"ERROR: Cannot find state_dict in checkpoint {checkpoint_path}.")
        sys.exit(1)
        
    model_state_dict = model.state_dict()
    new_state_dict = {}
    permuted_count_1 = 0
    permuted_count_2 = 0
    forced_permute_count = 0 
    skipped_count = 0    
    copied_count = 0     
    warning_count = 0    

    print("\n>>> Starting state_dict conversion...")
    keys_to_process = list(state_dict.keys())
    print(f"Total keys in checkpoint: {len(keys_to_process)}")
    print(f"Total keys in model: {len(model_state_dict)}")

    for key in keys_to_process:
       tensor = state_dict[key]
       tensor_shape = tensor.shape
       
       if key not in model_state_dict:
           new_state_dict[key] = tensor 
           skipped_count += 1
           continue 
       
       target_shape = model_state_dict[key].shape
       
       if (tensor.ndim == 5 and 
          tensor_shape == target_shape and 
          "pts_middle_encoder" in key and ".weight" in key 
         ):
           try:
             permuted_tensor_1 = tensor.permute(PERMUTE_1)
             if permuted_tensor_1.shape != tensor_shape:
                 new_state_dict[key] = permuted_tensor_1
                 forced_permute_count += 1 
                 continue 
           except RuntimeError:
               pass 

       if tensor.ndim == 5 and tensor_shape != target_shape and set(tensor_shape) == set(target_shape):
          if tensor.permute(PERMUTE_1).shape == tensor.permute(PERMUTE_2):
             raise ValueError("非常严重的错误，性能差可能就是这里导致的")
          try:
             permuted_tensor_1 = tensor.permute(PERMUTE_1)
             if permuted_tensor_1.shape == target_shape:
                 new_state_dict[key] = permuted_tensor_1
                 permuted_count_1 += 1
                 continue
          except RuntimeError:
              pass
          try:
              permuted_tensor_2 = tensor.permute(PERMUTE_2)
              if permuted_tensor_2.shape == target_shape:
                  new_state_dict[key] = permuted_tensor_2
                  permuted_count_2 += 1
                  continue 
          except RuntimeError:
               pass
      
       if tensor_shape == target_shape:
          new_state_dict[key] = tensor
          copied_count += 1
          continue 
          
       print(f"  [Warning]: Shape mismatch {key}: ckpt={tensor_shape}, model={target_shape}. No fix applied. Copying original.")
       new_state_dict[key] = tensor 
       warning_count += 1
       
    print(f"\n>>> Conversion Summary:")
    print(f"    Keys FORCED permute type 1 {PERMUTE_1}: {forced_permute_count}") 
    print(f"    Keys permuted type 1 {PERMUTE_1} (mismatch fix): {permuted_count_1}")
    print(f"    Keys permuted type 2 {PERMUTE_2} (mismatch fix): {permuted_count_2}")
    print(f"    Keys copied (shape matched & not forced): {copied_count}") 
    print(f"    Keys from ckpt not in model (kept in output): {skipped_count}")
    print(f"    Warnings (mismatch not fixed): {warning_count}")
    total_permuted = forced_permute_count + permuted_count_1 + permuted_count_2

    print(f"\n>>> Preparing to save to {output_path}")
    save_object = copy.deepcopy(ckpt) 
    note = f'Weights permuted (Force:{forced_permute_count}, P1:{permuted_count_1}, P2:{permuted_count_2}, Warn:{warning_count}) on {time.asctime()}.'
    if state_dict_key:
      print(f">>> Replacing key '{state_dict_key}' in original checkpoint structure.")
      save_object[state_dict_key] = new_state_dict
      if 'meta' not in save_object or not isinstance(save_object['meta'], dict) :
          save_object['meta'] = {}
      save_object['meta']['weight_conversion_note'] = note
      save_object['meta']['converter_mmdet3d_version'] = mmdet3d_version
    else:
      print(">>> Original checkpoint was treated as state_dict directly.")
      save_object = { 'state_dict': new_state_dict, 
                   'meta': {
                      'weight_conversion_note': note,
                      'converter_mmdet3d_version': mmdet3d_version
                    } 
                  }
      print(">>> Wrapped converted state_dict with 'state_dict' and 'meta' keys.")
      
    try:
      os.makedirs(osp.dirname(output_path) or '.', exist_ok=True)
      torch.save(save_object, output_path)
      print(f"SUCCESS: Saved converted checkpoint to: {output_path}")
    except Exception as e:
       print(f"ERROR saving checkpoint to {output_path}: {e}")
       sys.exit(1)
       
    print("\n>>> Validating by loading into model (strict=False)...")
    if warning_count == 0 and new_state_dict and (total_permuted + copied_count > 0) :
       try:
         load_info = model.load_state_dict(new_state_dict, strict=False) 
         print(">>> Model loading validation finished.")
         missing_keys_count = len(load_info.missing_keys)
         unexpected_keys_count = len(load_info.unexpected_keys)
         if missing_keys_count > 0:
            print(f"  Validation Missing keys: {missing_keys_count}")
            print(f"  ---> List of MISSING keys ({missing_keys_count}):")
            for k in load_info.missing_keys:
                print(f"      {k}")
         print(f"  Validation Unexpected keys: {unexpected_keys_count} (Expected: {skipped_count} from keys not in model definition)") 
         if unexpected_keys_count > 0:
            print(f"  ---> List of UNEXPECTED keys ({unexpected_keys_count}):")
            # 为了避免列表过长刷屏(400个)，可以只打印前/后若干个，或者全部打印
            print(f"  ---> List of UNEXPECTED keys: {load_info.unexpected_keys}") # 直接打印整个列表
            # _max_print = 50 # 最多打印多少个
            # for i, k in enumerate(load_info.unexpected_keys):
            #     print(f"      {k}")
            #     if i >= _max_print -1 and unexpected_keys_count > _max_print :
            #         print(f"      ... (and {unexpected_keys_count - _max_print} more keys)")
            #         break
       except RuntimeError as e:
           print(f"\nVALIDATION ERROR: model.load_state_dict failed: {e}.\n >>> Runtime shape mismatch detected despite conversion! Check logs/warnings above!!! <<<")
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