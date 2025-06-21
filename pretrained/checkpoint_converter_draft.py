# temp_file_converter_forced_fixed.py
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
# import torch.nn as nn # Not directly used

from mmcv import Config, DictAction
# from mmcv.runner import get_dist_info # Not directly used
from mmdet import __version__ as mmdet_version
# from mmdet.apis import set_random_seed # Not directly used
from mmdet3d import __version__ as mmdet3d_version
# from mmdet3d.apis import init_random_seed # Not directly used
from mmdet3d.models import build_model
# from mmdet3d.utils import get_root_logger # Not directly used
# from mmseg import __version__ as mmseg_version # Not directly used
from os import path as osp

# Robust import for setup_multi_processes
try:
    from mmdet.utils import setup_multi_processes
except ImportError:
     try:
        from mmdet3d.utils import setup_multi_processes
     except ImportError:
         print("Warning: cannot import setup_multi_processes from mmdet or mmdet3d")
         setup_multi_processes = lambda cfg: None # dummy function if all imports fail


def parse_args():
    parser = argparse.ArgumentParser(description='转换并保存模型权重 Convert and Save Model Checkpoint')
    parser.add_argument('config', help='模型配置文件路径 (Model config file path)')
    parser.add_argument(
         '--checkpoint',
         required=True, 
         help='需要转换的输入checkpoint路径 (Path to the input checkpoint to convert)')
    parser.add_argument(
        '--output',
        default=None,
         help='保存转换后checkpoint的路径。默认为在原文件后加_forced (Path to save the converted checkpoint. Default: input_name + _forced.pth)'
     )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpu-id', type=int, default=0, help='要使用的gpu ids（仅适用于非分布式训练）')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖使用的配置中的某些设置')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
   
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    # clean up other args parsing slightly
    return args

# Define Permutations
# Based on Error Log: [A, B, C, D, E] -> [B, C, D, E, A] => (1, 2, 3, 4, 0)
PERMUTE_1 = (1, 2, 3, 4, 0) 
# spconv 1.x to 2.x: [K, K, K, Cin, Cout] -> [Cout, K, K, K, Cin] 
PERMUTE_2 = (4, 0, 1, 2, 3) 

def main():
    args = parse_args()
    
    checkpoint_path = args.checkpoint
    if not osp.exists(checkpoint_path):
         print(f"ERROR: Input checkpoint not found at {checkpoint_path}")
         sys.exit(1)
         
    if args.output is None:
        input_dir, input_filename = osp.split(checkpoint_path)
        input_name, input_ext = osp.splitext(input_filename)
        # Use _forced name 
        output_path = osp.join(input_dir, f"{input_name}_forced{input_ext}")
    else:
        output_path = args.output
    if checkpoint_path == output_path:
         print(f"ERROR: Input and output paths are the same: {checkpoint_path}. Please specify a different output path.")
         sys.exit(1)
         
    print(f"\n>>> Config path: {args.config}")
    print(f">>> Input checkpoint: {checkpoint_path}")
    print(f">>> Output checkpoint: {output_path}\n")

    try:
       cfg = Config.fromfile(args.config)
    except Exception as e:
       print(f"ERROR loading config {args.config}: {e}")
       sys.exit(1)
       
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        
    if hasattr(cfg, 'plugin'):
       # ... plugin loading as before ...
        if cfg.plugin:
            print(">>> Loading plugins...")
            plugin_paths = []
            if hasattr(cfg, 'plugin_dir'):
                 # Simplified plugin loading 
                 if isinstance(cfg.plugin_dir, str):
                      plugin_paths.append(cfg.plugin_dir.replace('/', '.'))
                 elif isinstance(cfg.plugin_dir, list):
                       plugin_paths.extend([p.replace('/', '.') for p in cfg.plugin_dir])
            else:
                 # guess from config path
                  cfg_dir = os.path.dirname(args.config)
                  if cfg_dir:
                     plugin_paths.append(cfg_dir.replace('/', '.'))

            if not plugin_paths:
                 print("  WARNING: cfg.plugin=True but no plugin_dir found or derived.")
                 
            for _module_path in plugin_paths:
                 print(f"  Trying Plugin path: {_module_path}")
                 try:
                     plg_lib = importlib.import_module(_module_path)
                     print(f"  Successfully loaded plugin: {_module_path}")
                 except Exception as e:
                      print(f"   WARNING/ERROR loading plugin {_module_path}: {e}")
            print(">>> Plugins loading process finished.")


    cfg.gpu_ids = [args.gpu_id]

    print(">>> Building model structure...")
    try:
       # Ensure model is on CPU to avoid OOM during build/load
       model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).to('cpu')
       model.init_weights() 
       print(">>> Model structure built (on CPU).")
    except Exception as e:
       print(f"ERROR: Failed to build model from config {args.config}. Check your environment, config, and plugins.")
       print(f"Details: {e}")
       import traceback
       traceback.print_exc()
       sys.exit(1)

    print(f">>> Loading checkpoint from: {checkpoint_path}")
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
    
    # <<<--- COUNTER INITIALIZATION START --- >>>
    permuted_count_1 = 0
    permuted_count_2 = 0
    forced_permute_count = 0 # FIX: Initialize the counter for forced permutations
    skipped_count = 0      # key in ckpt not in model
    copied_count = 0       # shape matched (and not forced)
    warning_count = 0      # mismatch not fixed
     # <<<--- COUNTER INITIALIZATION END --- >>>


    print("\n>>> Starting state_dict conversion...")
    keys_to_process = list(state_dict.keys())
    print(f"Total keys in checkpoint: {len(keys_to_process)}")
    print(f"Total keys in model: {len(model_state_dict)}")

    # --- REVISED LOOP LOGIC V3 START ---
    for key in keys_to_process:
        tensor = state_dict[key]
        tensor_shape = tensor.shape
        
        # PATH 0: Key not in model
        if key not in model_state_dict:
             new_state_dict[key] = tensor 
             skipped_count += 1
             continue 
        
        # Key IS in model, get target shape
        target_shape = model_state_dict[key].shape
        
        # PATH 1: *** FORCE PERMUTE *** for "state_dict shape lies" case 
        # Check: 5D, shape seems to match, key contains pattern, and permute_1 actually changes the shape
        # Based on error log, PERMUTE_1 is the target permutation
        # This MUST come BEFORE the standard `tensor_shape == target_shape` check (PATH 3)
        if (tensor.ndim == 5 and 
            tensor_shape == target_shape and # The condition that caused the previous failure
            "pts_middle_encoder" in key and ".weight" in key # Filter specific layers based on error log
           ):
             try:
                permuted_tensor_1 = tensor.permute(PERMUTE_1)
                # Only apply if permute actually changes the shape (avoids symmetric cases like [64,64,3,3,3])
                if permuted_tensor_1.shape != tensor_shape:
                     # print(f"  FORCE_P1 {PERMUTE_1}: {key}. Shape match {tensor_shape} but detected candidate. New shape: {permuted_tensor_1.shape}")
                     new_state_dict[key] = permuted_tensor_1
                     forced_permute_count += 1  # FIX: Now this will work
                     continue # Move to next key
             except RuntimeError:
                  pass # if permute fails for some reason, fall through to other checks

        # PATH 2: Handle genuine shape mismatches for 5D tensors (elements match, order differs)
        if tensor.ndim == 5 and tensor_shape != target_shape and set(tensor_shape) == set(target_shape):
             # Try PERMUTE_1
            try:
                permuted_tensor_1 = tensor.permute(PERMUTE_1)
                if permuted_tensor_1.shape == target_shape:
                     # print(f"  PERMUTE_1 {PERMUTE_1}: {key} from {tensor_shape} to {permuted_tensor_1.shape} MATCH model")
                     new_state_dict[key] = permuted_tensor_1
                     permuted_count_1 += 1
                     continue # Move to next key
            except RuntimeError:
                 pass
            # Try PERMUTE_2
            try:
                 permuted_tensor_2 = tensor.permute(PERMUTE_2)
                 if permuted_tensor_2.shape == target_shape:
                      # print(f"  PERMUTE_2 {PERMUTE_2}: {key} from {tensor_shape} to {permuted_tensor_2.shape} MATCH model")
                      new_state_dict[key] = permuted_tensor_2
                      permuted_count_2 += 1
                      continue # Move to next key
            except RuntimeError:
                  pass
             # If neither permute worked, fall through to PATH 4 (warning)
       
        # PATH 3: Shapes match exactly (AND not caught by the FORCE PERMUTE logic in PATH 1)
        # This handles bias, bn, symmetric 5D kernels, etc. correctly now.
        if tensor_shape == target_shape:
            new_state_dict[key] = tensor
            copied_count += 1
            continue # Move to next key
            
        # PATH 4: All other cases are shape mismatches that we cannot fix with the defined permutations
        # Includes failed attempts from PATH 1 and PATH 2
        print(f"  [Warning]: Shape mismatch {key}: ckpt={tensor_shape}, model={target_shape}. No fix applied. Copying original.")
        new_state_dict[key] = tensor # Copy original tensor, load_state_dict will complain
        warning_count += 1
        # continue implicit
        
    # --- REVISED LOOP LOGIC V3 END ---
    
    # <<<--- SUMMARY UPDATE START --- >>>
    print(f"\n>>> Conversion Summary:")
    print(f"    Keys FORCED permute type 1 {PERMUTE_1}: {forced_permute_count}") 
    print(f"    Keys permuted type 1 {PERMUTE_1} (mismatch fix): {permuted_count_1}")
    print(f"    Keys permuted type 2 {PERMUTE_2} (mismatch fix): {permuted_count_2}")
    print(f"    Keys copied (shape matched & not forced): {copied_count}") 
    print(f"    Keys from ckpt not in model (kept in output): {skipped_count}")
    print(f"    Warnings (mismatch not fixed): {warning_count}") 
    
    total_permuted = forced_permute_count + permuted_count_1 + permuted_count_2
    # <<<--- SUMMARY UPDATE END --- >>>

    
    print(f"\n>>> Preparing to save to {output_path}")
    save_object = copy.deepcopy(ckpt) 
    # <<<--- NOTE UPDATE START --- >>>
    note = f'Weights permuted (Force:{forced_permute_count}, P1:{permuted_count_1}, P2:{permuted_count_2}, Warn:{warning_count}) on {time.asctime()}.'
     # <<<--- NOTE UPDATE END --- >>>
    if state_dict_key:
       print(f">>> Replacing key '{state_dict_key}' in original checkpoint structure.")
       save_object[state_dict_key] = new_state_dict
       if 'meta' not in save_object or not isinstance(save_object['meta'], dict) :
            save_object['meta'] = {}
       save_object['meta']['weight_conversion_note'] = note
       save_object['meta']['converter_mmdet3d_version'] = mmdet3d_version
    else:
       print(">>> Original checkpoint was treated as state_dict directly.")
       # Wrap it standard format if it was a bare state_dict
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
    # Validate only if there were no unfixed warnings AND something was actually processed
    if warning_count == 0 and new_state_dict and (total_permuted + copied_count > 0) :
        try:
           # model.load_state_dict performs the actual shape validation required by the layers
           load_info = model.load_state_dict(new_state_dict, strict=False) 
           print(">>> Model loading validation finished.")
           
           missing_keys_count = len(load_info.missing_keys)
           unexpected_keys_count = len(load_info.unexpected_keys)

           if missing_keys_count > 0:
              print(f"  Validation Missing keys: {missing_keys_count}")
              # print(f"    {load_info.missing_keys}") # uncomment to see list
              
           print(f"  Validation Unexpected keys: {unexpected_keys_count} (Expected: {skipped_count} from keys not in model definition)") 
           # print(f"    {load_info.unexpected_keys}") # uncomment to see list

           # Check if unexpected keys exactly match those we deliberately skipped
           if missing_keys_count == 0 and unexpected_keys_count == skipped_count :
                 print("\n  VALIDATION SUCCESS: All shapes matched, no missing keys, unexpected keys match those not present in model definition.")
           else:
                 print( "\n  VALIDATION NOTE: Loaded with strict=False, but check missing/unexpected key counts above. There might still be issues.")
                 
        except RuntimeError as e:
             # This is the crucial catch for size mismatch during loading
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

