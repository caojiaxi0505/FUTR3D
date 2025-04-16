import torch
img_ckpt = torch.load('work_dirs/vovnet_trainval/epoch_24.pth')
state_dict1 = img_ckpt['state_dict']
pts_ckpt = torch.load('work_dirs/co_dab_lidar_0075v_900q_trainval/epoch_20.pth')
state_dict2 = pts_ckpt['state_dict']
state_dict1.update(state_dict2)
torch.save({'state_dict':state_dict1}, 'checkpoint/lidar_vov_trainval.pth')
