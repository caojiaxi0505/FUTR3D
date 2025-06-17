# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.image import tensor2imgs
from mmdet3d.models import (Base3DDetector, Base3DSegmentor,SingleStageMono3DDetector)
from os import path as osp

def single_gpu_test(model,data_loader,show=False,out_dir=None,show_score_thr=0.3):
    """使用单个GPU测试模型。
    此方法使用单个GPU测试模型并提供'show'选项。
    通过设置``show=True``，可以将可视化结果保存在``out_dir``下。
    参数:
        model (nn.Module): 待测试的模型。
        data_loader (nn.Dataloader): Pytorch数据加载器。
        show (bool, optional): 是否保存可视化结果。
            默认值: True。
        out_dir (str, optional): 保存可视化结果的路径。
            默认值: None。
    返回:
        list[dict]: 预测结果。
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # 记录推理开始时间
    import time
    start_time = time.time()
    total_frames = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        total_frames += 1
        if show:
            models_3d = (Base3DDetector, Base3DSegmentor,SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data,result,out_dir=out_dir)
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)
                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]
                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None
                    model.module.show_result(img_show,result[i],show=show,out_file=out_file,score_thr=show_score_thr)
        results.extend(result)
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    # 计算推理时间
    total_time = time.time() - start_time
    fps = total_frames / total_time if total_time > 0 else 0
    print(f'处理总帧数: {total_frames}')
    print(f'总时间: {total_time:.2f} 秒')
    print(f'FPS: {fps:.2f}')
    return results
