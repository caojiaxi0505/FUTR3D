from argparse import ArgumentParser
import numpy as np
import requests
from mmdet3d.apis import inference_detector, init_model

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='点云文件')
    parser.add_argument('config', help='配置文件')
    parser.add_argument('checkpoint', help='检查点文件')
    parser.add_argument('model_name', help='服务器中的模型名称')
    parser.add_argument('--inference-addr', default='127.0.0.1:8080', help='推理服务器的地址和端口')
    parser.add_argument('--device', default='cuda:0', help='用于推理的设备')
    parser.add_argument('--score-thr', type=float, default=0.5, help='3D边界框分数阈值')
    return parser.parse_args()

def parse_result(input):
    bbox = input[0]['3dbbox']
    result = np.array(bbox)
    return result

def main(args):
    model = init_model(args.config, args.checkpoint, device=args.device)
    model_result, _ = inference_detector(model, args.pcd)
    if 'pts_bbox' in model_result[0].keys():
        pred_bboxes = model_result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = model_result[0]['pts_bbox']['scores_3d'].numpy()
    else:
        pred_bboxes = model_result[0]['boxes_3d'].tensor.numpy()
        pred_scores = model_result[0]['scores_3d'].numpy()
    model_result = pred_bboxes[pred_scores > 0.5]

    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.pcd, 'rb') as points:
        response = requests.post(url, points)
    server_result = parse_result(response.json())
    assert np.allclose(model_result, server_result)

if __name__ == '__main__':
    args = parse_args()
    main(args)
