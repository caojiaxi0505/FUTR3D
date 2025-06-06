# FUTR3D

## 项目框架说明
```
FUTR3D
├── causal-conv1d/      # 不作修改
├── checkpoints/        # 存储预训练权重
├── configs/            # 不作修改，存储官方支持模型配置
├── data/               # 数据集
├── mamba/              # 不作修改
├── mamba_experimental/ # 实验性mamba代码
├── mmdet3d/            # 核心组件
|   ├── apis/
|   ├── core/
|   ├── datasets/
|   ├── models/
|   ├── ops/
|   └── utils/
├── plugin/             # 核心组件
|   ├── dssmss/
|   └── futr3d/
└── tools/

```

## 环境配置
```
conda create -n futr3d python=3.8 -y
conda activate futr3d
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install mmcv-full==1.7.0
pip install mmdet==2.27.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mmsegmentation==0.30.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install nuscenes-devkit -i https://pypi.tuna.tsinghua.edu.cn/simple
cd FUTR3D/
pip install -v -e . 
pip install numpy==1.23.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install yapf==0.40.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
cd causal-conv1d
pip install -v -e .
cd ../
cd mamba
pip install -v -e .
```

## 性能

| iter | batch size | num GPUs | mAP | NDS | SyncBN | log |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2 | 1 | 0.6393 | 0.6925 | x | [log](work_dirs/lidar_0075v_900q_split1/一张卡每张卡2_BN2d/20250404_124520.log) |
| 14 | 4 | 2 | 0.5347 | 0.5979 | x | [log](work_dirs/lidar_0075v_900q_split14/两张卡每张卡4_BN2d/20250413_195705.log) |
| 40 | 1 | 4 | 0.3683 | 0.3664 | v | [log](work_dirs/lidar_0075v_900q_split40/四张卡每张卡1_SyBN2d/20250425_155400.log) |


## 对比实验
### SPLIT 1


### SPLIT 14
| Method | iter | mAP | NDS | 说明 | 详细说明 | cfg | log |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FUTR3D-L-bs4x2 | 14 | 0.5347 | 0.5979 | baseline，backbone为SECOND，未使用SyncBN | FPN输入使用128和256 | [cfg](work_dirs/lidar_0075v_900q_split14/两张卡每张卡4_BN2d/lidar_0075v_900q.py) | [log](work_dirs/lidar_0075v_900q_split14/两张卡每张卡4_BN2d/20250413_195705.log) |
| FUTR3D-hednetbackbone-bs4x2 | 14 | 0.5614 | 0.6135 | 将backbone换为hednet，使用SyncBN | hednet堆叠4层，每层12个Conv2d，2个ConvTranspose2d，Channel为256，FPN输入只使用256 | [cfg](work_dirs/lidar_0075v_900q_split14_hednetbackbone/两张卡每张卡4_SyBN2d/lidar_0075v_900q_split14_cascadeded.py) | [log](work_dirs/lidar_0075v_900q_split14_hednetbackbone/两张卡每张卡4_SyBN2d/20250425_125602.log) |
| FUTR3D-hednetbackbone-bs2x2 | 14 | 0.5499 | 0.5961 | 将backbone换为hednet，使用SyncBN | hednet堆叠4层，每层12个Conv2d，2个ConvTranspose2d，Channel为256，FPN输入只使用256 | [cfg](work_dirs/lidar_0075v_900q_split14_hednetbackbone/两张卡每张卡4_SyBN2d/lidar_0075v_900q_split14_cascadeded.py) | [log](work_dirs/lidar_0075v_900q_split14_hednetbackbone/两张卡每张卡2_SyBN2d/20250501_132035.log) |
| FUTR3D-hednetbackbone4-secondmamba1-bs2x2 | 14 | 0.5441 | 0.5903 | 将backbone换为hednet，使用SyncBN | hednet堆叠4层，每层12个Conv2d，2个ConvTranspose2d，Channel为256，后添加一层SECONDMambaBlock，FPN输入只使用256 | [cfg](work_dirs/lidar_0075v_900q_split14_hednetbackbone4_secondmamba1/两张卡每张卡2_SyBN2d/fix2.py) | [log](work_dirs/lidar_0075v_900q_split14_hednetbackbone4_secondmamba1/两张卡每张卡2_SyBN2d/20250505_025221.log) |


### SPLIT 40
| Method | iter | mAP | NDS | 说明 | 详细说明 | cfg | log | memory | time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FUTR3D-L-bs1x4 | 40 | 0.3683 | 0.3664 | baseline，backbone为SECOND，未使用SyncBN | FPN输入使用128和256 | [cfg](work_dirs/lidar_0075v_900q_split40/四张卡每张卡1_SyBN2d/lidar_0075v_900q_split40.py) | [log](work_dirs/lidar_0075v_900q_split40/四张卡每张卡1_SyBN2d/20250425_155400.log) | 3567 | 4h42min |
| FUTR3D-hetnetbackbone-bs1x4 | 40 | 0.4125 | 0.4050 | backbone换成hednet，使用SyncBN | hednet堆叠4层，每层12个Conv2d，2个ConvTranspose2d，Channel为256，FPN输入只使用256 | [cfg](work_dirs/lidar_0075v_900q_split40_hednetbackbone_split40/四张卡每张卡1_SyBN2d/lidar_0075v_900q_cascadeded_split40.py) | [log](work_dirs/lidar_0075v_900q_split40_hednetbackbone_split40/四张卡每张卡1_SyBN2d/20250425_071531.log) | 5774 | 6h1min |
| FUTR3D-hednetmiddleencoder128-hednetbackbone-bs1x4 | 40 | 0.4131 | 0.3893 | backbone和middle encoder都换成hednet，使用SyncBN | hednet堆叠4层，每层12个Conv2d，2个ConvTranspose2d，Channel为128，FPN输入只使用128 | [cfg](work_dirs/lidar_0075v_900q_split40_hednetmiddleencoder128_hednetbackbone/四张卡每张卡1_SyBN2d/lidar_0075v_900q_hednet_hednet_split40.py) | [log](work_dirs/lidar_0075v_900q_split40_hednetmiddleencoder128_hednetbackbone/四张卡每张卡1_SyBN2d/20250506_091518.log) | 5625 | 6h10min |
| FUTR3D-hednetmiddleencoder256-hednetbackbone-bs1x4 | 40 | 0.4722 | 0.4382 | backbone和middle encoder都换成hednet，使用SyncBN | hednet堆叠4层，每层12个Conv2d，2个ConvTranspose2d，Channel为256，FPN输入只使用256 | [cfg](work_dirs/lidar_0075v_900q_split40_hednetmiddleencoder256_hednetbackbone/四张卡每张卡1_SyBN2d/lidar_0075v_900q_hednet_hednet_split40_256.py) | [log](work_dirs/lidar_0075v_900q_split40_hednetmiddleencoder256_hednetbackbone/四张卡每张卡1_SyBN2d/20250506_010513.log) | 8071 | 8h6min |


## 测试说明
测试默认使用一张GPU，每张GPU上放一个Sample，验证集共包含6019个Samples

## SECOND-Backbone说明
Block0
```
Sequential(
  (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (4): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
  (6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (7): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (8): ReLU(inplace=True)
  (9): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (10): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (11): ReLU(inplace=True)
  (12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (13): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (14): ReLU(inplace=True)
  (15): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (16): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (17): ReLU(inplace=True)
)
```
Block1
```
Sequential(
  (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (4): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
  (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (7): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (8): ReLU(inplace=True)
  (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (10): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (11): ReLU(inplace=True)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (13): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (14): ReLU(inplace=True)
  (15): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (16): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
  (17): ReLU(inplace=True)
)
```


FPS: 4.56
mAP: 0.5339                                                                          
mATE: 0.3768
mASE: 0.2713
mAOE: 0.3987
mAVE: 0.4499
mAAE: 0.2031
NDS: 0.5969
Eval time: 51.1s
Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
car                     0.799   0.210   0.162   0.134   0.405   0.217 
truck                   0.498   0.415   0.215   0.169   0.353   0.222 
bus                     0.601   0.435   0.211   0.138   0.991   0.274 
trailer                 0.351   0.615   0.222   0.568   0.294   0.178 
construction_vehicle    0.185   0.794   0.463   0.975   0.145   0.357 
pedestrian              0.756   0.244   0.287   0.615   0.425   0.134 
motorcycle              0.558   0.261   0.260   0.361   0.729   0.232 
bicycle                 0.399   0.193   0.268   0.535   0.257   0.010 
traffic_cone            0.597   0.211   0.347   nan     nan     nan   
barrier                 0.595   0.390   0.277   0.093   nan     nan 


### SPLIT14

#### GPU:2 & batch size:2

| Method | num_GPUs | batch_size | SyncBN | mAP | NDS | time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| lidar_0075v_900q | 2 | 2 | v | 0.5082 | 0.5534 | 7h21min(val4) |
| lidar_0075v_900q_hednetbackbone4 | 2 | 2 | v | 0.5499 | 0.5961 | 10h29min(val4)(+42.63%) |

#### GPU:4 & batch size:4

| Method | num_GPUs | batch_size | SyncBN | mAP | NDS | time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| lidar_0075v_900q | 2 | 4 | v | 0.5343 | 0.5977 | 6h22min(val4) |
| lidar_0075v_900q | 2 | 4 | x | 0.5347 | 0.5979 | 8h48min(val20) |
| lidar_0075v_900q_hednetbackbone4 | 2 | 4 | v | 0.5614 | 0.6135 | 8h47min(val4)(+37.96%) |


### SPLIT40

#### 