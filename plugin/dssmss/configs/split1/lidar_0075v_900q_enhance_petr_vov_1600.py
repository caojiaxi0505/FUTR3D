_base_ = [
    "../../../../configs/_base_/datasets/nus-3d.py",
    # "../../../../configs/_base_/schedules/cyclic_20e.py",
    "../../../../configs/_base_/default_runtime.py",
]
plugin = "plugin/futr3d"
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False
)
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
# img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)   # PETR与DETR3D不同的地方
center_head = dict(
    type="CenterHead",
    in_channels=sum([256, 256]),
    tasks=[
        dict(num_class=1, class_names=["car"]),
        dict(num_class=2, class_names=["truck", "construction_vehicle"]),
        dict(num_class=2, class_names=["bus", "trailer"]),
        dict(num_class=1, class_names=["barrier"]),
        dict(num_class=2, class_names=["motorcycle", "bicycle"]),
        dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
    ],
    common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
    share_conv_channel=64,
    bbox_coder=dict(
        type="CenterPointBBoxCoder",
        pc_range=point_cloud_range[:2],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=500,
        score_threshold=0.1,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        code_size=9,
    ),
    separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
    loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
    loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
    norm_bbox=True,
)
model = dict(
    type="FUTR3D",
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_grid_mask=True,
    freeze_backbone=True,
    img_backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2), 
    pts_voxel_layer=dict(
        max_num_points=-1,
        voxel_size=voxel_size,
        max_voxels=(-1, -1),
        point_cloud_range=point_cloud_range,
    ),
    pts_voxel_encoder=dict(
        type="DynamicVFE",
        in_channels=5,
        feat_channels=[64, 128],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type="HEDNet",
        in_channels=128,
        sparse_shape=[41, 1440, 1440],
        model_cfg=dict(
            FEATURE_DIM=128,
            NUM_LAYERS=2,
            NUM_SBB=[2, 1, 1],
            DOWN_STRIDE=[1, 2, 2],
            DOWN_KERNEL_SIZE=[3, 3, 3],
        ),
    ),
    pts_backbone=dict(
        type="CascadeDEDBackbone",
        in_channels=256,
        model_cfg=dict(
            USE_SECONDMAMBA=False,
            FEATURE_DIM=256,
            NUM_LAYERS=4,
            NUM_SBB=[2, 1, 1],
            DOWN_STRIDES=[1, 2, 2],
        ),
    ),
    pts_neck=dict(
        type="FPN",
        norm_cfg=dict(type="BN2d", eps=1e-3, momentum=0.01),
        act_cfg=dict(type="ReLU", inplace=False),
        in_channels=[256],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type="FUTR3DHead",
        use_dab=True,
        # ---- dss相关参数 -----------------
        use_dss=True,
        use_hybrid=False,
        dss_date_version="0511",
        dss_drop_prob=0.3,
        dss_mamba_version="DSSMamba_Huge_EP2",
        dss_num_layers=2,
        dss_use_morton=True,
        dss_use_conv=True,
        dss_use_xy=True,
        dss_use_rope=True,
        dss_stack=True,
        dss_strong_cls=True,
        # ---- dss相关参数 -----------------
        anchor_size=3,
        # use_aux=True,
        # aux_head=center_head,
        # mix_selection=False,
        num_query=900,
        num_classes=10,
        in_channels=256,
        pc_range=point_cloud_range,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        transformer=dict(
            type="FUTR3DTransformer",
            use_dab=True,
            decoder=dict(
                type="FUTR3DTransformerDecoder",
                num_layers=6,
                use_dab=True,
                anchor_size=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="FUTR3DAttention",
                            use_lidar=True,
                            use_camera=True,
                            use_radar=False,
                            pc_range=point_cloud_range,
                            embed_dims=256,
                        ),
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0),
    ),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            pc_range=point_cloud_range,
            grid_size=[1440, 1440, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(type="IoUCost", weight=0),
            ),
        )
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type="circle",
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            max_num=300,
            score_threshold=0,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        )
    ),
)
dataset_type = "NuScenesDataset"
data_root = "data/nuscenes/"
file_client_args = dict(backend="disk")
ida_aug_conf = {
    "resize_lim": (0.94, 1.25),
    "final_dim": (640, 1600),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
train_pipeline = [
    # ==== both modality
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    # ==== img modality ====
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),    # PETR特有，添加到mmdet3d.datasets.pipelines.transforms_3d
    dict(type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True), # PETR特有，添加到mmdet3d.datasets.pipelines.transforms_3d
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    # ==== lidar modality ====
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    # ==== last ====
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "img", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    # ==== img modality
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    # ==== lidar modality ====
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # ==== futr3d特有 ====
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
            # ==== futr3d特有 ====
            dict(
                type="DefaultFormatBundle3D",
                class_names=class_names,
                with_label=False),
            dict(type="Collect3D", keys=["points", "img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
    ),
    val=dict(
        pipeline=test_pipeline,
        classes=class_names,
        ann_file=data_root + "nuscenes_infos_val.pkl",
        modality=input_modality,
    ),
    test=dict(
        pipeline=test_pipeline,
        classes=class_names,
        ann_file=data_root + "nuscenes_infos_val.pkl",
        modality=input_modality,
    ),
)
find_unused_parameters = True
cudnn_benchmark = True
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
# ==== Strategy ====
runner = dict(type="EpochBasedRunner", max_epochs=6)
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
            "img_neck": dict(lr_mult=0.1),
            "pts_middle_encoder": dict(lr_mult=0.1),
            "pts_backbone": dict(lr_mult=0.1),
            "pts_neck": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

load_from = 'pretrained/petr_fused_convert.pth'