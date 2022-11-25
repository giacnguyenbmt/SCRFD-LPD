lr_mult = 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.001,
    step=[30*lr_mult, 55*lr_mult, 68*lr_mult, 75*lr_mult])
total_epochs = 80*lr_mult
checkpoint_config = dict(interval=5)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = 'work_dirs/scrfd_crowdhuman_500m_fe/epoch_320.pth'
load_from = None
resume_from = None # 'work_dirs/scrfd_crowdhuman_2.5g_fe_aug/epoch_50.pth'
workflow = [('train', 1)]
dataset_type = 'RetinaVehicleDataset'
data_root = 'data/vehicle_data/'
train_root = 'data/vehicle_data/train/'
val_root = 'data/vehicle_data/val/'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0], to_rgb=True)
albu_train_transforms=[
        dict(
            type='OneOf',
            transforms=[
                dict(
                    type='Affine',
                    scale=None,
                    rotate=(-30, 30),
                    shear=None,
                    interpolation=0,
                    fit_output=True,
                    ),
                # dict(
                #     type='RandomRotate90',
                #     ),
            ],
            p=0.1
        ),
        dict(
            type='OneOf',
            transforms=[
                dict(
                    type='MotionBlur',
                    ),
                dict(
                    type='GaussianBlur',
                    blur_limit=3,
                    ),
            ],
            p=0.2
        ),
        # dict(
        #     type='OneOf',
        #     transforms=[
        #         dict(
        #             type='IAAEmboss',
        #             ),
        #         dict(
        #             type='RandomBrightnessContrast',
        #             ),
        #         dict(
        #             type='RandomBrightness',
        #             ),
        #         dict(
        #             type='RandomContrast',
        #             ),
        #     ],
        #     p=0.2
        # ),
        # dict(
        #     type='OneOf',
        #     transforms=[
        #         dict(
        #             type='ISONoise',
        #             p=0.1),
        #         dict(
        #             type='GaussNoise',
        #             p=0.1),
        #     ],
        #     p=0.2
        # ),
        # dict(
        #     type='RandomGamma',
        #     p=0.2),
        dict(
            type='ToGray',
            p=0.2),
        # dict(
        #     type='JpegCompression',
        #     quality_lower=30, 
        #     quality_upper=80,
        #     p=1),
    ]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='RetinaVehicleDataset',
        ann_file='data/vehicle_data/train/label_fullbody.txt',
        img_prefix='data/vehicle_data/train/images/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
            dict(
                type='Albu',
                transforms=albu_train_transforms,
                bbox_params=dict(
                        type='BboxParams',
                        format='pascal_voc',
                        label_fields=['gt_labels'],
                        min_visibility=0.5),
                keypoint_params=dict(
                        type='KeypointParams',
                        format='xy'),
                keymap={
                    'img': 'image',
                    'gt_bboxes': 'bboxes',
                    'gt_keypointss': 'keypoints'
                },
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='RandomSquareCrop',
                crop_choice=[
                    0.45, 0.6, 0.8, 1.0, 1.2, 1.4
                ],
                bbox_clip_border=False),
            dict(
                type='Resize',
                img_scale=(768, 768),
                keep_ratio=False,
                bbox_clip_border=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                    'gt_keypointss'
                ])
        ]),
    val=dict(
        type='RetinaVehicleDataset',
        ann_file='data/vehicle_data/val/label_fullbody.txt',
        img_prefix='data/vehicle_data/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(768, 768),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    #dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(768, 768), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RetinaVehicleDataset',
        ann_file='data/vehicle_data/val/label_fullbody.txt',
        img_prefix='data/vehicle_data/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(768, 768),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    #dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(768, 768), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
model = dict(
    type='SCRFD',
    backbone=dict(
        type='ResNetV1e',
        depth=0,
        block_cfg=dict(
            block='BasicBlock',
            stage_blocks=(3, 5, 3, 2),
            stage_planes=[24, 48, 48, 80]),
        base_channels=24,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='PAFPN',
        in_channels=[24, 48, 48, 80],
        out_channels=24,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='SCRFDHead',
        num_classes=7,
        in_channels=24,
        stacked_convs=2,
        feat_channels=64,
        norm_cfg=dict(type='BN', requires_grad=True),
        #norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
        cls_reg_share=True,
        strides_share=False,
        scale_mode=2,
        #anchor_generator=dict(
        #    type='AnchorGenerator',
        #    ratios=[1.0],
        #    scales = [1,2],
        #    base_sizes = [16, 64, 256],
        #    strides=[8, 16, 32]),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[0.5, 1.0, 2.0],
            scales = [3],
            base_sizes = [8, 16, 32, 64, 128, 256],
            strides=[8, 16, 32, 64, 128, 256]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=False,
        reg_max=8,
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
        use_kps=False,
        loss_kps=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.1),
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=-1,
            min_bbox_size=0,
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.4),
            max_per_img=-1)))
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9, mode=0),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=-1,
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=-1)
evaluation = dict(interval=40, metric='mAP')
