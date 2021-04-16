_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'CocoDataset'
num_classes=2
input_size = 384
data_root = '/data1/qilei_chen/DATA/polyp_xinzi/'
img_scale=(input_size,input_size)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)

model = dict(
    backbone=dict(
        input_size=input_size,),
    bbox_head=dict(
        num_classes=num_classes,
        anchor_generator=dict(input_size=input_size),)
)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'annotations/train.json',
            img_prefix=data_root + 'images/',
            pipeline=train_pipeline)),
    val=dict(
        type='CocoDataset',
        classes=classes,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        classes=classes,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline)   
)
# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(_delete_=True)
classes = ('Adenomatous','non-Adenomatous')