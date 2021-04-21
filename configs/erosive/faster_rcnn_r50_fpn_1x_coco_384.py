_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('erosive',)
num_classes=1
data_root = '/data1/qilei_chen/DATA/erosive/'
img_scale=(384,384)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/train4.19.json',
        img_prefix=data_root+'images/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test4.19.json',
        img_prefix=data_root+'images/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test4.19.json',
        img_prefix=data_root+'images/'),)

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 1.
model = dict(
    roi_head=dict(
        bbox_head=
            dict(type='Shared2FCBBoxHead',
                num_classes=num_classes)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.5),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.1),
            max_per_img=100)))# explicitly over-write all the `num_classes` field from default 80 to 1.


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

runner = dict(type='EpochBasedRunner', max_epochs=24)
#resume_from = "/data1/qilei_chen/DATA/erosive/work_dirs/faster_rcnn_r50_fpn_1x_coco_with_empty/latest.pth"