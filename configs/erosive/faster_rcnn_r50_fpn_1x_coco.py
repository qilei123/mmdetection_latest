_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('erosive',)
data_root = '/data1/qilei_chen/DATA/erosive/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/train.json',
        img_prefix=data_root+'images/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test.json',
        img_prefix=data_root+'images/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test.json',
        img_prefix=data_root+'images/'),)

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 1.
model = dict(
    roi_head=dict(
        bbox_head=
            dict(type='Shared2FCBBoxHead',
                num_classes=1)),
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

#runner = dict(type='EpochBasedRunner', max_epochs=24)
#resume_from = "/data1/qilei_chen/DATA/erosive/work_dirs/faster_rcnn_r50_fpn_1x_coco/latest.pth"