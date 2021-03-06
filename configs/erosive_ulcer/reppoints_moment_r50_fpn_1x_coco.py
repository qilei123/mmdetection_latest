_base_ = '../reppoints/reppoints_moment_r50_fpn_1x_coco.py'
# 1. dataset settings
classes = ('ulcer','erosive')
num_classes = 2
data_root = '/data1/qilei_chen/DATA/erosive_ulcer/'
data = dict(
    train=dict(
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/train.json',
        img_prefix=data_root+'images/'),
    val=dict(
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test.json',
        img_prefix=data_root+'images/'),
    test=dict(
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test.json',
        img_prefix=data_root+'images/'),)

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 1.
model = dict(
    bbox_head=
        dict(num_classes=num_classes),
    test_cfg=dict(
        nms_pre=100,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.01),
        max_per_img=100))# explicitly over-write all the `num_classes` field from default 80 to 1.

runner = dict(type='EpochBasedRunner', max_epochs=96)
#resume_from = "/data1/qilei_chen/DATA/erosive_ulcer/work_dirs/reppoints_moment_r50_fpn_1x_coco/latest.pth"