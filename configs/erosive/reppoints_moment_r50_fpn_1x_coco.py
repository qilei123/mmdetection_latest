_base_ = '../reppoints/reppoints_moment_r50_fpn_1x_coco.py'
# 1. dataset settings
classes = ('erosive',)
data_root = '/data1/qilei_chen/DATA/erosive/'
data = dict(
    train=dict(
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/train4.19.json',
        img_prefix=data_root+'images/'),
    val=dict(
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test4.19.json',
        img_prefix=data_root+'images/'),
    test=dict(
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test4.19.json',
        img_prefix=data_root+'images/'),)

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 1.
model = dict(
    bbox_head=
        dict(num_classes=1),
    test_cfg=dict(
        nms_pre=100,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.01),
        max_per_img=100))# explicitly over-write all the `num_classes` field from default 80 to 1.

runner = dict(type='EpochBasedRunner', max_epochs=64)
resume_from = "/data1/qilei_chen/DATA/erosive/work_dirs/reppoints_moment_r50_fpn_1x_coco_4_19/latest.pth"