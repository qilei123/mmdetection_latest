_base_ = '../reppoints/reppoints_moment_r50_fpn_1x_coco.py'
# 1. dataset settings
classes = ('erosive',)
data_root = '/data1/qilei_chen/DATA/erosive/'
data = dict(
    train=dict(
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/train.json',
        img_prefix=data_root+'images_clahe/'),
    val=dict(
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test.json',
        img_prefix=data_root+'images_clahe/'),
    test=dict(
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test.json',
        img_prefix=data_root+'images_clahe/'),)

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 1.
model = dict(
    bbox_head=
        dict(num_classes=1))# explicitly over-write all the `num_classes` field from default 80 to 1.

runner = dict(type='EpochBasedRunner', max_epochs=24)
#resume_from = "/data1/qilei_chen/DATA/erosive/work_dirs/reppoints_moment_r50_fpn_1x_coco/latest.pth"