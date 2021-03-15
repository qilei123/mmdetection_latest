_base_ = '../detr/detr_r50_8x2_150e_coco.py'
# 1. dataset settings
classes = ('erosive',)
data_root = '/data1/qilei_chen/DATA/erosive/'
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
    roi_head=dict(
        bbox_head=
            dict(num_classes=1)))# explicitly over-write all the `num_classes` field from default 80 to 1.
