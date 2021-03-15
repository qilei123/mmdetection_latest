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
            dict(num_classes=1)))# explicitly over-write all the `num_classes` field from default 80 to 1.
                