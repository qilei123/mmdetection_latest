_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('erosive')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/data1/qilei_chen/DATA/erosive/annotations/train.json',
        img_prefix='/data1/qilei_chen/DATA/erosive/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/data1/qilei_chen/DATA/erosive/annotations/test.json',
        img_prefix='/data1/qilei_chen/DATA/erosive/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/data1/qilei_chen/DATA/erosive/annotations/test.json',
        img_prefix='/data1/qilei_chen/DATA/erosive/'))

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 5.
model = dict(
    roi_head=dict(
        bbox_head=
            dict(num_classes=1)))# explicitly over-write all the `num_classes` field from default 80 to 5.
                