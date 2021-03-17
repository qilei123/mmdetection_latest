_base_ = '../yolo/yolov3_d53_320_273e_coco.py'
dataset_type = 'CocoDataset'
classes = ('erosive',)
data_root = '/data1/qilei_chen/DATA/erosive/'
data = dict(
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

model = dict(
    bbox_head=
        dict(num_classes=1))# explicitly over-write all the `num_classes` field from default 80 to 1.
