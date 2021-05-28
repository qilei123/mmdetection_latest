_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

dataset_type = 'CocoDataset'
classes = ('erosive','ulcer')
data_root = '/data1/qilei_chen/DATA/erosive_ulcer_mix/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/train_mix.json',
        img_prefix=data_root+'images/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test_mix.json',
        img_prefix=data_root+'images/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/test_mix.json',
        img_prefix=data_root+'images/'),)

# explicitly over-write all the `num_classes` field from default 80 to 1.
model = dict(
    bbox_head=
        dict(num_classes=1))# explicitly over-write all the `num_classes` field from default 80 to 1.
runner = dict(type='EpochBasedRunner', max_epochs=24)
#resume_from = "/data1/qilei_chen/DATA/erosive/work_dirs/retinanet_r50_fpn_1x_coco_fine/latest.pth"