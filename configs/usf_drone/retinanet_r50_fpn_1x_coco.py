_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

dataset_type = 'CocoDataset'
classes = ('car','truck')
data_root = '/data2/qilei_chen/DATA/usf_drone/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/USF_drone_train.json',
        img_prefix=data_root+'images/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/USF_drone_test.json',
        img_prefix=data_root+'images/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file= data_root+'annotations/USF_drone_test.json',
        img_prefix=data_root+'images/'),)

# explicitly over-write all the `num_classes` field from default 80 to 1.
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
model = dict(
    bbox_head=
        dict(num_classes=2))# explicitly over-write all the `num_classes` field from default 80 to 1.
runner = dict(type='EpochBasedRunner', max_epochs=24)
#resume_from = "/data1/qilei_chen/DATA/erosive_ulcer_mix/work_dirs/retinanet_r50_fpn_1x_coco/latest.pth"