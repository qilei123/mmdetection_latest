# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=13546 \
# ./tools/dist_train.sh \
#     configs/erosive/faster_rcnn_r50_fpn_1x_coco.py 4 \
#     '--work-dir /data1/qilei_chen/DATA/erosive/work_dirs/faster_rcnn_r50_fpn_1x_coco/'

# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=13646 \
# ./tools/dist_train.sh \
#     configs/erosive/retinanet_r50_fpn_1x_coco.py 4 \
#     '--work-dir /data1/qilei_chen/DATA/erosive/work_dirs/retinanet_r50_fpn_1x_coco/'

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=14646 \
./tools/dist_train.sh \
    configs/erosive/detr_r50_8x2_150e_coco.py 4 \
    '--work-dir /data1/qilei_chen/DATA/erosive/work_dirs/detr_r50_8x2_150e_coco/'

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12646 \
./tools/dist_train.sh \
    configs/erosive/reppoints_moment_r50_fpn_1x_coco.py 4 \
    '--work-dir /data1/qilei_chen/DATA/erosive/work_dirs/reppoints_moment_r50_fpn_1x_coco/'

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=9646 \
./tools/dist_train.sh \
    configs/erosive/cascade_rcnn_r50_fpn_1x_coco.py 4 \
    '--work-dir /data1/qilei_chen/DATA/erosive/work_dirs/cascade_rcnn_r50_fpn_1x_coco/'