CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 \
./tools/dist_train.sh \
    configs/erosive/faster_rcnn_r50_fpn_1x_coco.py 4 \
    --work-dir=/data1/qilei_chen/DATA/erosive/work_dirs/faster_rcnn_r50_fpn_1x_coco