CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=13546 \
./tools/dist_train.sh \
    configs/polyp/retinanet_r50_fpn_1x_coco_512.py 4 \
    '--work-dir /data2/qilei_chen/DATA/new_polyp_data_combination/work_dirs/retinanet_r50_fpn_1x_coco_512/'