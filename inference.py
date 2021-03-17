from mmdet.apis import init_detector, inference_detector
import mmcv
from pycocotools.coco import COCO
import os
import cv2
# Specify the path to model config and checkpoint file
config_file = 'configs/erosive/reppoints_moment_r50_fpn_1x_coco.py'
checkpoint_file = '/data1/qilei_chen/DATA/erosive/work_dirs/reppoints_moment_r50_fpn_1x_coco/epoch_83.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# test images and show the results
set_name = 'test' #['train','test']
anns_file = '/data1/qilei_chen/DATA/erosive/annotations/'+set_name+'.json'
coco_instance = COCO(anns_file)
coco_imgs = coco_instance.imgs

for key in coco_imgs:
    annIds = coco_instance.getAnnIds(imgIds= coco_imgs[key]['id'])
    anns = coco_instance.loadAnns(annIds)

    img_file_name = coco_imgs[key]["file_name"]
    img_dir = os.path.join("/data1/qilei_chen/DATA/erosive/images",img_file_name)
    img = mmcv.imread(img_dir)

    for ann in anns:
        [x,y,w,h] = ann['bbox']
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)

    result = inference_detector(model, img)
    score_thr=0.3
    model.show_result(img, result,score_thr=score_thr,bbox_color =(255,0,0),
                    text_color = (255,0,0),font_size=5, 
                    out_file='/data1/qilei_chen/DATA/erosive/work_dirs/reppoints_moment_r50_fpn_1x_coco/'+set_name+'_result_'+str(score_thr)+'/'+img_file_name)
