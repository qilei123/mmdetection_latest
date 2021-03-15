from mmdet.apis import init_detector, inference_detector
import mmcv
from pycocotools.coco import COCO

# Specify the path to model config and checkpoint file
config_file = 'configs/erosive/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/data1/qilei_chen/DATA/erosive/work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_10.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# test images and show the results
anns_file = '/data1/qilei_chen/DATA/erosive/annotations/test.json'
coco_instance = COCO(anns_file)
coco_imgs = coco_instance.imgs
print(coco_imgs)
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# # visualize the results in a new window
# model.show_result(img, result)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')

# # test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)