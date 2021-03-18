from mmdet.apis import init_detector, inference_detector
import mmcv
from pycocotools.coco import COCO
import os
import cv2
import json
import pickle
def test_data():
    # Specify the path to model config and checkpoint file
    model_name = 'reppoints_moment_r50_fpn_1x_coco'
    score_thr=0.25

    config_file = 'configs/erosive/'+model_name+'.py'
    checkpoint_file = '/data1/qilei_chen/DATA/erosive/work_dirs/'+model_name+'/epoch_83.pth'

    # build the model from a config file and a checkpoint file
    #model = init_detector(config_file, checkpoint_file, device='cuda:0')


    # test images and show the results
    set_name = 'test' #['train','test']
    anns_file = '/data1/qilei_chen/DATA/erosive/annotations/'+set_name+'.json'
    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs

    count_images_with_anns = 0
    count_images_without_anns = 0
    count_anns = 0
    img_file_name_list = {}
    count_zero_ann = 0

    for key in coco_imgs:
        annIds = coco_instance.getAnnIds(imgIds= coco_imgs[key]['id'])
        anns = coco_instance.loadAnns(annIds)
        if not len(anns)==0:
            count_images_with_anns+=1
            count_anns+=len(anns)
        else:
            count_images_without_anns+=1
        img_file_name = coco_imgs[key]["file_name"]
        if not img_file_name in img_file_name_list:
            img_file_name_list[img_file_name] = []
            img_file_name_list[img_file_name].append(anns)
        else:
            print(img_file_name)
            print(img_file_name_list[img_file_name])
            if len(img_file_name_list[img_file_name][0])==0:
                count_zero_ann+=1
            print(anns)
        '''
        img_file_name = coco_imgs[key]["file_name"]
        img_dir = os.path.join("/data1/qilei_chen/DATA/erosive/images",img_file_name)
        img = mmcv.imread(img_dir)

        result = inference_detector(model, img)

        for ann in anns:
            [x,y,w,h] = ann['bbox']
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
        
        model.show_result(img, result,score_thr=score_thr,bbox_color =(255,0,0),
                        text_color = (255,0,0),font_size=5, 
                        out_file='/data1/qilei_chen/DATA/erosive/work_dirs/'+model_name+'/'+set_name+'_result_'+str(score_thr)+'/'+img_file_name)
        '''
    print(count_images_with_anns)
    print(count_anns)
    print(count_images_without_anns)
    print(len(img_file_name_list))
    print(count_zero_ann)

def inference_and_save_result(model,coco_instance,img_folder_dir,result_save_dir):
    coco_imgs = coco_instance.imgs
    results = dict()
    for key in coco_imgs:
        img_file_name = coco_imgs[key]["file_name"]
        img_dir = os.path.join(img_folder_dir,img_file_name)
        img = mmcv.imread(img_dir)
        result = inference_detector(model, img)   

        results[coco_imgs[key]['id']] = dict()
        results[coco_imgs[key]['id']]['file_name'] = img_file_name
        results[coco_imgs[key]['id']]['result'] = result

    with open(result_save_dir, 'wb') as fp:
        pickle.dump(results, fp)

if __name__=="__main__":
    # Specify the path to model config and checkpoint file
    model_name = 'reppoints_moment_r50_fpn_1x_coco'
    score_thr=0.25

    config_file = 'configs/erosive/'+model_name+'.py'
    checkpoint_file = '/data1/qilei_chen/DATA/erosive/work_dirs/'+model_name+'/epoch_83.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # test images and show the results
    set_name = 'test' #['train','test']
    anns_file = '/data1/qilei_chen/DATA/erosive/annotations/'+set_name+'.json'
    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs    
    inference_and_save_result(model,coco_instance,"/data1/qilei_chen/DATA/erosive/images",checkpoint_file+".json")

def eval(result_dir,coco_instance,thres = 0.3):
    fp = open(result_dir,'rb')
    results = pickle.load(fp)
    
    