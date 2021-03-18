from mmdet.apis import init_detector, inference_detector
import mmcv
from pycocotools.coco import COCO
import os
import cv2
import json
import pickle
from metric_polyp import Metric

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

        import datetime
        time0=datetime.datetime.now()
        result = inference_detector(model, img)   
        time1=datetime.datetime.now()
        print("model process")
        print((time1-time0).microseconds/1000)

        results[coco_imgs[key]['id']] = dict()
        results[coco_imgs[key]['id']]['file_name'] = img_file_name
        results[coco_imgs[key]['id']]['result'] = result

    with open(result_save_dir, 'wb') as fp:
        pickle.dump(results, fp)

def generate_result(coco_instance,set_name = 'test'):
    # Specify the path to model config and checkpoint file
    model_name = 'reppoints_moment_r50_fpn_1x_coco'

    config_file = 'configs/erosive/'+model_name+'.py'
    checkpoint_file = '/data1/qilei_chen/DATA/erosive/work_dirs/'+model_name+'/epoch_83.pth'

    # build the model from a config file and a checkpoint file

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

   
    inference_and_save_result(model,coco_instance,"/data1/qilei_chen/DATA/erosive/images",checkpoint_file+"_"+set_name+".pkl")


def xyxy2xywh(box):
    return [box[0],box[1],box[2]-box[0],box[3]-box[1]]

def xyxycenter(box):
    return ((box[0]+box[2])/2,(box[1]+box[3])/2)

def xywh2xyxy(box):
    return [box[0],box[1],box[2]+box[0],box[3]+box[1]]
def xywhcenter(box):
    return (box[0]+box[2]/2,box[1]+box[3]/2)

def center_in_xyxyrule(Point,Bbox):
    if Point[0]>=Bbox[0] and Point[1]>=Bbox[1] and Point[0]<=Bbox[0]+Bbox[2] and Point[1]<=Bbox[1]+Bbox[3]:
        return True
    return False
def center_in_xywhrule(Point,Bbox):
    if Point[0]>=Bbox[0] and Point[1]>=Bbox[1] and Point[0]<=Bbox[2] and Point[1]<=Bbox[3]:
        return True
    return False
def filt_boxes(boxes_with_scores,thres):
    filted_boxes = []
    for box in boxes_with_scores:
        if box[4]>=thres:
            filted_boxes.append(box[0:4])
    return filted_boxes
    
def anns2gtboxes(gtanns):
    gtboxes = []
    for ann in gtanns:
        gtboxes.append(xywh2xyxy(ann['bbox']))
    return gtboxes

def peval(result_dir,coco_instance,thresh = 0.3,with_empty_images=True):
    
    fp = open(result_dir,'rb')
    results = pickle.load(fp)
    eval = Metric()
    
    for img_id in results:
        filed_boxes = filt_boxes(results[img_id]['result'][0],thresh)
        gtannIds = coco_instance.getAnnIds(imgIds= img_id)
        gtanns = coco_instance.loadAnns(gtannIds)  
        gtboxes = anns2gtboxes(gtanns)  
        if len(gtboxes)==0 and (not with_empty_images):
            continue
        eval.eval_add_result(gtboxes,filed_boxes)   
     
    precision, recall = eval.get_result()
    F1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
    F2 = 5 * (precision * recall) / max((4 * precision + recall), 1e-5)
    out = "precision: {:.4f}  recall:  {:.4f} F1: {:.4f} F2: {:.4f} thresh: {:.4f} TP: {:3} FP: {:3} FN: {:3} FP+FN: {:3}" \
        .format(precision, recall, F1, F2, thresh, len(eval.TPs), len(eval.FPs), len(eval.FNs), len(eval.FPs)+len(eval.FNs))
    print (out)

if __name__=="__main__":
    # test images and show the results
    sets = ['train','test']
    set_name = sets[0] #
    anns_file = '/data1/qilei_chen/DATA/erosive/annotations/'+set_name+'.json'
    coco_instance = COCO(anns_file)
    
    generate_result(coco_instance,set_name)
    '''
    model_name = 'reppoints_moment_r50_fpn_1x_coco'
    results_file_dir = '/data1/qilei_chen/DATA/erosive/work_dirs/'+model_name+'/epoch_83.pth'+'_'+set_name+'.pkl'
    peval(results_file_dir,coco_instance,thresh=0.3,with_empty_images=False)
    '''
    