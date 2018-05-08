# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:03:01 2018

@author: twang

record for demo video using detectron

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals #uncideo string literals


import sys
import os
sys.path.append(os.path.join('/home/twang/Documents/tensorflow-models/research','slim'))
sys.path.append(os.path.join('/home/twang/Documents/tensorflow-models/research','object_detection'))
sys.path.append('/home/twang/Documents/tensorflow-models/research')
sys.path.append('/home/twang/Documents/detectron/lib')


from collections import defaultdict
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import time
import numpy as np
import argparse

from caffe2.python import workspace
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils

import pycocotools.mask as mask_util
from utils.colormap import colormap

#import utils.vis as vis_utils


import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

from object_detection.utils import label_map_util

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


'''test each function running speed performance'''

if sys.platform == 'win32':
    sys.path.append(os.path.join(os.getcwd(),'object_detection'))
    PATH_TO_LABELS = os.getcwd()
    MODEL_DIR = r"D:\Pedestrian Detection\models\tensorflow"
else:
    PATH_TO_LABELS = r"/media/network_shared_disk/WangTao/Person_attributes_detection/COCOPersonDetection+RAPBodyPartition+RAPGender/label_map"
    MODEL_DIR = r"/media/network_shared_disk/WangTao/Person_attributes_detection/COCOPersonDetection+RAPBodyPartition+RAPGender/tensorlow_models"


def get_category_index(path_to_label,num_class):
    
    label_map = label_map_util.load_labelmap(path_to_label)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_class, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    return category_index
    
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label
  
  
'''Person gender classification'''
class Person_Gender():
    
    def __init__(self,checkpoint_path,category_index,gpu_option):
        model_name_to_variables = {'inception_v3':'InceptionV3','inception_v4':'InceptionV4','resnet_v1_50':'resnet_v1_50','resnet_v1_152':'resnet_v1_152'}
        preprocessing_name = 'gender_vgg16'
        eval_image_size = None
        model_name = 'vgg_16'
        num_classes = 2
        
        model_variables = model_name_to_variables.get(model_name)

        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        else:
            checkpoint_path = checkpoint_path
            
        # Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()
        self.image_string = tf.placeholder(tf.string) 

        #image = tf.image.decode_image(image_string, channels=3)
        image = tf.image.decode_jpeg(self.image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.9) ## To process corrupted image files

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)

        network_fn = nets_factory.get_network_fn(model_name, num_classes, is_training=False)

        if eval_image_size is None:
            eval_image_size = network_fn.default_image_size

        processed_image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        processed_images  = tf.expand_dims(processed_image, 0) # Or tf.reshape(processed_image, (1, eval_image_size, eval_image_size, 3))

        logits, _ = network_fn(processed_images)

        self.probabilities = tf.nn.softmax(logits)  
        
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option,allow_soft_placement=True))
        
        init_fn(self.sess)
        
        
    def predict(self,frame):
        
        
        frame_encoded = tf.gfile.FastGFile(frame,'rb').read() # You can also use x = open(fl).read()
        
        #frame_encoded = tf.image.encode_jpeg(frame,format="rgb")
        #print("frame_encoded",type(frame_encoded))
        
        probs = self.sess.run(self.probabilities, feed_dict={self.image_string:frame_encoded})
        #np_image, network_input, probs = sess.run([image, processed_image, probabilities], feed_dict={image_string:x})
 
        probs = probs[0, 0:]
        
        #print("probs",probs)
        return probs
    
    def close(self):
        self.sess.close()  
        

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--video_dir',dest='video_dir',help='image or folder of images', default=''
    )
    
    return parser.parse_args()

def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes
    
def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def vis_class(img, pos, class_str, font_scale=0.35):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, thick=1):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img
    
def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0'), class_text

        
def main(args):
    
    #just show box and mask
    cfg_file = r'/home/twang/Documents/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'
    weights_file = r'/home/twang/Documents/detectron/model-weights/mask_rcnn_R-101-FPN_2x_model_final.pkl'
    
#==============================================================================
#     #show keypoint
#     cfg_file = r'/home/twang/Documents/detectron/model-weights/e2e_KeyPoint_Mask_RCNN/e2e_keypoint_rcnn_X-101-64x4d-FPN_1x.yaml'
#     weights_file = r'/home/twang/Documents/detectron/model-weights/e2e_KeyPoint_Mask_RCNN/X-101-64x4d-FPN_1x.pkl'    
#     
#==============================================================================
    merge_cfg_from_file(cfg_file)
    cfg.NUM_GPUS =2
    weights = cache_url(weights_file, cfg.DOWNLOAD_CACHE)
    
    assert_and_infer_cfg()
    
    model = infer_engine.initialize_model_from_cfg(weights)
    
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    
    Person_Gender_CKPT = '/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/airport_gender3/training_all_layers/vgg_16_750000'
    Person_Gender_LABELS = os.path.join('/home/twang/Documents/tensorflow-classfication2/gender-airport-entrance','labels.txt')
    Person_Gender_category_index = load_labels(Person_Gender_LABELS)

    Person_Gender_gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)


    Gender = Person_Gender(Person_Gender_CKPT, Person_Gender_category_index, Person_Gender_gpu_option)
    
    video_dir = args.video_dir 
    
    cap = cv2.VideoCapture(video_dir)
    
    
    video_box_mask = cv2.VideoWriter('out_box_mask.mp4',cv2.VideoWriter_fourcc('M','J','P','G'),5,(1280,720))
    
    while cap.isOpened():
        
        t1 = time.time()
        
        ret, frame = cap.read()

        if not ret:
            break
        
        frame = cv2.resize(frame,dsize=(1280,720))   
        
        timers = defaultdict(Timer)

        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(model, frame, None, timers=timers) 
        
        thresh=0.7
        
        show_box = True
        show_class = True
        crop_person = True
        filter_big_box =True
        
        dataset=dummy_coco_dataset
        
        frame_for_person_crop = frame.copy()
        frame_for_box = frame.copy()
        frame_for_mask = frame.copy()
        frame_for_both = frame.copy()
        
        """Constructs a numpy array with the detections visualized."""
        if isinstance(cls_boxes, list):
            boxes, segms, keypoints, classes = convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)

        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
            return frame

        if segms is not None and len(segms) > 0:
            masks = mask_util.decode(segms)
            color_list = colormap() 
            color_list_selected = color_list[0:5]
            
        else:
            color_list = colormap() 
            color_list_selected = color_list[0:5]            
            
        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        
        for i in sorted_inds:
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            if score < thresh:
                continue   
                
            # show class (person, backpack, handbag, suitcase)
            class_default = ['person','backpack','handbag','suitcase']
            if show_class:
                class_str, class_text = get_class_string(classes[i], score, dataset)
                
                if class_text in class_default:
                    
                    frame = vis_class(frame, (bbox[0], bbox[1] - 2), class_str)
                    
                    frame_for_both = vis_class(frame_for_both, (bbox[0], bbox[1] - 2), class_str)
                    
                    #filter big box
                    if filter_big_box:
                        
                        aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
                        
                        if aspect_ratio < 1.5 : 
                    
                            #show bounding box
                            if show_box:
                                frame = vis_bbox(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                                
                                frame_for_box =  vis_bbox(frame_for_box, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                                frame_for_both = vis_bbox(frame_for_both, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                                
                            # crop each person box, recognize gender
                            if crop_person and class_text =='person':
                                
                                (x1, y1, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))
                                x2 = x1 +w
                                y2 = y1 +h
                            
                                cropped = frame_for_person_crop[y1:y2, x1:x2]
                                
                                cv2.imwrite('/home/twang/Documents/detectron/temp/Hk-demo/genderwrite.jpg',cropped)
                                
                                gender_frame_saved='/home/twang/Documents/detectron/temp/Hk-demo/genderwrite.jpg'
                                prob = Gender.predict(gender_frame_saved)
                                
                                if prob[0] > prob[1] :
                                    
                                    #show class name and probality
                                    frame_for_both = vis_class(frame_for_both, (x1,y1+10), "Female:%.1f"%(prob[0]))

                                    #show female mask
                                    color_mask_female = color_list_selected[0, 0:3]
                                    frame_for_both = vis_mask(frame_for_both, masks[..., i], color_mask_female)
                                    
                                    
                                    frame_for_box = vis_class(frame_for_box, (x1,y1+10), "Female:%.1f"%(prob[0])) 
                                    frame_for_box = vis_mask(frame_for_box, masks[..., i], color_mask_female)
                                    
                                else:
                                    
                                    #show class name and probality
                                    frame_for_both = vis_class(frame_for_both, (x1,y1+10), "Male:%.1f"%(prob[1]))
                                    
                                    #show male mask
                                    color_mask_male = color_list_selected[1, 0:3]
                                    frame_for_both = vis_mask(frame_for_both, masks[..., i], color_mask_male)
                                    
                                    
                                    frame_for_box = vis_class(frame_for_box, (x1,y1+10), "Male:%.1f"%(prob[1]))
                                    frame_for_box = vis_mask(frame_for_box, masks[..., i], color_mask_male)
                             
                             
                            #show mask, different other category has deifferent color
                            if segms is not None and len(segms) > i:
                                if class_text == 'backpack':
                                    color_mask = color_list_selected[2, 0:3]
                                    frame_for_both = vis_mask(frame_for_both, masks[..., i], color_mask)     
                                elif class_text == 'handbag':
                                    color_mask = color_list_selected[3, 0:3]
                                    frame_for_both = vis_mask(frame_for_both, masks[..., i], color_mask)  
                                elif class_text == 'suitcase':
                                    color_mask = color_list_selected[4, 0:3]
                                    frame_for_both = vis_mask(frame_for_both, masks[..., i], color_mask)  
                                    
                                
                    else:
                            #show bounding box
                            if show_box:
                                frame = vis_bbox(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                                
                            # crop each box 
                            if crop_person and class_text =='person':
                                
                                (x1, y1, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))
                                x2 = x1 +w
                                y2 = y1 +h
                            
                                cropped = frame_for_person_crop[y1:y2, x1:x2]
                                
                                cv2.imwrite('/home/twang/Documents/detectron/temp/Hk-demo/genderwrite.jpg',cropped)
                                
                                gender_frame_saved='/home/twang/Documents/detectron/temp/Hk-demo/genderwrite.jpg'
                                prob = Gender.predict(gender_frame_saved)
                                
                                if prob[0] > prob[1] :
                                    
                                    cv2.putText(frame,"Female:%.1f"%(prob[0]*100),(x1,y1+10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, _GRAY, lineType=cv2.LINE_AA)
            
                                else:
                                    cv2.putText(frame,"Male:%.1f"%(prob[1]*100),(x1,y1+10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, _GRAY, lineType=cv2.LINE_AA)
                                
                                

        t2 = time.time()
        durr = float(t2-t1)
        fps = 1.0 / durr
        #cv2.putText(frame,"fps:%.3f"%fps,(20,20),4, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Detection using and box mask',frame_for_both)
        
        video_box_mask.write(frame_for_both)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
    cap.release() 
    video_box_mask.release()
    cv2.destroyAllWindows()            

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parse_args()
    main(args)
