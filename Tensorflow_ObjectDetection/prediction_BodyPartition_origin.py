# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:46:42 2018

@author: TWang

two detection model connected 


Models available:
    MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08'
    MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2017_11_08'
    MODEL_NAME = 'faster_rcnn_inception_v2_coco_2017_11_08'
    MODEL_NAME = 'faster_rcnn_nas_coco_2017_11_08'
    MODEL_NAME = 'faster_rcnn_nas_lowproposals_coco_2017_11_08'
    MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08'
    MODEL_NAME = 'faster_rcnn_resnet101_lowproposals_coco_2017_11_08'
    MODEL_NAME = 'faster_rcnn_resnet50_lowproposals_coco_2017_11_08'
    MODEL_NAME = 'rfcn_resnet101_coco_2017_11_08'
    MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17' 


"""

import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time

from utils import label_map_util

if sys.platform == 'win32':
    sys.path.append(os.path.join(os.getcwd(),'object_detection'))
    PATH_TO_LABELS = os.getcwd()
    MODEL_DIR = r"D:\Pedestrian Detection\models\tensorflow"
else:
    PATH_TO_LABELS = "/media/shared_disk/LIYUXIN/Library/models/research"
    MODEL_DIR = "/media/shared_disk/LIYUXIN/Models/tensorflow"

def get_category_index(path_to_label,num_class):
    
    label_map = label_map_util.load_labelmap(path_to_label)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_class, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    return category_index

class Det():
    def __init__(self,path_to_ckpt,category_index,gpu_option):
        self.detection_graph = tf.Graph()
        self.sess = tf.Session(graph=self.detection_graph,
                               config=tf.ConfigProto(gpu_options=gpu_option,allow_soft_placement=True))
        
        tf.reset_default_graph()
        with self.detection_graph.as_default():
            detect_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                detect_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(detect_graph_def, name='')
                    
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.category_index = category_index
        
    def predict(self,frame):
    
        h, w = frame.shape[:2]
        
        image_np_expanded = np.expand_dims(frame, axis=0)
        
        boxes, scores, classes, num = self.sess.run([self.detection_boxes,
                                                     self.detection_scores,
                                                     self.detection_classes,
                                                     self.num_detections],
                                                    feed_dict={self.image_tensor: image_np_expanded})
        
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        scores = scores[scores > confidence_threshold]
        boxes = boxes[:len(scores)]
        classes = classes[:len(scores)]
        
        boxes_to_return = []
        classes_to_return = []
        scores_to_return = []
        for box, score, cls in zip(boxes,scores,classes):
            class_name = self.category_index[cls]['name']
            
            if class_name == 'person':
                box = box * np.array([h,w,h,w])
                y1, x1, y2, x2 = box.astype(np.int32)
                boxes_to_return.append((x1,y1,x2,y2))
                classes_to_return.append(cls)
                scores_to_return.append(score)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)
                y1 = y1 - 20 if y1 > 20 else max(y1 - 20,0)
                cv2.putText(frame,'%s:%.2f%%'%(class_name,score*100),(x1,y1),4,0.75,(0,0,255),1,cv2.LINE_AA)
                    
        return frame,boxes_to_return,classes_to_return,scores_to_return
    
    def close(self):
        self.sess.close()
        
class Atr():
    def __init__(self,path_to_ckpt,category_index,gpu_option):
        self.detection_graph = tf.Graph()
        self.sess = tf.Session(graph=self.detection_graph,
                               config=tf.ConfigProto(gpu_options=gpu_option,allow_soft_placement=True))
        
        tf.reset_default_graph()
        with self.detection_graph.as_default():
            detect_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                detect_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(detect_graph_def, name='')
                    
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.category_index = category_index
        
    def predict(self,frame):
    
        h, w = frame.shape[:2]
        
        image_np_expanded = np.expand_dims(frame, axis=0)
        
        boxes, scores, classes, num = self.sess.run([self.detection_boxes,
                                                     self.detection_scores,
                                                     self.detection_classes,
                                                     self.num_detections],
                                                    feed_dict={self.image_tensor: image_np_expanded})
        
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        scores = scores[scores > confidence_threshold]
        boxes = boxes[:len(scores)]
        classes = classes[:len(scores)]
        
        boxes_to_return = []
        for box, score, cls in zip(boxes,scores,classes):
            box = box * np.array([h,w,h,w])
            y1, x1, y2, x2 = box.astype(np.int32)
            boxes_to_return.append((x1,y1,x2,y2))
            class_name = self.category_index[cls]['name']
            if class_name =='HeadShoulder':
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
                y1 = y1 - 20 
                cv2.putText(frame,'%s:%.2f%%'%(class_name,score*100),(x1,y1),4,0.75,(255,0,0),1,cv2.LINE_AA)
                
            elif class_name == 'UpperBody':
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)
                y1 = y1 - 20 
                cv2.putText(frame,'%s:%.2f%%'%(class_name,score*100),(x1,y1),4,0.75,(0,255,0),1,cv2.LINE_AA) 
                
            elif class_name == 'LowerBody':
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),4)
                y1 = y1 - 20 
                cv2.putText(frame,'%s:%.2f%%'%(class_name,score*100),(x1,y1),4,0.75,(0,0,255),1,cv2.LINE_AA)              
                    
        return frame,boxes_to_return,classes,scores
    
    def close(self):
        self.sess.close()
        
MODEL_ACCURATE = 'faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08'
MODEL_MEDIUM = 'faster_rcnn_inception_v2_coco_2017_11_08'
MODEL_FAST = 'ssd_inception_v2_coco_2017_11_17'

MODEL_NAME = MODEL_MEDIUM
MODEL_NAME = os.path.join(MODEL_DIR,MODEL_NAME)

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = r'D:\TF_Try\tensorflow_models\research\object_detection\data\mscoco_label_map.pbtxt'

ATR_DIR = os.path.join(MODEL_DIR,"ssd_inception_v2_coco_2017_11_17_for_attributes_detection")

det_model_path = PATH_TO_CKPT
atr_model_path = os.path.join(ATR_DIR,'frozen_inference_graph.pb')

path_to_det_label = PATH_TO_LABELS
path_to_atr_label = os.path.join(ATR_DIR,'label_map.pbtxt')

det_category_index = get_category_index(path_to_det_label,90)
atr_category_index = get_category_index(path_to_atr_label,3)

det_gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
atr_gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

model_detect = Det(det_model_path,det_category_index,det_gpu_option)
model_rcgnz  = Atr(atr_model_path,atr_category_index,atr_gpu_option)

if sys.platform == 'win32':
    video_name = r"D:\Pedestrian Detection\test_video\street_1.mp4"
else:
    video_name = "/media/shared_disk/LIYUXIN/Videos/street_1.mp4"

cap = cv2.VideoCapture(video_name)

confidence_threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    frame_for_roi = frame.copy()
    tic = time.time()
#    frame = cv2.resize(frame,dsize=(1080,720))
    processed_frame, boxes,classes,scores = model_detect.predict(frame)
    
    preds = []
    for box in boxes:
        x1,y1,x2,y2 = box
        print(box)
        roi_frame = frame_for_roi[y1:y2,x1:x2]
        pred = model_rcgnz.predict(roi_frame)
        print(pred)
        preds.append(pred)
    
    toc = time.time()
    durr = float(toc - tic)
    fps = 1.0 / durr
    cv2.putText(processed_frame,"fps:%.2f"%fps,(20,20),4, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('detect',processed_frame)
    cv2.putText(frame_for_roi,"fps:%.2f"%fps,(20,20),4, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('rois',frame_for_roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

model_detect.close()
model_rcgnz.close()            
cap.release()      
cv2.destroyAllWindows()