# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:36:37 2018

@author: TWang

two detection model connected 
one is coco dataset pretrained person detection model
one is body partition model

one classification model : male or female using keras Tensorflow

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
    
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label
  
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
            
            box = box * np.array([h,w,h,w])
            y1, x1, y2, x2 = box.astype('int')
            class_name = self.category_index[cls]['name']

            if class_name == 'person':
                boxes_to_return.append((x1,y1,x2,y2))
                classes_to_return.append(cls)
                scores_to_return.append(score)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)
                y1 = y1 - 15 if y1 > 15 else max(y1 - 15,0)
                cv2.putText(frame,'%s:%.3f'%(class_name,score*100),(x1,y1),4,0.75,(0,0,255),2,cv2.LINE_AA)
                
            elif class_name =='backpack':
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
                y1 = y1 - 15 if y1 > 15 else max(y1 - 15,0)
                cv2.putText(frame,'%s:%.3f'%(class_name,score*100),(x1,y1),4,0.75,(0,0,255),2,cv2.LINE_AA)
                
            elif class_name =='umbrella':
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
                y1 = y1 - 15 if y1 > 15 else max(y1 - 15,0)
                cv2.putText(frame,'%s:%.3f'%(class_name,score*100),(x1,y1),4,0.75,(0,0,255),2,cv2.LINE_AA)  
                
            elif class_name =='handbag':
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
                y1 = y1 - 15 if y1 > 15 else max(y1 - 15,0)
                cv2.putText(frame,'%s:%.3f'%(class_name,score*100),(x1,y1),4,0.75,(0,0,255),2,cv2.LINE_AA)
                
            elif class_name =='suitcase':
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
                y1 = y1 - 15 if y1 > 15 else max(y1 - 15,0)
                cv2.putText(frame,'%s:%.3f'%(class_name,score*100),(x1,y1),4,0.75,(0,0,255),2,cv2.LINE_AA)
                
            elif class_name =='tie':
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
                y1 = y1 - 15 if y1 > 15 else max(y1 - 15,0)
                cv2.putText(frame,'%s:%.3f'%(class_name,score*100),(x1,y1),4,0.75,(0,0,255),2,cv2.LINE_AA)
                    
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

class Atr_Gender():
    def __init__(self,path_to_ckpt,category_index,gpu_option):
        
        self.classify_graph = tf.Graph()
        graph_def = tf.GraphDef()
        
        with open(path_to_ckpt, "rb") as f:
            graph_def.ParseFromString(f.read())
        with self.classify_graph.as_default():
            tf.import_graph_def(graph_def)
            
        self.category_index = category_index
        
        self.sess = tf.Session(graph=self.classify_graph,
                               config=tf.ConfigProto(gpu_options=gpu_option,allow_soft_placement=True))
        
    def read_tensor_from_image_file(image,input_height,input_width,input_mean,input_std):
        
        img = image
        resized = cv2.resize(img,(input_height, input_width))
        float_img= np.asfarray(resized,dtype='float32')
        image_np_expanded = np.expand_dims(float_img, axis=0)
        normalized = np.divide(np.subtract(image_np_expanded,[input_mean]),[input_std])
  
        return normalized
        
    def predict(self,frame):
        
        h, w = frame.shape[:2]

        input_height = 224
        input_width = 224
        input_mean = 128
        input_std = 128
        
        t = Atr_Gender.read_tensor_from_image_file(frame,                                      
                                                   input_height=input_height,
                                                   input_width=input_width,
                                                   input_mean=input_mean,
                                                   input_std=input_std)
        
        input_layer = "input"
        output_layer = "final_result"   
        
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        
        input_operation = self.classify_graph.get_operation_by_name(input_name)
        output_operation = self.classify_graph.get_operation_by_name(output_name)
#==============================================================================
#         print("input",input_operation)
#         print("output",output_operation)
#==============================================================================
        
        results = self.sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
        results = np.squeeze(results)
        
#==============================================================================
#         top_k = results.argsort()[-5:][::-1]
#         labels = self.category_index
#==============================================================================
        
#==============================================================================
#         for i in top_k:
#             print(labels[i], results[i])
#==============================================================================
#==============================================================================
#         if results[0] > 0.5 :
#             gender = 'F'
#         else:
#             gender = 'M'
#         cv2.putText(frame, gender,(5,20),4, 1, (0, 0, 255), 2, cv2.LINE_AA)
#==============================================================================
        cv2.rectangle(frame,(0,0),(w,h),(0,255,0),4)
        cv2.putText(frame,"F:%.5f"%results[0],(5,20),4, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,"M:%.5f"%results[1],(5,40),4, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    
        return frame 
    
    def close(self):
        self.sess.close()
            
MODEL_ACCURATE = 'faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08'
MODEL_MEDIUM = 'faster_rcnn_inception_v2_coco_2017_11_08'
MODEL_FAST = 'ssd_inception_v2_coco_2017_11_17'

People_Det_MODEL = MODEL_MEDIUM
People_Det_MODEL = os.path.join(MODEL_DIR,People_Det_MODEL)
People_Det_CKPT = People_Det_MODEL + '/frozen_inference_graph.pb'
People_Det_LABELS = r'D:\TF_Try\tensorflow_models\research\object_detection\data\mscoco_label_map.pbtxt'

Body_part_Det_MODEL = os.path.join(MODEL_DIR,"body_partition_ssd_inception_v2_coco_2017_11_17")
Body_part_Det_CKPT = os.path.join(Body_part_Det_MODEL,'frozen_inference_graph.pb')
Body_part_Det_LABELS = os.path.join(Body_part_Det_MODEL,'label_map.pbtxt')

Gender_Classify_MODEL = os.path.join(MODEL_DIR,"gender_mobilenet_v1_0.50_224_50000")
Gender_Classify_CKPT = os.path.join(Gender_Classify_MODEL,'output_graph.pb')
Gender_Classify_LABELS = os.path.join(Gender_Classify_MODEL,'label_map.txt')

People_Det_category_index = get_category_index(People_Det_LABELS,90)
Body_part_Det_category_index = get_category_index(Body_part_Det_LABELS,3)
Gender_Classify_category_index = load_labels(Gender_Classify_LABELS)

People_Det_gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
Body_part_Det_gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
Gender_Classify_gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)


model_detect = Det(People_Det_CKPT,People_Det_category_index,People_Det_gpu_option)
model_rcgnz  = Atr(Body_part_Det_CKPT,Body_part_Det_category_index,Body_part_Det_gpu_option)
model_gender = Atr_Gender(Gender_Classify_CKPT,Gender_Classify_category_index,Gender_Classify_gpu_option)


if sys.platform == 'win32':
    video_name = r"D:\Pedestrian Detection\test_video\airport\Entrance_Peak Hour.avi"
else:
    video_name = "/media/shared_disk/LIYUXIN/Videos/street_1.mp4"

cap = cv2.VideoCapture(video_name)

confidence_threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    frame_for_roi = frame.copy()
    frame_gender = frame.copy()
    
    tic = time.time()
#    frame = cv2.resize(frame,dsize=(1080,720))
    processed_frame, boxes,classes,scores = model_detect.predict(frame)
    
    for box in boxes:
        x1,y1,x2,y2 = box

        roi_frame = frame_for_roi[y1:y2,x1:x2]
        pred = model_rcgnz.predict(roi_frame)
        
        gender_frame = frame_gender[y1:y2,x1:x2]
        gender = model_gender.predict(gender_frame)

    toc = time.time()
    durr = float(toc - tic)
    fps = 1.0 / durr
    cv2.putText(processed_frame,"fps:%.2f"%fps,(20,20),4, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('Detect',processed_frame)
    
    cv2.putText(frame_for_roi,"fps:%.2f"%fps,(20,20),4, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('Body_part',frame_for_roi)
    
    cv2.putText(frame_gender,"fps:%.2f"%fps,(20,20),4, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('gender',frame_gender)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

model_detect.close()
model_rcgnz.close()            
cap.release()      
cv2.destroyAllWindows()
