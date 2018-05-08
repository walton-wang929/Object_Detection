# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:00:35 2018

@author: TWang
"""

import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time
import glob

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

CWD = os.getcwd()
sys.path.append(os.path.join(CWD,'object_detection'))

from utils import label_map_util
from utils import visualization_utils as vis_util


MODEL_NAME = r'D:\TF_Try\tensorflow_models\research\oid_argumented\prepareForCloudTraining\Faster_Rcnn_inception\output_200000'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = r'D:\TF_Try\tensorflow_models\research\oid_argumented\oid_person_label_map.pbtxt'

NUM_CLASSES = 29

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

confidence_threshold = 0.3

PATH_TO_TEST_IMAGES_DIR = r'D:\TF_Try\tensorflow_models\research\oid_argumented\data\validation'
#PATH_TO_TEST_IMAGES_DIR = r'D:\TF_Try\gender\data\test\airport\female'  

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        
        ticcc = time.time()
        #for i in range(1,200):
        #for i in range(10):
        images = os.listdir(PATH_TO_TEST_IMAGES_DIR)
#        for img in os.listdir(PATH_TO_TEST_IMAGES_DIR):
        import random    
        for i in range(1000):
            img = random.choice(images)        
            #path = PATH_TO_TEST_IMAGES_DIR + str(i) + '.jpg'
            path = os.path.join(PATH_TO_TEST_IMAGES_DIR,img)
            
            frame = cv2.imread(path)
            #print('test')  
            #h, w, channels = frame.shape[:2]
            h = np.size(frame,0)
            w = np.size(frame,1)
            
            tic = time.time()
            image_np_expanded = np.expand_dims(frame, axis=0)
            
            boxes, scores, classes, num = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                   feed_dict={image_tensor: image_np_expanded})
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            scores = scores[scores > confidence_threshold]
            boxes = boxes[:len(scores)]
            classes = classes[:len(scores)]
            
            for box, score, cls in zip(boxes,scores,classes):
                box = box * np.array([h,w,h,w])
                y1, x1, y2, x2 = box.astype('int')
                class_name = category_index[cls]['name']
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)
                y1 = y1 + 15
                cv2.putText(frame,'%s:%.3f'%(class_name,score*100),(x1,y1),3,1,(0,0,255),2,cv2.LINE_AA)
                
            toc = time.time()
            durr = float(toc - tic)
            fps = 1.0 / durr
            cv2.putText(frame,"fps:%.3f"%fps,(20,20),3, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('detect',frame)
            
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                savepath = r"D:\TF_Try\tensorflow_models\research\oid_argumented\data\demo"
                savename = os.path.join(savepath,img)
                cv2.imwrite(savename,frame)
            elif key == ord('q'):
                break
            else:
                continue
            
        tocccc = time.time()
        print(tocccc-ticcc)
            
cv2.destroyAllWindows()