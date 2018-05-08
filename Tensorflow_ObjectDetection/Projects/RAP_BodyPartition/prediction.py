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

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

CWD = os.getcwd()
sys.path.append(os.path.join(CWD,'object_detection'))

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = r'D:\TF_Try\tensorflow_models\research\object_detection\My_Project\BodyPartition\training\output_inference_graph.pb'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = r'D:\TF_Try\tensorflow_models\research\object_detection\My_Project\BodyPartition\training\graph.pbtxt'

NUM_CLASSES = 3

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

confidence_threshold = 0.5

PATH_TO_TEST_IMAGES_DIR = 'D:\TF_Try\tensorflow_models\research\object_detection\My_Project\BodyPartition\test'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'img_{}.png'.format(i)) for i in range(41520, 41585) ]
            
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        
        for image in TEST_IMAGE_PATHS:
            
            frame = cv2.imread(image)
            
            h, w = frame.shape[:2]
        
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
                y1 = y1 - 15 if y1 > 15 else max(y1 - 15,0)
                cv2.putText(frame,'%s:%.3f'%(class_name,score*100),(x1,y1),3,0.5,(0,0,255),2,cv2.LINE_AA)
                
            toc = time.time()
            durr = float(toc - tic)
            fps = 1.0 / durr
            cv2.putText(frame,"fps:%.3f"%fps,(20,20),3, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('detect',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
cv2.destroyAllWindows()