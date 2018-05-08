# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:59:16 2018

@author: twang

Perform inference on a video."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='/home/twang/Documents/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='/home/twang/Documents/detectron/model-weights/R-50-C4-model_final.pkl',
        type=str
    )
    parser.add_argument(
        '--video_name', help='image or folder of images', default='/media/network_shared_disk/WangTao/test_video/HK_airport/IMG_0684.MOV'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()



def main(args):
    
    logger = logging.getLogger(__name__)
    
    merge_cfg_from_file(args.cfg)
    
    cfg.NUM_GPUS = 2
    
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    
    assert_and_infer_cfg()
    
    model = infer_engine.initialize_model_from_cfg(args.weights)
    
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    

    if os.path.isdir(args.video_name):
        print("video_name",args.video_name)
    else:
        print("video is not existing")

    cap = cv2.VideoCapture(args.video_name)
    
    while cap.isOpened():
        
        ret, frame = cap.read()

        if not ret:
            break
        
        frame = cv2.resize(frame,dsize=(1280,720))   
        
        timers = defaultdict(Timer)
        t1 = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(model, frame, None, timers=timers)
        logger.info('Inference time: {:.3f}s'.format(time.time() - t1))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        frame = vis_utils.vis_one_image_opencv(frame, 
                                       cls_boxes, 
                                       segms=cls_segms, 
                                       keypoints=cls_keyps, 
                                       thresh=0.8, 
                                       kp_thresh=2,
                                       show_box=False, 
                                       dataset=dummy_coco_dataset, 
                                       show_class=False)
        t2 = time.time()
        durr = float(t2-t1)
        fps = 1.0 / durr
        cv2.putText(frame,"fps:%.3f"%fps,(20,20),4, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Detection', frame)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    cap.release()    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)