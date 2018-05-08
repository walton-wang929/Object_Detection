#!/usr/bin/env python2

'''
Perform Key point Detection inference on a video.
Allows for using a combination of multiple models. 

1st: one model may be used for RPN, 
another model for Fast R-CNN style box detection, 
yet another model to predict masks, 
and yet another model to predict keypoints.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import sys
import yaml
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
import core.rpn_generator as rpn_engine
import core.test_engine as model_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference on video')
    parser.add_argument(
        '--video', 
        dest='video_name', 
        help='input video', 
        default=None, 
        type=str
    )
    parser.add_argument(
        '--rpn-pkl',
        dest='rpn_pkl',
        help='rpn model file (pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--rpn-cfg',
        dest='rpn_cfg',
        help='cfg model file (yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--models_to_run',
        help='list of pkl, yaml pairs',
        default=None,
        nargs=argparse.REMAINDER
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

    
def get_rpn_box_proposals(im, args):
    cfg.immutable(False)
    
    """Load a yaml config file and merge it into the global config."""
    merge_cfg_from_file(args.rpn_cfg)
    
    '''Number of GPUs to use (applies to both training and testing)'''
    cfg.NUM_GPUS = 1
    
    '''Indicates the model's computation terminates with the production of RPN  proposals (i.e., it outputs proposals ONLY, no actual object detections)'''
    cfg.MODEL.RPN_ONLY = True
    
    '''Number of top scoring RPN proposals to keep before applying NMS When FPN is used, this is *per FPN level* (not total)'''
    cfg.TEST.RPN_PRE_NMS_TOP_N = 10000
    
    '''Number of top scoring RPN proposals to keep after applying NMS his is the total number of RPN proposals produced (for both FPN and non-FPN cases)'''
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000
    
    '''Call this function in your script after you have finished setting all cfg values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.)'''
    assert_and_infer_cfg()

    """Initialize a model from the global cfg. Loads test-time weights and creates the networks in the Caffe2 workspace. """
    model = model_engine.initialize_model_from_cfg(args.rpn_pkl)
    
    with c2_utils.NamedCudaScope(0):
        """Generate RPN proposals on a single image."""
        boxes, scores = rpn_engine.im_proposals(model, im)
    return boxes, scores
    
    
def main(args):
    
    """A dummy COCO dataset that includes only the 'classes' field."""
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    
    ''''load initial Detectron config system'''
    cfg_orig = yaml.load(yaml.dump(cfg))
    
    print("video is :",args.video_name)
    cap = cv2.VideoCapture(args.video_name)
            
    while cap.isOpened():
        
        ret, frame = cap.read()

        if not ret:
            break
        
        t1 = time.time()
        frame = cv2.resize(frame,dsize=(1280,720))  

        if args.rpn_pkl is not None:
            
            proposal_boxes, _proposal_scores = get_rpn_box_proposals(frame, args)
            
            workspace.ResetWorkspace()
            
        else:
            proposal_boxes = None

        cls_boxes, cls_segms, cls_keyps = None, None, None
        
        for i in range(0, len(args.models_to_run), 2):
            pkl = args.models_to_run[i]
            yml = args.models_to_run[i + 1]
            
            cfg.immutable(False)
            
            '''load initial global Detectron config system'''
            merge_cfg_from_cfg(cfg_orig)
            
            """Load a yaml config file and merge it into the global config."""
            merge_cfg_from_file(yml)
            
            if len(pkl) > 0:
                weights_file = pkl
            else:
                weights_file = cfg.TEST.WEIGHTS
                
            '''Number of GPUs to use'''    
            cfg.NUM_GPUS = 1
            
            assert_and_infer_cfg()
            
            '''Initialize a model from the global cfg.'''
            model = model_engine.initialize_model_from_cfg(weights_file)
            
            with c2_utils.NamedCudaScope(0):
                '''Inference detecting all'''
                cls_boxes_, cls_segms_, cls_keyps_ = model_engine.im_detect_all(model, frame, proposal_boxes)
                
            cls_boxes = cls_boxes_ if cls_boxes_ is not None else cls_boxes
            cls_segms = cls_segms_ if cls_segms_ is not None else cls_segms
            cls_keyps = cls_keyps_ if cls_keyps_ is not None else cls_keyps
            workspace.ResetWorkspace()
            
        """Constructs a numpy array with the detections visualized."""    
        frame = vis_utils.vis_one_image_opencv(
                               frame, 
                               cls_boxes, 
                               segms=cls_segms, 
                               keypoints=cls_keyps, 
                               thresh=0.8, 
                               kp_thresh=2,
                               show_box=True, 
                               dataset=dummy_coco_dataset, 
                               show_class=True)
                               
                               
        t2 = time.time()
        durr = float(t2-t1)
        fps = 1.0 / durr
        cv2.putText(frame,"fps:%.3f"%fps,(20,20),4, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Detection', frame)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    cap.release()    
    cv2.destroyAllWindows()
                               
    
def check_args(args):
    '''determine if is None'''
    assert (
        (args.rpn_pkl is not None and args.rpn_cfg is not None) or
        (args.rpn_pkl is None and args.rpn_cfg is None)
    )
    
    if args.rpn_pkl is not None:
        '''Download the file specified by the URL to the cache_dir and return the path to the cached file. 
        If the argument is not a URL, simply return it as is.'''
        args.rpn_pkl = cache_url(args.rpn_pkl, cfg.DOWNLOAD_CACHE)
        assert os.path.exists(args.rpn_pkl)
        assert os.path.exists(args.rpn_cfg)
    
        
    if args.models_to_run is not None:
        '''determine if length ==2'''
        assert len(args.models_to_run) % 2 == 0
        
        for i, model_file in enumerate(args.models_to_run):
            if len(model_file) > 0:
                if i % 2 == 0:
                    '''download 2nd model file'''
                    model_file = cache_url(model_file, cfg.DOWNLOAD_CACHE)
                    args.models_to_run[i] = model_file
                assert os.path.exists(model_file), '\'{}\' does not exist'.format(model_file)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    check_args(args)
    main(args)
