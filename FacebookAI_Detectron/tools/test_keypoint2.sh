python video_test_Mask_RCNN.py \
    --cfg /home/twang/Documents/detectron/model-weights/e2e_KeyPoint_Mask_RCNN/e2e_keypoint_rcnn_X-101-64x4d-FPN_1x.yaml \
    --wts /home/twang/Documents/detectron/model-weights/e2e_KeyPoint_Mask_RCNN/X-101-64x4d-FPN_1x.pkl \
    --video_name /home/twang/Documents/detectron/tools/out_box_mask.mp4


:<<EOF
python video_test_Mask_RCNN.py \
    --cfg /home/twang/Documents/detectron/model-weights/e2e_KeyPoint_Mask_RCNN/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml \
    --wts /home/twang/Documents/detectron/model-weights/e2e_KeyPoint_Mask_RCNN/R_101_FPN_s1x.pkl \
    --video_name /media/network_shared_disk/WangTao/test_video/KLA_airport/Entrance_Peak_Hour.avi

python video_test_Mask_RCNN.py \
    --cfg /home/twang/Documents/detectron/model-weights/e2e_KeyPoint_Mask_RCNN/e2e_keypoint_rcnn_R-50-FPN_1x.yaml \
    --wts /home/twang/Documents/detectron/model-weights/e2e_KeyPoint_Mask_RCNN/R-50-FPN_1x.pkl \
    --video_name /media/network_shared_disk/WangTao/test_video/KLA_airport/Entrance_Peak_Hour.avi

python video_test_Mask_RCNN.py \
    --cfg /home/twang/Documents/detectron/model-weights/e2e_KeyPoint_Mask_RCNN/e2e_keypoint_rcnn_X-101-64x4d-FPN_1x.yaml \
    --wts /home/twang/Documents/detectron/model-weights/e2e_KeyPoint_Mask_RCNN/X-101-64x4d-FPN_1x.pkl \
    --video_name /media/network_shared_disk/WangTao/test_video/KLA_airport/Entrance_Peak_Hour.avi

EOF
