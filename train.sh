conda activate fsdet


arch=50 
network=mask_rcnn 
num_gpus=8

dataset=coco
detection_folder=${dataset^^}-detection


# standard softmax Mask-RCNN
suffix=''
python3 -m tools.train_net --num-gpus $num_gpus --config-file configs/${detection_folder}/${network}/${network}_R_${arch}_FPN_base${suffix}.yaml


# # sigmoid Mask-RCNN
# suffix='_sigmoid_classifier'
# python3 -m tools.train_net --num-gpus $num_gpus --config-file configs/${detection_folder}/${network}/${network}_R_${arch}_FPN_base${suffix}.yaml


# # contribution 2
# suffix='_sigmoid_classifier'
# python3 -m tools.ckpt_surgery_box_predictor_only --${dataset} \
#         --src1 checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_base${suffix}/model_final.pth \
#         --method remove \
#         --save-dir checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_all${suffix} --tar-name model_freeze_all_but_boxes

# # # num_gpus=1 # to ensure that we all have proposal >= 0.7 iou threshold

# suffix_new='_sigmoid_classifier_box_iou_uncertainty'
# python3 -m tools.train_net --num-gpus $num_gpus --config-file configs/${detection_folder}/${network}/${network}_R_${arch}_FPN_base${suffix_new}.yaml \
# --opts MODEL.WEIGHTS checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_all${suffix}/model_reset_all_remove.pth
