
conda activate fsdet

dataset=coco
detection_folder=${dataset^^}-detection

arch=50 # 50
network=mask_rcnn
num_gpus=1
suffix='_sigmoid_classifier_box_iou_uncertainty' # '_sigmoid_classifier_box_iou_uncertainty' or '_sigmoid_classifier' or ''
suffix2='_bayesian' # '_bayesian' or ''

# #### evaluate the base classes
python3 -m tools.test_net --num-gpus $num_gpus \
        --config-file configs/${detection_folder}/${network}/${network}_R_${arch}_FPN_base${suffix}.yaml \
        --eval-only \
        --opts MODEL.WEIGHTS checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_base${suffix}/model_final.pth #VISUALIZATION.SHOW True VISUALIZATION.FOLDER vis


#### evaluate the novel classes (finetuning)
shot=1
python3 -m tools.test_net --num-gpus $num_gpus \
        --config-file configs/COCO-detection/${network}/${network}_R_${arch}_FPN_ft_novel_${shot}shot${suffix}${suffix2}.yaml \
        --eval-only \
        --opts MODEL.WEIGHTS checkpoints/coco/${network}/${network}_R_${arch}_FPN_ft_novel_${shot}shot${suffix}${suffix2}/model_final_official.pth VISUALIZATION.SHOW True VISUALIZATION.FOLDER vis_uncertainty2 VISUALIZATION.CONF_THRESH 0.15


#### evluate on both novel and base classes
python3 -m tools.test_net --num-gpus $num_gpus \
        --config-file configs/${detection_folder}/${network}/${network}_R_${arch}_FPN_test_all_${shot}shot${suffix}${suffix2}.yaml \
        --eval-only \
        --opts MODEL.WEIGHTS checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_all_final_${shot}shot${suffix}${suffix2}/model_reset_combine.pth
