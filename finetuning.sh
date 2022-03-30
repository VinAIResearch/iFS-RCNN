
conda activate fsdet


arch=50 # 50
network=mask_rcnn
num_gpus=8

suffix='' # '_sigmoid_classifier_box_iou_uncertainty': Contribution 2 or '_sigmoid_classifier': our baseline or '': original softmax Mask-RCNN
suffix2='' # '_bayesian' or '' # Contribution 1

dataset=coco
detection_folder=${dataset^^}-detection

# Surgery for Mask-RCNN sigmoid

python3 -m tools.ckpt_surgery --$dataset \
        --src1 checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_base${suffix}/model_final.pth \
        --method remove \
        --save-dir checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_all${suffix}


# finetuning on novel classes
shot=1

python3 -m tools.train_net --num-gpus $num_gpus \
        --config-file configs/COCO-detection/${network}/${network}_R_${arch}_FPN_ft_novel_${shot}shot${suffix}${suffix2}.yaml \
        --opts MODEL.WEIGHTS checkpoints/coco/${network}/${network}_R_${arch}_FPN_all${suffix}/model_reset_remove.pth


# # For test all base + novel classes
python3 -m tools.ckpt_surgery --coco \
        --src1 checkpoints/coco/${network}/${network}_R_${arch}_FPN_base${suffix}/model_final_early.pth \
        --src2 checkpoints/coco/${network}/${network}_R_${arch}_FPN_ft_novel_${shot}shot${suffix}${suffix2}/model_final.pth \
        --method combine \
        --save-dir checkpoints/coco/${network}/${network}_R_${arch}_FPN_all_final_${shot}shot${suffix}${suffix2}


python3 -m tools.test_net --num-gpus $num_gpus \
        --config-file configs/COCO-detection/${network}/${network}_R_${arch}_FPN_test_all_${shot}shot${suffix}${suffix2}.yaml \
        --eval-only \
        --opts MODEL.WEIGHTS checkpoints/coco/${network}/${network}_R_${arch}_FPN_all_final_${shot}shot${suffix}${suffix2}/model_reset_combine.pth





