
conda activate fsdet

arch=50
network=mask_rcnn
num_gpus=8

dataset=coco
detection_folder=${dataset^^}-detection

suffix='' # '_sigmoid_classifier_box_iou_uncertainty': Contribution 2 or '_sigmoid_classifier': our baseline or '': original softmax Mask-RCNN
suffix2='' # '_bayesian' or '' # Contribution 1

suffix=${suffix}${suffix2}

for shot in 1 2 3 5 10 30
do 
        base_folder=checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_ft_novel_${shot}shot${suffix} # for novel classes only

        # base_folder=checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_test_all_${shot}shot${suffix} # for both novel and base classes

        # base_folder=checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_ft_all_${shot}shot${suffix} # for both novel and base classes for softmax-Mask-RCNN

        python3 -m tools.aggregate_seeds --shots $shot --seeds 10 --network $network --arch $arch \
                --$dataset --base_folder $base_folder # --suffix $suffix

done
