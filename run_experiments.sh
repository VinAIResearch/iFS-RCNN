
conda activate fsdet

arch=50 # 50, 101
network=mask_rcnn # mask_rcnn, faster_rcnn
num_gpus=8

suffix='_sigmoid_classifier_box_iou_uncertainty' # '_sigmoid_classifier_box_iou_uncertainty': Contribution 2 or '_sigmoid_classifier': our baseline or '': original softmax Mask-RCNN
suffix2='_bayesian' # '_bayesian' or '' # Contribution 1


##### train and evaluate on novel classes only for incremental setting
weights_file=checkpoints/coco/${network}/${network}_R_${arch}_FPN_all${suffix}/model_reset_remove.pth
base_config=configs/COCO-detection/${network}/${network}_R_${arch}_FPN_ft_novel_1shot${suffix}${suffix2}.yaml


##### evaluate on base and novel classes
base_config=configs/COCO-detection/${network}/${network}_R_${arch}_FPN_test_all_1shot${suffix}${suffix2}.yaml 
weights_file=checkpoints/coco/${network}/${network}_R_${arch}_FPN_all${suffix}/model_reset_remove.pth


python3 -m tools.run_experiments --num-gpus $num_gpus --network $network --arch $arch \
        --shots 1 2 3 5 10 30 --seeds 0 10 --coco --base_config $base_config --weights_file $weights_file --suffix $suffix --suffix2 $suffix2

