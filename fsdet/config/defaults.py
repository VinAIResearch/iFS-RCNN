from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C

# adding additional default values built on top of the default values in detectron2

_CC = _C

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
_CC.MODEL.ROI_HEADS.COMBINE_TYPE = 'softmax'
_CC.MODEL.ROI_HEADS.LOSS_TYPE = 'CE'

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0
_CC.MODEL.ROI_HEADS.USE_BIAS = True
_CC.MODEL.ROI_HEADS.USE_LABEL_REFINE = False
_CC.MODEL.ROI_HEADS.BASE_WEIGHTS = ""
_CC.MODEL.ROI_HEADS.NEG_RATIO = 10
_CC.MODEL.ROI_HEADS.COSINE_NOVEL_ONLY = False
_CC.MODEL.ROI_HEADS.ADAPT_BASE_WEIGHTS = False
_CC.MODEL.ROI_HEADS.USE_BAYESIAN = False
_CC.MODEL.ROI_MASK_HEAD.NUM_PARTS = 16
_CC.MODEL.ROI_MASK_HEAD.NUM_PLANES = 16
_CC.MODEL.ROI_MASK_HEAD.FREEZE = False
_CC.MODEL.ROI_MASK_HEAD.USE_CAM = False
_CC.MODEL.ROI_MASK_HEAD.USE_SAM = False

_CC.MODEL.ROI_HEADS.OUTPUT_BOX_IOU_LAYER = "FastRCNNOutputLayers"
_CC.MODEL.ROI_HEADS.BOX_IOU_THRES = 0.7
_CC.MODEL.EXTRACT_MODE = False
_CC.MODEL.BOX_IOU_ON = False
_CC.MODEL.FREEZE_ALL = False
_CC.MODEL.FREEZE_ALL_BUT_BOXES = False

# Backward Compatible options.
_CC.MUTE_HEADER = True

# ---------------------------------------------------------------------------- #
# Visualization Options
# ---------------------------------------------------------------------------- #
_CC.VISUALIZATION = CN()
_CC.VISUALIZATION.SHOW = False
_CC.VISUALIZATION.CONF_THRESH = 0.3
_CC.VISUALIZATION.FOLDER = 'vis'

# ---------------------------------------------------------------------------- #
# SOLOv2 Options
# ---------------------------------------------------------------------------- #
_CC.MODEL.SOLOV2 = CN()

_CC.MODEL.SOLOV2.FREEZE_FEAT = False

# Instance hyper-parameters
_CC.MODEL.SOLOV2.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_CC.MODEL.SOLOV2.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
_CC.MODEL.SOLOV2.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_CC.MODEL.SOLOV2.SIGMA = 0.2
# Channel size for the instance head.
_CC.MODEL.SOLOV2.INSTANCE_IN_CHANNELS = 256
_CC.MODEL.SOLOV2.INSTANCE_CHANNELS = 512
# Convolutions to use in the instance head.
_CC.MODEL.SOLOV2.NUM_INSTANCE_CONVS = 4
_CC.MODEL.SOLOV2.USE_DCN_IN_INSTANCE = False
_CC.MODEL.SOLOV2.TYPE_DCN = 'DCN'
_CC.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]
# Number of foreground classes.
_CC.MODEL.SOLOV2.NUM_CLASSES = 80
_CC.MODEL.SOLOV2.NUM_KERNELS = 256
_CC.MODEL.SOLOV2.NORM = "GN"
_CC.MODEL.SOLOV2.USE_COORD_CONV = True
_CC.MODEL.SOLOV2.PRIOR_PROB = 0.01

# Mask hyper-parameters.
# Channel size for the mask tower.
_CC.MODEL.SOLOV2.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_CC.MODEL.SOLOV2.MASK_IN_CHANNELS = 256
_CC.MODEL.SOLOV2.MASK_CHANNELS = 128
_CC.MODEL.SOLOV2.NUM_MASKS = 256

# Test cfg.
_CC.MODEL.SOLOV2.NMS_PRE = 500
_CC.MODEL.SOLOV2.SCORE_THR = 0.1
_CC.MODEL.SOLOV2.UPDATE_THR = 0.05
_CC.MODEL.SOLOV2.MASK_THR = 0.5
_CC.MODEL.SOLOV2.MAX_PER_IMG = 100
# NMS type: matrix OR mask.
_CC.MODEL.SOLOV2.NMS_TYPE = "matrix"
# Matrix NMS kernel type: gaussian OR linear.
_CC.MODEL.SOLOV2.NMS_KERNEL = "gaussian"
_CC.MODEL.SOLOV2.NMS_SIGMA = 2

# Loss cfg.
_CC.MODEL.SOLOV2.LOSS = CN()
_CC.MODEL.SOLOV2.LOSS.FOCAL_USE_SIGMOID = True
_CC.MODEL.SOLOV2.LOSS.FOCAL_ALPHA = 0.25
_CC.MODEL.SOLOV2.LOSS.FOCAL_GAMMA = 2.0
_CC.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT = 1.0
_CC.MODEL.SOLOV2.LOSS.DICE_WEIGHT = 3.0

# New network option 
_CC.MODEL.SOLOV2.USE_COSINE_SIM = ""
_CC.MODEL.SOLOV2.USE_LABEL_REFINE = False
_CC.MODEL.SOLOV2.USE_BAYESIAN = False
_CC.MODEL.SOLOV2.USE_INS_DEV = False
_CC.MODEL.SOLOV2.USE_UNCERTAINTY = False
_CC.MODEL.SOLOV2.USE_BAYESIAN_KERNEL = False

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_CC.MODEL.FCOS = CN()

_CC.MODEL.FCOS.FREEZE_FEAT = False

# This is the number of foreground classes.
_CC.MODEL.FCOS.NUM_CLASSES = 80
_CC.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_CC.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_CC.MODEL.FCOS.PRIOR_PROB = 0.01
_CC.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_CC.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_CC.MODEL.FCOS.NMS_TH = 0.6
_CC.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_CC.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_CC.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_CC.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_CC.MODEL.FCOS.TOP_LEVELS = 2
_CC.MODEL.FCOS.NORM = "GN"  # Support GN or none
_CC.MODEL.FCOS.USE_SCALE = True
_CC.MODEL.FCOS.CLS_AGNOSTIC_BOX_REG = True
_CC.MODEL.FCOS.CLS_AGNOSTIC_CTRNESS = True


# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_CC.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_CC.MODEL.FCOS.LOSS_ALPHA = 0.25
_CC.MODEL.FCOS.LOSS_GAMMA = 2.0
_CC.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_CC.MODEL.FCOS.USE_RELU = True
_CC.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_CC.MODEL.FCOS.NUM_CLS_CONVS = 4
_CC.MODEL.FCOS.NUM_BOX_CONVS = 4
_CC.MODEL.FCOS.NUM_SHARE_CONVS = 0
_CC.MODEL.FCOS.CENTER_SAMPLE = True
_CC.MODEL.FCOS.POS_RADIUS = 1.5
_CC.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'

# ---------------------------------------------------------------------------- #
# CenterMask
# ---------------------------------------------------------------------------- #
_CC.MODEL.ROI_MASK_HEAD.ASSIGN_CRITERION = "area"
_CC.MODEL.MASKIOU_ON = False
_CC.MODEL.MASKIOU_LOSS_WEIGHT = 1.0

_CC.MODEL.ROI_MASKIOU_HEAD = CN()
_CC.MODEL.ROI_MASKIOU_HEAD.NAME = "MaskIoUHead"
_CC.MODEL.ROI_MASKIOU_HEAD.CONV_DIM = 256
_CC.MODEL.ROI_MASKIOU_HEAD.NUM_CONV = 4

# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_CC.MODEL.MOBILENET = False
