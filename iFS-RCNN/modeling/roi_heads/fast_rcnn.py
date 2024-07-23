"""Implement the CosineSimOutputLayers and  FastRCNNOutputLayers with FC layers."""

import logging
import math

import numpy as np
import torch
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.structures.boxes import matched_boxlist_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F


ROI_HEADS_OUTPUT_REGISTRY = Registry("ROI_HEADS_OUTPUT")
ROI_HEADS_OUTPUT_REGISTRY.__doc__ = """
Registry for the output layers in ROI heads in a generalized R-CNN model."""

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


COCO_BASE_INDICES = sorted(
    set(range(81)) - set([0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62])
)
LVIS_BASE_INDICES = sorted(
    set(range(1230))
    - set(
        [
            0,
            6,
            9,
            13,
            14,
            15,
            20,
            21,
            30,
            37,
            38,
            39,
            41,
            45,
            48,
            50,
            51,
            63,
            64,
            69,
            71,
            73,
            82,
            85,
            93,
            99,
            100,
            104,
            105,
            106,
            112,
            115,
            116,
            119,
            121,
            124,
            126,
            129,
            130,
            135,
            139,
            141,
            142,
            143,
            146,
            149,
            154,
            158,
            160,
            162,
            163,
            166,
            168,
            172,
            180,
            181,
            183,
            195,
            198,
            202,
            204,
            205,
            208,
            212,
            213,
            216,
            217,
            218,
            225,
            226,
            230,
            235,
            237,
            238,
            240,
            241,
            242,
            244,
            245,
            248,
            249,
            250,
            251,
            252,
            254,
            257,
            258,
            264,
            265,
            269,
            270,
            272,
            279,
            283,
            286,
            290,
            292,
            294,
            295,
            297,
            299,
            302,
            303,
            305,
            306,
            309,
            310,
            312,
            315,
            316,
            317,
            319,
            320,
            321,
            323,
            325,
            327,
            328,
            329,
            334,
            335,
            341,
            343,
            349,
            350,
            353,
            355,
            356,
            357,
            358,
            359,
            360,
            365,
            367,
            368,
            369,
            371,
            377,
            378,
            384,
            385,
            387,
            388,
            392,
            393,
            401,
            402,
            403,
            405,
            407,
            410,
            412,
            413,
            416,
            419,
            420,
            422,
            426,
            429,
            432,
            433,
            434,
            437,
            438,
            440,
            441,
            445,
            453,
            454,
            455,
            461,
            463,
            468,
            472,
            475,
            476,
            477,
            482,
            484,
            485,
            487,
            488,
            492,
            494,
            495,
            497,
            508,
            509,
            511,
            513,
            514,
            515,
            517,
            520,
            523,
            524,
            525,
            526,
            529,
            533,
            540,
            541,
            542,
            544,
            547,
            550,
            551,
            552,
            554,
            555,
            561,
            563,
            568,
            571,
            572,
            580,
            581,
            583,
            584,
            585,
            586,
            589,
            591,
            592,
            593,
            595,
            596,
            599,
            601,
            604,
            608,
            609,
            611,
            612,
            615,
            616,
            625,
            626,
            628,
            629,
            630,
            633,
            635,
            642,
            644,
            645,
            649,
            655,
            657,
            658,
            662,
            663,
            664,
            670,
            673,
            675,
            676,
            682,
            683,
            685,
            689,
            695,
            697,
            699,
            702,
            711,
            712,
            715,
            721,
            722,
            723,
            724,
            726,
            729,
            731,
            733,
            734,
            738,
            740,
            741,
            744,
            748,
            754,
            758,
            764,
            766,
            767,
            768,
            771,
            772,
            774,
            776,
            777,
            781,
            782,
            784,
            789,
            790,
            794,
            795,
            796,
            798,
            799,
            803,
            805,
            806,
            807,
            808,
            815,
            817,
            820,
            821,
            822,
            824,
            825,
            827,
            832,
            833,
            835,
            836,
            840,
            842,
            844,
            846,
            856,
            862,
            863,
            864,
            865,
            866,
            868,
            869,
            870,
            871,
            872,
            875,
            877,
            882,
            886,
            892,
            893,
            897,
            898,
            900,
            901,
            904,
            905,
            907,
            915,
            918,
            919,
            920,
            921,
            922,
            926,
            927,
            930,
            931,
            933,
            939,
            940,
            944,
            945,
            946,
            948,
            950,
            951,
            953,
            954,
            955,
            956,
            958,
            959,
            961,
            962,
            963,
            969,
            974,
            975,
            988,
            990,
            991,
            998,
            999,
            1001,
            1003,
            1005,
            1008,
            1009,
            1010,
            1012,
            1015,
            1020,
            1022,
            1025,
            1026,
            1028,
            1029,
            1032,
            1033,
            1046,
            1047,
            1048,
            1049,
            1050,
            1055,
            1066,
            1067,
            1068,
            1072,
            1073,
            1076,
            1077,
            1086,
            1094,
            1099,
            1103,
            1111,
            1132,
            1135,
            1137,
            1138,
            1139,
            1140,
            1144,
            1146,
            1148,
            1150,
            1152,
            1153,
            1156,
            1158,
            1165,
            1166,
            1167,
            1168,
            1169,
            1171,
            1178,
            1179,
            1180,
            1186,
            1187,
            1188,
            1189,
            1203,
            1204,
            1205,
            1213,
            1215,
            1218,
            1224,
            1225,
            1227,
        ]
    )
)


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image, use_nms):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
            use_nms,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image(boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, use_nms):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...

    box_uncertainty = None
    if boxes.dim() > 2:
        box_uncertainty = boxes[..., 1].reshape(-1, num_bbox_reg_classes, 4)
        boxes = boxes[..., 0]

    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero(as_tuple=False)
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    if box_uncertainty is not None:
        if num_bbox_reg_classes == 1:
            box_uncertainty = box_uncertainty[filter_inds[:, 0], 0]
        else:
            box_uncertainty = box_uncertainty[filter_mask]

    if use_nms:
        # Apply per-class NMS
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

        if box_uncertainty is not None:
            box_uncertainty = box_uncertainty[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]

    if box_uncertainty is not None:
        result.pred_box_uncertainty = box_uncertainty

    return result, filter_inds[:, 0]


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
        cfg,
        gt_boxes=None,
        box_iou_on=False,
        iou_thres=0.5,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]

        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas

        self.smooth_l1_beta = smooth_l1_beta
        self.loss_type = cfg.MODEL.ROI_HEADS.LOSS_TYPE
        self.combine_type = cfg.MODEL.ROI_HEADS.COMBINE_TYPE
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.neg_ratio = cfg.MODEL.ROI_HEADS.NEG_RATIO
        # self.use_bayesian = cfg.MODEL.ROI_HEADS.USE_BAYESIAN
        self.use_bayesian = "Bayesian" in cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_iou_on = box_iou_on or gt_boxes is not None

        # use_rpn_proposals
        self.iou_thres = iou_thres

        self.acti = "tanh"

        if gt_boxes is None:
            # cat(..., dim=0) concatenates over all images in the batch
            try:
                box_type = type(proposals[0].proposal_boxes)
                self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            except:  # noqa: E722
                box_type = type(proposals[0].pred_boxes)
                self.proposals = box_type.cat([p.pred_boxes for p in proposals])
                self.ori_proposals = proposals

            self.image_shapes = [x.image_size for x in proposals]
        else:
            box_type = type(proposals[0])
            try:
                self.proposals = box_type.cat(proposals)
            except:  # noqa: E722
                self.proposals = proposals

            self.gt_boxes = gt_boxes

        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"

        # The following fields should exist only when training.
        if hasattr(proposals[0], "gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            # self.img_indices = torch.cat([torch.full([len(x.gt_classes)], i) for i, x in enumerate(proposals)])
            self.num_images = len(proposals)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        if self.loss_type == "CE":
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")
        else:
            gt_classes = self.gt_classes.clone()

            gt_classes = F.one_hot(gt_classes, num_classes=self.num_classes + 1).float()[:, :-1]
            pred_class_logits = self.pred_class_logits[:, :-1]

            num_pos = (self.gt_classes != self.num_classes).sum() + 1

            return (
                sigmoid_focal_loss_jit(
                    pred_class_logits,
                    gt_classes,
                    reduction="sum",
                    gamma=2,
                    alpha=0.25,
                )
                / num_pos
            )

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(self.proposals.tensor, self.gt_boxes.tensor)
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(1)
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        self.fg_inds = fg_inds
        self.fg_proposal_deltas = self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols]
        self.gt_boxes_filter = self.gt_boxes[fg_inds]
        self.proposals_filter = self.proposals[fg_inds]

        if self.pred_proposal_deltas.dim() == 3:
            box_pred = self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols]
            m, v = box_pred[..., 0], box_pred[..., 1]
            v = F.softplus(v) + 1e-6
            b = gt_proposal_deltas[fg_inds]

            loss_box_reg = 1 / 2 * (m - b).pow(2) / v + 1 / 2 * v
            loss_box_reg = loss_box_reg.sum()
        else:
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """

        if self.box_iou_on:
            gt_ious = matched_boxlist_iou(self.proposals, self.gt_boxes)
            s = self.iou_thres
            pred_ious = self.pred_class_logits[:, 0]

            if self.acti == "tanh":
                new_gt_ious = 2 / (1 - s) * gt_ious - (s + 1) / (1 - s)  # convert to range(-1, 1) for tanh
                new_gt_ious.clamp_(-1, 1)
                pred_ious = pred_ious.tanh()
                smooth_l1_beta = 0.5
            else:
                new_gt_ious = 1 / (1 - s) * gt_ious - s / (1 - s)
                new_gt_ious.clamp_(0, 1)
                pred_ious = pred_ious.sigmoid()
                smooth_l1_beta = 0.25

            iou_loss = smooth_l1_loss(
                pred_ious,
                new_gt_ious,
                smooth_l1_beta,
                reduction="mean",
            )

            src_boxes = self.proposals.tensor
            filter_inds = (src_boxes[:, 2] - src_boxes[:, 0]) * (src_boxes[:, 3] - src_boxes[:, 1]) > 0

            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor[filter_inds], self.gt_boxes.tensor[filter_inds]
            )

            loss_box_reg2 = smooth_l1_loss(
                self.pred_proposal_deltas[filter_inds],
                gt_proposal_deltas,
                self.smooth_l1_beta,
                reduction="mean",
            )

            return {
                "loss_iou": iou_loss,
                "loss_box_reg2": loss_box_reg2,
            }

        return_losses = {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

        if self.use_bayesian:
            return_losses["loss_reg"] = self.box_predictor.var.mean()

        return return_losses

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]

        pred_proposal_uncertainty = None
        if self.pred_proposal_deltas.dim() == 3:
            pred_proposal_uncertainty = F.softplus(self.pred_proposal_deltas[..., 1])
            self.pred_proposal_deltas = self.pred_proposal_deltas[..., 0]

        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        ).view(num_pred, K * B)

        if pred_proposal_uncertainty is not None:
            boxes = torch.stack([boxes, pred_proposal_uncertainty], -1)

        return boxes.split(self.num_preds_per_image, dim=0)

    def predict_boxes_iou(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        # num_pred = len(self.proposals)

        if self.pred_proposal_deltas.dim() == 3:
            self.fg_proposal_uncertainty = F.softplus(
                self.fg_proposal_deltas[..., 1]
            )  # best for both training and finetuning
            self.fg_proposal_deltas = self.fg_proposal_deltas[..., 0]

        # B = self.proposals.tensor.shape[1]
        # K = self.pred_proposal_deltas.shape[1] // B

        boxes = (
            self.box2box_transform.apply_deltas(
                self.fg_proposal_deltas,
                self.proposals_filter.tensor,
            )
            .detach()
            .clone()
        )

        list_boxes = []
        cur = 0

        for i in range(self.num_images):
            lower, cur = cur, cur + self.num_preds_per_image[i]
            temp_boxes = Boxes(boxes[(self.fg_inds >= lower) & (self.fg_inds < cur)])
            temp_boxes.clip(self.image_shapes[i])
            list_boxes.append(temp_boxes)

        return list_boxes

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """

        if self.box_iou_on:
            s = self.iou_thres

            if self.acti == "tanh":
                probs = self.pred_class_logits.tanh()  # in range(-1, 1)
                probs = 1 / 2 * (probs * (1 - s) + (s + 1))  # convert back to range(s, 1)
            else:
                probs = self.pred_class_logits.sigmoid()  # in range(0, 1)
                probs = (1 - s) * probs + s  # convert back to range (s, 1)

        elif self.combine_type == "softmax":
            probs = F.softmax(self.pred_class_logits, dim=-1)
        elif self.combine_type == "sigmoid":
            probs = self.pred_class_logits.sigmoid()
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image, use_nms=True):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        if self.box_iou_on:
            results = []
            for proposal, box, score, image_shape in zip(self.ori_proposals, boxes, scores, image_shapes):
                new_box = Boxes(box)
                # new_box = proposal.pred_boxes

                new_score = proposal.scores * score[:, 0]

                new_class = proposal.pred_classes

                new_box.clip(image_shape)

                if hasattr(proposal, "pred_box_uncertainty"):
                    pred_box_uncertainty = proposal.pred_box_uncertainty

                if use_nms:
                    filter_mask = new_score > score_thresh
                    new_box = new_box[filter_mask]
                    new_score = new_score[filter_mask]
                    new_class = new_class[filter_mask]

                    keep = batched_nms(new_box.tensor, new_score, new_class, nms_thresh)

                    if topk_per_image >= 0:
                        keep = keep[:topk_per_image]
                    new_box, new_score, new_class = new_box[keep], new_score[keep], new_class[keep]

                    if hasattr(proposal, "pred_box_uncertainty"):
                        pred_box_uncertainty = pred_box_uncertainty[filter_mask][keep]

                result = Instances(image_shape)
                result.pred_boxes = new_box
                result.scores = new_score
                result.pred_classes = new_class

                if hasattr(proposal, "pred_box_uncertainty"):
                    result.pred_box_uncertainty = pred_box_uncertainty

                results.append(result)

            return results, None

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thresh,
            nms_thresh,
            topk_per_image,
            use_nms,
        )


@ROI_HEADS_OUTPUT_REGISTRY.register()
class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)

        self.cls_score = nn.Linear(input_size, num_classes + 1)

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if cfg.MODEL.ROI_HEADS.LOSS_TYPE == "focal" and num_classes > 0:
            num_classes = min(100, num_classes)
            prior_prob = 1 / num_classes  # cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)

        return scores, proposal_deltas


@ROI_HEADS_OUTPUT_REGISTRY.register()
class CosineSimOutputLayers(nn.Module):
    """
    Two outputs
    (1) proposal-to-detection box regression deltas (the same as
        the FastRCNNOutputLayers)
    (2) classification score is based on cosine_similarity
    """

    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(CosineSimOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1, bias=False)
        self.scale = cfg.MODEL.ROI_HEADS.COSINE_SCALE
        if self.scale == -1:
            # learnable global scaling factor
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(temp_norm + 1e-5)
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas


@ROI_HEADS_OUTPUT_REGISTRY.register()
class BoxUncertaintyFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super().__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)

        self.cls_score = nn.Linear(input_size, num_classes + 1)

        num_bbox_reg_classes = (1 if cls_agnostic_bbox_reg else num_classes) * 2
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if cfg.MODEL.ROI_HEADS.LOSS_TYPE == "focal" and num_classes > 0:
            prior_prob = 1 / num_classes  # cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)

        proposal_deltas = proposal_deltas.reshape(proposal_deltas.size(0), -1, 2)

        return scores, proposal_deltas


@ROI_HEADS_OUTPUT_REGISTRY.register()
class BayesianOutputLayers(nn.Module):
    """
    Two outputs
    (1) proposal-to-detection box regression deltas (the same as
        the FastRCNNOutputLayers)
    (2) classification score is based on cosine_similarity
    """

    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(BayesianOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)

        self.cls_score_sigma = nn.Linear(input_size, num_classes + 1, bias=True)
        nn.init.constant_(self.cls_score_sigma.weight, math.log(math.e - 1))
        nn.init.constant_(self.cls_score_sigma.bias, 6)

        self.scale = cfg.MODEL.ROI_HEADS.COSINE_SCALE
        if self.scale == -1:
            # learnable global scaling factor
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if cfg.MODEL.ROI_HEADS.LOSS_TYPE == "focal" and num_classes > 0:
            num_classes = min(100, num_classes)
            prior_prob = 1 / num_classes  # cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.cosine_novel_only = cfg.MODEL.ROI_HEADS.COSINE_NOVEL_ONLY

        if self.cosine_novel_only:
            if "lvis" in cfg.DATASETS.TEST[0]:
                self.BASE_INDICES = COCO_BASE_INDICES
            else:
                self.BASE_INDICES = LVIS_BASE_INDICES

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        if self.cosine_novel_only:
            ori_scores = self.cls_score(x)

        self.var = F.softplus(self.cls_score_sigma.weight).unsqueeze(0)
        weight = self.cls_score.weight.unsqueeze(0)
        x_ = x.unsqueeze(1)

        m = (x_ * weight).sum(-1)
        v = (x_ * x_ * self.var).sum(-1)
        k = (1 + math.pi * v / 8).pow(-1 / 2)

        scores = self.scale * k * m + self.cls_score.bias[None, :]

        if self.cosine_novel_only:
            scores[:, self.BASE_INDICES] = ori_scores[:, self.BASE_INDICES]

        proposal_deltas = self.bbox_pred(x)

        return scores, proposal_deltas


@ROI_HEADS_OUTPUT_REGISTRY.register()
class BoxUncertaintyBayesianOutputLayers(nn.Module):
    """
    Two outputs
    (1) proposal-to-detection box regression deltas (the same as
        the FastRCNNOutputLayers)
    (2) classification score is based on cosine_similarity
    """

    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super().__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)

        self.cls_score_sigma = nn.Linear(input_size, num_classes + 1, bias=True)
        nn.init.constant_(self.cls_score_sigma.weight, math.log(math.e - 1))
        nn.init.constant_(self.cls_score_sigma.bias, 6)

        self.scale = cfg.MODEL.ROI_HEADS.COSINE_SCALE
        if self.scale == -1:
            # learnable global scaling factor
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        num_bbox_reg_classes = (1 if cls_agnostic_bbox_reg else num_classes) * 2
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if cfg.MODEL.ROI_HEADS.LOSS_TYPE == "focal" and num_classes > 0:
            num_classes = min(100, num_classes)
            prior_prob = 1 / num_classes  # cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.cosine_novel_only = cfg.MODEL.ROI_HEADS.COSINE_NOVEL_ONLY

        if self.cosine_novel_only:
            if "lvis" in cfg.DATASETS.TEST[0]:
                self.BASE_INDICES = COCO_BASE_INDICES
            else:
                self.BASE_INDICES = LVIS_BASE_INDICES

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        if self.cosine_novel_only:
            ori_scores = self.cls_score(x)

        self.var = F.softplus(self.cls_score_sigma.weight).unsqueeze(0)
        weight = self.cls_score.weight.unsqueeze(0)
        x_ = x.unsqueeze(1)

        m = (x_ * weight).sum(-1)
        v = (x_ * x_ * self.var).sum(-1)
        k = (1 + math.pi * v / 8).pow(-1 / 2)

        scores = self.scale * k * m + self.cls_score.bias[None, :]

        if self.cosine_novel_only:
            scores[:, self.BASE_INDICES] = ori_scores[:, self.BASE_INDICES]

        proposal_deltas = self.bbox_pred(x)

        proposal_deltas = proposal_deltas.reshape(proposal_deltas.size(0), -1, 2)

        return scores, proposal_deltas
