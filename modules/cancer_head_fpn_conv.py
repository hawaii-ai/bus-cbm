import cv2
import numpy as np
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
from typing import Dict, List, Optional, Tuple

from detectron2.layers import Conv2d, ShapeSpec, get_norm, cat
from detectron2.modeling import ROI_HEADS_REGISTRY, ROIHeads
from detectron2.config import configurable
from detectron2.config import CfgNode as CN, get_cfg
from detectron2.utils.registry import Registry
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import select_foreground_proposals, BaseMaskRCNNHead
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from torch.nn import Conv2d, BatchNorm2d, ReLU, Identity, AdaptiveAvgPool2d, LazyLinear, Linear, Flatten

import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, add_last_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.last_relu = add_last_relu

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Apply residual connection
        out += self.shortcut(identity)
        if self.last_relu:
            out = self.relu(out)
        return out

@ROI_HEADS_REGISTRY.register()
class CBMCancerHead(BaseMaskRCNNHead):
    @configurable
    def __init__(self,
                 n_class_attr, # no. of mass lexicon characteristics
                 concepts,
                 train_cancer, # if we're training cancer or not 
                 side_channel, # if we're including a side channel with the cancer training 
                 input_shape, # number of bbox dims from backbone
                 n_feats, # number of features to use in linear 
                 num_classes, # cancer or no cancer 
                 conv_dims, # number of filters in convolutional layers
                 loss_weight, # lambda in CBM paper, weight balancing
                 use_sigmoid, # use sigmoid in intermediate layer, tunable
                 linear # whether or not we're training a linear cancer head
                ):
    
        super(BaseMaskRCNNHead, self).__init__()
        self.train_cancer = train_cancer
        self.side_channel = side_channel
        self.linear = linear

        self.concepts = concepts
        if len(self.concepts) > 0:
            self.concept_mappings = {'shape' : 'shapes',
                                    'orientation': 'orients', 
                                    'margin' : 'margins', 
                                    'posterior': 'posts', 
                                    'echo' : 'echos'}
        else: self.concept_mappings = {}

        self.use_sigmoid = use_sigmoid
         # adding some conv layers similar to the mask head 
        in_channels = input_shape.channels
        layers = []

        # we're going to fix this as three bottleneck layers for simplicity
        layers.append(Bottleneck(in_channels=in_channels, 
                                out_channels=conv_dims[0]))
        layers.append(nn.MaxPool2d(3, stride=2))

        layers.append(Bottleneck(in_channels=conv_dims[0], 
                                out_channels=conv_dims[1]))
        layers.append(nn.MaxPool2d(3, stride=2))

        if len(self.concepts) < 1:
            layers.append(Bottleneck(in_channels=conv_dims[1], out_channels=1, add_last_relu=False))
            layers.append(nn.MaxPool2d(2))
            layers.append(Flatten())

        self.first_model = nn.Sequential(*layers)

        if len(self.concepts) > 0:
            self.transfer_concepts = nn.Sequential(
                Bottleneck(in_channels=conv_dims[1], out_channels=n_class_attr, add_last_relu=False), 
                nn.MaxPool2d(2), 
                Flatten()
            )

        if self.train_cancer and len(self.concepts) > 0:
            # if we're training cancer, need to freeze the concepts
            for param in self.first_model.parameters():
                param.requires_grad = False
            for param in self.transfer_concepts.parameters():
                param.requires_grad = False 

            if self.side_channel:
                if self.linear:
                    raise AssertionError("Cannot train side channel with linear cancer head")
                self.transfer_side = nn.Sequential(
                    Bottleneck(in_channels=conv_dims[1], out_channels=1, add_last_relu=False), 
                    nn.MaxPool2d(2), 
                    Flatten()
                )

            self.second_model = nn.Sequential(
                Linear(in_features=n_class_attr+self.side_channel, out_features=n_feats),
                ReLU(),
                Linear(in_features=n_feats, out_features=n_feats),
                ReLU(),
                Linear(in_features=n_feats, out_features=n_feats),
                ReLU(),
                Linear(in_features=n_feats, out_features=num_classes)
            )

            # if we're training linear, adjust the second model to be a single layer
            if self.linear:
                self.second_model = nn.Sequential(Linear(in_features=n_class_attr, out_features=num_classes))

        else:
            self.second_model = nn.Sequential(
                Identity()
            )

        self.loss = nn.BCEWithLogitsLoss()
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        return {
            'n_class_attr' : len(cfg.MODEL.CBM.CONCEPTS),
            'concepts' : cfg.MODEL.CBM.CONCEPTS,
            'input_shape' : input_shape,
            'linear' : cfg.MODEL.CBM.LINEAR,
            'side_channel' : cfg.MODEL.CBM.SIDE_CHANNEL,
            'loss_weight' : cfg.MODEL.ROI_CANCER_HEAD.LAMBDA,
            'n_feats' : cfg.MODEL.ROI_CANCER_HEAD.NUM_FC,
            'num_classes' : cfg.MODEL.ROI_CANCER_HEAD.NUM_CLASSES,
            'conv_dims' : cfg.MODEL.ROI_CANCER_HEAD.NUM_CONV,
            'use_sigmoid' : cfg.MODEL.CBM.USE_SIGMOID, 
            "train_cancer" : cfg.MODEL.CBM.CANCER_ON, 
        }

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        first_pass = self.first_model(x)
        if len(self.concepts) < 1:
            # if we're training the non-explainable cancer head 
            if self.training:
                loss_dict = self.loss(None, first_pass, self.concepts, instances)
                return loss_dict
            else:
                return self.inference(None, first_pass, self.concepts, instances)
        else:    
            pred_attr_logits = self.transfer_concepts(first_pass)
            if self.use_sigmoid:
                stage2_inputs = torch.sigmoid(pred_attr_logits)
            else:
                stage2_inputs = pred_attr_logits 
            
            if self.side_channel:
                side_stage2_inputs = self.transfer_side(first_pass)
                stage2_inputs = torch.cat((stage2_inputs, side_stage2_inputs), dim=1)
        
            #attr_pred_logits = torch.cat(pred_attr_logits, dim=1)
            class_pred_logits = self.second_model(stage2_inputs)
        
        if self.training:
            loss_dict = self.loss(pred_attr_logits, class_pred_logits, self.concepts, instances)
            return loss_dict
        else:
            return self.inference(pred_attr_logits, class_pred_logits, self.concepts, instances)
    
    @torch.jit.unused
    def loss(self,  attr_pred_logits, pred_targets, concepts, instances):
        '''
        we are going to assume that the order of the concepts passed in 
        here in "concepts" is going to be the same as the order in which 
        they are coded in the network, so concepts[x] corresponds the name 
        of the prediction logit in attr_pred_logits[x]
        '''
        gt_concepts = dict()
        concepts_loss_dict = dict()

        for x in concepts:
            gt_concepts['gt_' + self.concept_mappings[x]] = []

        gt_concepts['gt_cancers'] = []

        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue

            # loop through the concepts and add their GTs to a dictionary of lists
            for x in gt_concepts.keys():
                gt_concepts_per_image = instances_per_image.get(x).to(dtype=torch.float32)
                gt_concepts[x].append(gt_concepts_per_image)

        # computing all the concept BCE losses 
        for i, x in enumerate(concepts):
            gt_concept_temp = cat(gt_concepts['gt_' + self.concept_mappings[x]], dim=0)
            loss_temp = F.binary_cross_entropy_with_logits(attr_pred_logits[:, i], 
                                                           gt_concept_temp, 
                                                           reduction="mean")
            concepts_loss_dict[x + '_loss'] = loss_temp

        if self.train_cancer:
            gt_cancer_temp = cat(gt_concepts['gt_cancers'], dim=0)
            loss_temp = F.binary_cross_entropy_with_logits(torch.squeeze(pred_targets, dim=-1), 
                                                            gt_cancer_temp, 
                                                            reduction="mean")
            concepts_loss_dict['cancer_loss'] = loss_temp

        return concepts_loss_dict
    
    def inference(self, attr_pred_logits, pred_targets, concepts, pred_instances):
        # Select masks corresponding to the predicted classes
        if (attr_pred_logits is not None and len(attr_pred_logits) > 0) or len(self.concepts) < 1:
            concepts_pred_dict = dict()
            
            num_boxes_per_image = [len(i) for i in pred_instances]
            
            for i, x in enumerate(concepts):
                concepts_pred_dict[x] = torch.sigmoid(attr_pred_logits[:, i])
                concepts_pred_dict[x + '_class'] = torch.where(concepts_pred_dict[x] > 0.5, 1, 0)

                concepts_pred_dict[x] = concepts_pred_dict[x].split(num_boxes_per_image, dim=0)
                concepts_pred_dict[x + '_class'] = concepts_pred_dict[x + '_class'].split(num_boxes_per_image, dim=0)


            if self.train_cancer:
                concepts_pred_dict['cancer'] = torch.sigmoid(pred_targets)
                concepts_pred_dict['cancer_class'] = torch.where(concepts_pred_dict['cancer'] > 0.5, 1, 0)
                concepts_pred_dict['cancer'] = concepts_pred_dict['cancer'].split(num_boxes_per_image, dim=0)
                concepts_pred_dict['cancer_class'] = concepts_pred_dict['cancer_class'].split(num_boxes_per_image, dim=0)


            # ("shape", "margin", "orientation", "echo", "posterior")
            for instances in pred_instances:
                if 'shape' in concepts:
                    for x, y in zip(concepts_pred_dict['shape'], concepts_pred_dict['shape_class']):
                        instances.shape_scores = x
                        instances.shape_pred = y
                if 'orientation' in concepts:
                    for x, y in zip(concepts_pred_dict['orientation'], concepts_pred_dict['orientation_class']):
                        instances.orient_scores = x
                        instances.orient_pred = y
                if 'margin' in concepts:
                    for x, y in zip(concepts_pred_dict['margin'], concepts_pred_dict['margin_class']):
                        instances.margin_scores = x
                        instances.margin_pred = y
                if 'echo' in concepts:
                    for x, y in zip(concepts_pred_dict['echo'], concepts_pred_dict['echo_class']):
                        instances.echo_scores = x
                        instances.echo_pred = y
                if 'posterior' in concepts:
                    for x, y in zip(concepts_pred_dict['posterior'], concepts_pred_dict['posterior_class']):
                        instances.post_scores = x
                        instances.post_pred = y
                
                if self.train_cancer:
                    for x, y in zip(concepts_pred_dict['cancer'], concepts_pred_dict['cancer_class']):
                        instances.cancer_scores = torch.squeeze(x, dim=-1)
                        instances.cancer_pred = torch.squeeze(y, dim=-1)    
                
        else:
            for instances in pred_instances:
                instances.concept_scores =  torch.empty(0)
                if self.train_cancer:
                    instances.cancer_scores = torch.empty(0)
        
        return pred_instances

def build_cancer_head(cfg, input_shape):
    """
    Build a cancer head defined by `cfg.MODEL.ROI_CANCER_HEAD.NAME`.
     """
    name = cfg.MODEL.ROI_CANCER_HEAD.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)

