from typing import List, Optional, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ReLU, Identity, Linear, Flatten

from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.config import CfgNode as CN
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import BaseMaskRCNNHead

class Bottleneck(nn.Module):
    """
        Bottleneck residual block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            add_last_relu (bool, optional): Whether to apply ReLU activation after the last convolution. 
                                            Defaults to True.
        """
    def __init__(self, in_channels: int, out_channels: int, add_last_relu: bool = True):
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
                 n_class_attr: int, # no. of mass lexicon characteristics
                 concepts: List[str], # names of mass lexicon characteristics
                 train_cancer: bool, # if we're training cancer or not 
                 side_channel: bool, # if we're including a side channel with the cancer training 
                 input_shape: ShapeSpec, # number of bbox dims from backbone
                 n_feats: int, # number of features to use in linear 
                 num_classes: int, # number of output classes
                 conv_dims: List[int], # number of filters in convolutional layers
                 use_sigmoid: bool, # use sigmoid in intermediate layer, tunable
                 linear: bool # whether or not we're training a linear cancer head
                ):
    
        super(BaseMaskRCNNHead, self).__init__()

        self.train_cancer = train_cancer
        self.side_channel = side_channel
        self.linear = linear
        self.concepts = concepts
        self.use_sigmoid = use_sigmoid

        # ad-hoc from BUS CBM paper - needs to be updated for your data if concepts are encoded differently 
        # between COCO format annotations and the instance predictions 
        if len(self.concepts) > 0:
            self.concept_mappings = {'shape' : 'shapes',
                                    'orientation': 'orients', 
                                    'margin' : 'margins', 
                                    'posterior': 'posts', 
                                    'echo' : 'echos'}
        else: self.concept_mappings = {}
        
        in_channels = input_shape.channels
        layers = []

        # we're going to fix this as three bottleneck layers for simplicity
        layers.append(Bottleneck(in_channels=in_channels, 
                                out_channels=conv_dims[0]))
        layers.append(nn.MaxPool2d(3, stride=2))
        layers.append(Bottleneck(in_channels=conv_dims[0], 
                                out_channels=conv_dims[1]))
        layers.append(nn.MaxPool2d(3, stride=2))

        # if we're training the non-explainable cancer head (no concepts)
        if len(self.concepts) < 1:
            layers.append(Bottleneck(in_channels=conv_dims[1], out_channels=1, add_last_relu=False))
            layers.append(nn.MaxPool2d(2))
            layers.append(Flatten())

        self.first_model = nn.Sequential(*layers)

        # if we're training any version of the explainable head 
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

            # if we're training a bottleneck layer with a side channel 
            if self.side_channel:
                # linear cancer head is incompatible with a side channel in our implementation (not interpretable)
                if self.linear:
                    raise AssertionError("Cannot train side channel with linear cancer head")
                self.transfer_side = nn.Sequential(
                    Bottleneck(in_channels=conv_dims[1], out_channels=1, add_last_relu=False), 
                    nn.MaxPool2d(2), 
                    Flatten()
                )

            # model to move from concept space to cancer space, number of input features depends on side channel existence 
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

        # if we're training concepts only (train_cancer = False) then fill in dummy second model
        else:
            self.second_model = nn.Sequential(
                Identity()
            )

        # loss is always BCE, for concepts or for cancer training 
        self.loss = nn.BCEWithLogitsLoss()
        
    @classmethod
    def from_config(cls, cfg: CN, input_shape: ShapeSpec):
        # fmt: off
        return {
            'n_class_attr' : len(cfg.MODEL.CBM.CONCEPTS), # no. of mass lexicon characteristics
            'concepts' : cfg.MODEL.CBM.CONCEPTS, # names of mass lexicon characteristics
            'input_shape' : input_shape, # number of bbox dims from backbone
            'linear' : cfg.MODEL.CBM.LINEAR, # whether or not we're training a linear cancer head
            'side_channel' : cfg.MODEL.CBM.SIDE_CHANNEL, # if we're including a side channel with the cancer training
            'n_feats' : cfg.MODEL.ROI_CANCER_HEAD.NUM_FC, # number of features to use in linear 
            'num_classes' : cfg.MODEL.ROI_CANCER_HEAD.NUM_CLASSES, # number of output classes
            'conv_dims' : cfg.MODEL.ROI_CANCER_HEAD.NUM_CONV, # number of filters in convolutional layers
            'use_sigmoid' : cfg.MODEL.CBM.USE_SIGMOID, # use sigmoid in intermediate layer, tunable
            "train_cancer" : cfg.MODEL.CBM.CANCER_ON, # if we're training cancer or not 
        }

    def forward(self, x: torch.Tensor, instances: List[Instances]):
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
        # predict the concepts 
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
            # if we're applying sigmoid at the intermediate step 
            if self.use_sigmoid:
                stage2_inputs = torch.sigmoid(pred_attr_logits)
            else:
                stage2_inputs = pred_attr_logits 
            
            # if we're training a model with a side channel in the bottleneck
            if self.side_channel:
                side_stage2_inputs = self.transfer_side(first_pass)
                # tack on the side channel features to the concept features
                stage2_inputs = torch.cat((stage2_inputs, side_stage2_inputs), dim=1)
        
            # predict the cancer logits (or identities, for stage 2)
            class_pred_logits = self.second_model(stage2_inputs)
        
        if self.training:
            loss_dict = self.loss(pred_attr_logits, class_pred_logits, self.concepts, instances)
            return loss_dict
        else:
            return self.inference(pred_attr_logits, class_pred_logits, self.concepts, instances)
    
    @torch.jit.unused
    def loss(self, 
             attr_pred_logits: torch.Tensor, # predicted logits of concepts
             pred_targets: torch.Tensor,  # predicted final class
             concepts: List[str], # list of names of concepts 
             instances: List[Instances]): # instances which hold the ground truth 
        '''
        we are going to assume that the order of the concepts passed in 
        here in "concepts" is going to be the same as the order in which 
        they are coded in the network, so concepts[x] corresponds the name 
        of the prediction logit in attr_pred_logits[x]
        '''
        gt_concepts = dict()
        concepts_loss_dict = dict()

        # create a dictionary to store concept values 
        for x in concepts:
            gt_concepts['gt_' + self.concept_mappings[x]] = []

        # add on final classification 
        gt_concepts['gt_cancers'] = []

        for instances_per_image in instances:
            # if we didn't predict any instances, no loss to compute here
            if len(instances_per_image) == 0:
                continue

            # loop through the concepts and add their GTs to a dictionary of lists
            for x in gt_concepts.keys():
                gt_concepts_per_image = instances_per_image.get(x).to(dtype=torch.float32)
                gt_concepts[x].append(gt_concepts_per_image)

        # computing all the concept BCE losses - ORDER MATTERS 
        for i, x in enumerate(concepts):
            gt_concept_temp = cat(gt_concepts['gt_' + self.concept_mappings[x]], dim=0)
            loss_temp = F.binary_cross_entropy_with_logits(attr_pred_logits[:, i], 
                                                           gt_concept_temp, 
                                                           reduction="mean")
            concepts_loss_dict[x + '_loss'] = loss_temp

        # if we're training cancer, compute the loss 
        if self.train_cancer:
            gt_cancer_temp = cat(gt_concepts['gt_cancers'], dim=0)
            loss_temp = F.binary_cross_entropy_with_logits(torch.squeeze(pred_targets, dim=-1), 
                                                            gt_cancer_temp, 
                                                            reduction="mean")
            concepts_loss_dict['cancer_loss'] = loss_temp

        return concepts_loss_dict
    
    def inference(self, 
                  attr_pred_logits: Optional[torch.Tensor], 
                  pred_targets: Optional[torch.Tensor], 
                  concepts: List[str], 
                  pred_instances: List[Instances]):
        """
            Perform inference using the predicted logits.

            Args:
                attr_pred_logits (Optional[torch.Tensor]): Predicted logits for concepts.
                pred_targets (Optional[torch.Tensor]): Predicted targets.
                concepts (List[str]): List of concept names.
                pred_instances (List[Instances]): List of predicted instances.

            Returns:
                List[Instances]: Predicted instances after inference.
        """
        # If we a) predicted some concepts or b) don't have any concepts 
        if (attr_pred_logits is not None and len(attr_pred_logits) > 0) or len(self.concepts) < 1:
            concepts_pred_dict = dict()
            num_boxes_per_image = [len(i) for i in pred_instances]
            
            # store the concept soft and binarized predictions 
            for i, x in enumerate(concepts):
                concepts_pred_dict[x] = torch.sigmoid(attr_pred_logits[:, i])
                concepts_pred_dict[x + '_class'] = torch.where(concepts_pred_dict[x] > 0.5, 1, 0)
                # associate predictions with the predicted boxes 
                concepts_pred_dict[x] = concepts_pred_dict[x].split(num_boxes_per_image, dim=0)
                concepts_pred_dict[x + '_class'] = concepts_pred_dict[x + '_class'].split(num_boxes_per_image, dim=0)

            # if we're training cancer, add in those predictions as well 
            if self.train_cancer:
                concepts_pred_dict['cancer'] = torch.sigmoid(pred_targets)
                concepts_pred_dict['cancer_class'] = torch.where(concepts_pred_dict['cancer'] > 0.5, 1, 0)
                # associate predictions with the predicted boxes 
                concepts_pred_dict['cancer'] = concepts_pred_dict['cancer'].split(num_boxes_per_image, dim=0)
                concepts_pred_dict['cancer_class'] = concepts_pred_dict['cancer_class'].split(num_boxes_per_image, dim=0)

            # for each instance in our predicted instances, add on our scores and predictions to the returned instances
            # HARD CODED VALUES NEED TO CHANGE WITH YOUR CODE 
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
            # if we predicted nothing for this set, fill in our concept and cancer values with 0 values
            for instances in pred_instances:
                instances.concept_scores =  torch.empty(0)
                if self.train_cancer:
                    instances.cancer_scores = torch.empty(0)
        
        return pred_instances

def build_cancer_head(cfg: Any, input_shape: ShapeSpec):
    """
    Build a cancer head defined by `cfg.MODEL.ROI_CANCER_HEAD.NAME`.
     """
    name = cfg.MODEL.ROI_CANCER_HEAD.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)