import cv2
import copy
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd
import numpy as np
from skimage import draw, measure
import scipy
import random
import detectron2.data.transforms as T
import torch
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import Boxes, pairwise_iou, BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper

from typing import List, Union
import pycocotools.mask as mask_util
from collections import defaultdict
import torch
from PIL import Image

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

class ValidationMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True, **kwargs):
        super().__init__(cfg, is_train, **kwargs)
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            # logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]   # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        # if not self.is_train:
        #     # dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    
    if len(annos):
        if "region_shape" in annos[0]:
            shapes = [None if obj["region_shape"] is None else int(obj["region_shape"]) for obj in annos]
            if not all(v is None for v in shapes):
                shapes = torch.tensor(shapes, dtype=torch.int64)
                target.gt_shapes = shapes
                
        if "region_cancer" in annos[0]:
            cancers = [None if obj["region_cancer"] is None else int(obj["region_cancer"]) for obj in annos]
            if not all(v is None for v in cancers):
                cancers = torch.tensor(cancers, dtype=torch.int64)
                target.gt_cancers = cancers

        if "region_orientation" in annos[0]:
            orients = [None if obj["region_orientation"] is None else int(obj["region_orientation"]) for obj in annos]
            if not all(v is None for v in orients):
                orients = torch.tensor(orients, dtype=torch.int64)
                target.gt_orients = orients
                
        if "region_margin" in annos[0]:
            margins = [None if obj["region_margin"] is None else int(obj["region_margin"]) for obj in annos]
            if not all(v is None for v in margins):
                margins = torch.tensor(margins, dtype=torch.int64)
                target.gt_margins = margins

        if "region_echo" in annos[0]:
            echos = [None if obj["region_echo"] is None else int(obj["region_echo"]) for obj in annos]
            if not all(v is None for v in echos):
                echos = torch.tensor(echos, dtype=torch.int64)
                target.gt_echos = echos
                
        if "region_posterior" in annos[0]:
            posts = [None if obj["region_posterior"] is None else int(obj["region_posterior"]) for obj in annos]
            if not all(v is None for v in posts):
                posts = torch.tensor(posts, dtype=torch.int64)
                target.gt_posts = posts
                
    else:
        shapes = torch.tensor([], dtype=torch.int64)
        target.gt_shapes = shapes
        cancer = torch.tensor([], dtype=torch.int64)
        target.gt_cancers = cancer
        orients = torch.tensor([], dtype=torch.int64)
        target.gt_orients = orients
        margins = torch.tensor([], dtype=torch.int64)
        target.gt_margins = margins
        echos = torch.tensor([], dtype=torch.int64)
        target.gt_echos = echos
        posts = torch.tensor([], dtype=torch.int64)
        target.gt_posts = posts
    

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    # padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(binary_mask)
    contours = np.subtract(contours, 1, dtype=object)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        min_x = min(segmentation[::2])
        min_y = min(segmentation[1::2])
        width = max(segmentation[::2]) - min_x
        height = max(segmentation[1::2]) - min_y
        polygons.append([segmentation, min_x, min_y, width, height])

    return polygons

def find_closest_bbox(bbox_list_1, bbox_list_2):
    sorted_order = []
    sorted_errors = []
    
    # for every bbox in the first list provided
    for x in list(range(len(bbox_list_1))):
        box1 = bbox_list_1[x]
        errors = []
        
        for y in list(range(len(bbox_list_2))):
            box2 = bbox_list_2[y]
            
            error = sum([abs(res1 - res2) for (res1, res2) in zip(box1, box2)])
            errors.append(error)
         
        sorted_order.append(np.argmin(errors))
        sorted_errors.append(np.min(errors))
        
    if (len(sorted_order) > len(bbox_list_2)):
        while len(sorted_order) > len(bbox_list_2):
            remove = np.argmin(sorted_errors)
            sorted_errors.pop(remove)
            sorted_order.pop(remove)
            bbox_list_1.pop(remove)
        
    return sorted_order, bbox_list_1

class CustomMapper(DatasetMapper):
    def __init__(self, cfg, is_train: bool = True, augmentations: List = [], **kwargs):
        super().__init__(cfg, is_train, **kwargs)
        self.is_train = is_train
        self.mode = "training" if is_train else "inference"
        self.augmentations = augmentations
        try:
            self.fields = cfg.MODEL.CBM.CONCEPTS
        except AttributeError:
            self.fields = []

        self.image_format           = None
        self.use_instance_mask      = True
        self.instance_mask_format   = "polygon"
        self.use_keypoint           = False
        self.keypoint_hflip_indices = None
        self.proposal_topk          = None
        self.recompute_boxes        = False

    def __call__(self, dataset_dict):
        # making a copy of the dataset instance we're being passed into this function 
        dataset_dict = copy.deepcopy(dataset_dict)
        
        img = Image.new('L', (int(dataset_dict['width']), int(dataset_dict['height'])), 0)
        segmentation = [x['segmentation'][0] for x in dataset_dict['annotations']]
        for x in segmentation:
            seg = [int(y) for y in x]
            ImageDraw.Draw(img).polygon(seg, outline=1, fill=1)
        mask = np.array(img)
            
        # reading in the image using the Detectron2 method, instead of OpenCV
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        num_lesions = len(dataset_dict['annotations'])
        
        for x, y in zip(dataset_dict['annotations'], dataset_dict['annotations'][1:]):
            if np.all(x == y):
                dataset_dict['annotations'].remove(x)
                
        # Reformatting bounding boxes and categories to play nice with albumentations 
        bboxes = [x['bbox'] for x in dataset_dict['annotations']]

        # Defining our augmentations for training, using the albumentations library here for future-proofing, in case we want to 
        # add copy-paste or another more complex augmentation than Detectron2 allows for. 
        #print(dataset_dict['annotations'])
        if (self.mode == 'training'):
            # REDUNDANCY IN FLIPPING 
            augmentations = T.AugmentationList(self.augmentations)
        
        else:
            augmentations = T.AugmentationList([])
            
        concept_kwargs = defaultdict(list)

        category_ids = [x['category_id'] for x in dataset_dict['annotations']]
        region_cancer = [x['region_cancer'] for x in dataset_dict['annotations']]

        if 'shape' in self.fields:
            region_shape = [x['region_shape'] for x in dataset_dict['annotations']]
            concept_kwargs['region_shapes'] = region_shape

        if "orientation" in self.fields:
            region_orient = [x['region_orientation'] for x in dataset_dict['annotations']]
            concept_kwargs['region_orients'] = region_orient
        
        if "margin" in self.fields:
            region_margin = [x['region_margin'] for x in dataset_dict['annotations']]
            concept_kwargs['region_margins'] = region_margin

        if "echo" in self.fields:
            region_echo = [x['region_echo'] for x in dataset_dict['annotations']]
            concept_kwargs['region_echos'] = region_echo

        if "posterior" in self.fields:
            region_posterior = [x['region_posterior'] for x in dataset_dict['annotations']]
            concept_kwargs['region_posteriors'] = region_posterior

        concept_kwargs['category_ids'] = category_ids
        concept_kwargs['region_cancers'] = region_cancer

        aug_input = T.AugInput(image, sem_seg=None)
        transforms = augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        image_shape = image.shape[:2]

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
            
        
        return dataset_dict
    
    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)