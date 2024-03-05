# Most code in this file is taken from detectron2.Visualizer, adapted for our purposes
# All credit for unmodified code and structure goes to the original authors.  
import torch
import random
import numpy as np
import matplotlib.colors as mplc
from typing import Dict, List, Optional, Any, Tuple

from detectron2.structures import BoxMode, Instances
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask

class MyVisualizer(Visualizer):
    def __init__(self, img_rgb: np.ndarray,
                  metadata = None, 
                  scale: float = 1.0, 
                  instance_mode: ColorMode = ColorMode.IMAGE, 
                  show_lesion: bool = True, plain_cancer: bool = False):
        """
        Initializes the MyVisualizer class.

        Args:
            img_rgb: RGB image data.
            metadata: Metadata of the image.
            scale: Scaling factor.
            instance_mode: Instance mode for visualization.
            show_lesion: Whether to show only lesions.
            plain_cancer: Whether to show only plain cancer.
        """
        super().__init__(img_rgb, metadata, scale, instance_mode)
        # if we're showing the objectness scores for the lesion class
        self.show_lesion = show_lesion
        # if we're only showing cancer prediction, not concept prediction 
        self.plain_cancer = plain_cancer
        
    # overloaded method from detectron2 (adding in the new BIRADS attributes)
    def draw_dataset_dict(self, dic: Dict):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)

        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None

            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]
            
            keypts = None

            colors = ["yellow", "green", "pink"]
            
            category_ids = [x["category_id"] for x in annos]
            
            # HARD-CODED VALUES, NEED TO CHANGE WITH YOUR IMPLEMENTATION
            cancers = [None if ("region_cancer" not in obj or obj["region_cancer"] is None) else int(obj["region_cancer"]) for obj in annos]
            shapes = [None if ("region_shape" not in obj or obj["region_shape"] is None) else int(obj["region_shape"]) for obj in annos]
            orient = [None if ("region_orientation" not in obj or obj["region_orientation"] is None) else int(obj["region_orientation"]) for obj in annos]
            margin = [None if ("region_margin" not in obj or obj["region_margin"] is None) else int(obj["region_margin"]) for obj in annos]
            echo =  [None if ("region_echo" not in obj or obj["region_echo"] is None) else int(obj["region_echo"]) for obj in annos]
            posterior =  [None if ("region_posterior" not in obj or obj["region_posterior"] is None) else int(obj["region_posterior"]) for obj in annos]

            # All annos/fields which we don't have are all lists of Nones in the above list 
            
            # HARD-CODED VALUES, NEED TO CHANGE WITH YOUR IMPLEMENTATION
            names = self.metadata.get("thing_classes", None)
            shape_names = self.metadata.get("shape_classes", None)
            orient_names = self.metadata.get("orientation_classes", None)
            margin_names = self.metadata.get("margin_classes", None)
            posterior_names = self.metadata.get("posterior_classes", None)
            echo_names = self.metadata.get("echo_classes", None)
            cancer_names = self.metadata.get("cancer_classes", None)

            # show_lesion should be true whenever we're trying to look at the lesion only model 
            if self.show_lesion:
                full_cats = [category_ids]
                full_names = [names]
            elif self.plain_cancer:
                full_cats = [category_ids]
                full_names = [cancer_names]
            else:
                full_cats = []
                full_names = []
            
            if not all(v is None for v in cancers):
                full_cats.append(cancers)
                full_names.append(cancer_names)
        
            if not all(v is None for v in shapes):
                full_cats.append(shapes)
                full_names.append(shape_names)
        
            if not all(v is None for v in orient):
                full_cats.append(orient)
                full_names.append(orient_names)
                
            if not all(v is None for v in margin):
                full_cats.append(margin)
                full_names.append(margin_names)
                
            if not all(v is None for v in posterior):
                full_cats.append(posterior)
                full_names.append(posterior_names)
                
            if not all(v is None for v in echo):
                full_cats.append(echo)
                full_names.append(echo_names)

            if self._instance_mode == ColorMode.IMAGE and self.metadata.get("thing_colors"):
                colors = [
                    self.metadata.thing_colors[c] for c in cancers
                ]
        
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    (self._jitter([x / 255 for x in self.metadata.thing_colors[c]]))
                    for c in cancers
                ]

            labels = _create_group_text_labels(
                full_cats,
                scores=None,
                class_names=full_names,
                is_crowd=[x.get("iscrowd", 0) for x in annos],
            )
            
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
            )

        return self.output
    
    def draw_dataset_instances(self, instances: Instances):
        """
        Draw annotations/segmentations in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = instances.gt_boxes if instances.has("gt_boxes") else [None]
        category_ids = instances.gt_classes if instances.has("gt_classes") else [None] 
        masks = instances.gt_masks if instances.has("gt_masks") else [None] 

        shapes = instances.gt_shapes if instances.has("gt_shapes") else [None ]
        cancers = instances.gt_cancers if instances.has("gt_cancers") else [None]
        orient = instances.gt_orients if instances.has("gt_orients") else [None] 
        margin = instances.gt_margins if instances.has("gt_margins") else [None] 
        posterior = instances.gt_posts if instances.has("gt_posts") else [None] 
        echo = instances.gt_echos if instances.has("gt_echos") else [None] 
            
        keypts = None

        colors = ["yellow", "green", "pink"]
            
        # All annos/fields which we don't have are all lists of Nones in the above list 

        class_to_color = {0.0 : 'green', 1.0 : 'yellow'}
        
        names = self.metadata.get("thing_classes", None)
        shape_names = self.metadata.get("shape_classes", None)
        orient_names = self.metadata.get("orientation_classes", None)
        margin_names = self.metadata.get("margin_classes", None)
        posterior_names = self.metadata.get("posterior_classes", None)
        echo_names = self.metadata.get("echo_classes", None)
        cancer_names = self.metadata.get("cancer_classes", None)


        colors = None
        # show_lesion should be true whenever we're trying to look at the lesion only model 
        if self.show_lesion:
            full_cats = [category_ids.numpy()]
            full_names = [names]
        elif self.plain_cancer:
            full_cats = [category_ids.numpy()]
            full_names = [cancer_names]
        else:
            full_cats = []
            full_names = []
        
        if not all(v is None for v in cancers):
            full_cats.append(cancers)
            full_names.append(cancer_names)

            if self._instance_mode == ColorMode.IMAGE and self.metadata.get("thing_colors"):
                colors = [
                    self.metadata.thing_colors[c] for c in cancers
                ]

            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    (self._jitter([x / 255 for x in self.metadata.thing_colors[c]]))
                    for c in cancers
                ]
    
        if not all(v is None for v in shapes):
            full_cats.append(shapes)
            full_names.append(shape_names)
    
        if not all(v is None for v in orient):
            full_cats.append(orient)
            full_names.append(orient_names)
            
        if not all(v is None for v in margin):
            full_cats.append(margin)
            full_names.append(margin_names)
            
        if not all(v is None for v in posterior):
            full_cats.append(posterior)
            full_names.append(posterior_names)
            
        if not all(v is None for v in echo):
            full_cats.append(echo)
            full_names.append(echo_names)

        labels = _create_group_text_labels(
            full_cats,
            scores=None,
            class_names=full_names,
            is_crowd=[0 * len(boxes)],
        )
        
        self.overlay_instances(
            labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
        )

        return self.output

    def draw_instance_predictions(self, predictions: Instances):
        """
        Draw instance-level prediction results on an image.
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes.to(torch.device('cpu')) if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        
        if self.show_lesion:
            classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
            labels =  self.metadata.get("thing_classes", None)
        elif self.plain_cancer:
            classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
            labels = self.metadata.get("cancer_classes", None)
        else:
            classes = None
            labels =  [None]
        
        scores_shapes = predictions.shape_scores if predictions.has("shape_scores") else None
        classes_shapes = predictions.shape_pred.tolist() if predictions.has("shape_pred") else None
        shape_labels = self.metadata.get("shape_classes", None) if scores_shapes is not None else None

        scores_orient = predictions.orient_scores if predictions.has("orient_scores") else None
        classes_orient = predictions.orient_pred.tolist() if predictions.has("orient_pred") else None
        orient_labels = self.metadata.get("orientation_classes", None) if scores_orient is not None else None
        
        scores_margin = predictions.margin_scores if predictions.has("margin_scores") else None
        classes_margin  = predictions.margin_pred.tolist() if predictions.has("margin_pred") else None
        margin_labels = self.metadata.get("margin_classes", None) if scores_margin is not None else None

        scores_post = predictions.post_scores if predictions.has("post_scores") else None
        classes_post = predictions.post_pred.tolist() if predictions.has("post_pred") else None
        post_labels = self.metadata.get("posterior_classes", None) if scores_post is not None else None
        
        scores_echo = predictions.echo_scores if predictions.has("echo_scores") else None
        classes_echo = predictions.echo_pred.tolist() if predictions.has("echo_pred") else None
        echo_labels = self.metadata.get("echo_classes", None) if scores_echo is not None else None
        
        scores_cancer = predictions.cancer_scores if predictions.has("cancer_scores") else None
        classes_cancer = predictions.cancer_pred.tolist() if predictions.has("cancer_pred") else None
        cancer_labels = self.metadata.get("cancer_classes", None) if scores_cancer is not None else None

        
        full_cats = [classes, classes_cancer, classes_shapes, classes_orient, classes_margin, classes_post, classes_echo]
        full_names = [labels, cancer_labels, shape_labels, orient_labels, margin_labels, post_labels, echo_labels]
        full_scores = [scores, scores_cancer, scores_shapes, scores_orient, scores_margin, scores_post, scores_echo]
        

        full_cats = remove_nones_from_list(full_cats)
        full_scores = remove_nones_from_list(full_scores)
        full_names = remove_nones_from_list(full_names)

        labels = _create_group_text_labels(
                full_cats,
                scores=full_scores,
                class_names=full_names
            )
        
        
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks.cpu())
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if classes_cancer is None:
            colors = [[random.random(), random.random(), random.random()]  # Generates random RGB values
                      for c in classes]
            alpha = 0.8
        else:
            colors = [self.metadata.thing_colors[c] for c in classes_cancer]
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
    
    def draw_text(
        self,
        text: str,
        position: Tuple[float, float],
        *,
        font_size: Optional[int] = None,
        color: str = "g",
        horizontal_alignment: str = "center",
        rotation: int = 0,
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        if text.count('%') > 6:
            y = y - 80           
            
        self.output.ax.text(
            x-10,
            y-40,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output
    
def remove_nones_from_list(list_w_possible_nones: List[Optional[Any]]):
    new_lst = [item for item in list_w_possible_nones if item is not None]
    return new_lst

def _create_group_text_labels(classes: List[List[Optional[int]]], 
                              scores: List[List[Optional[float]]], 
                              class_names: List[List[Optional[str]]], 
                              is_crowd: Optional[List[List[Optional[bool]]]] = None):
    """
    Returns:
        list[str] or None
    """
    if classes is not None:
        complete_labels = ['' for i in list(range(len(classes[0])))]

        for x in list(range(len(classes))):
            if scores is not None:
                display_scores = []
                for y in list(range(len(classes[x]))):
                    if (classes[x][y] < 1) and not (any(name == 'lesion' for name in class_names[x])):
                        display_scores.append(1.0 - scores[x][y])
                    else:
                        display_scores.append(scores[x][y])
                lesion_label_list = _create_text_labels(classes[x], display_scores, class_names[x], is_crowd)
            else:
                lesion_label_list = _create_text_labels(classes[x], scores, class_names[x], is_crowd)
                

            for y in list(range(len(lesion_label_list))):
                if x == 3:
                    complete_labels[y] = complete_labels[y] + ', \n' + lesion_label_list[y]
                elif x > 0:
                    complete_labels[y] = complete_labels[y] + ', ' + lesion_label_list[y]
                else:
                    complete_labels[y] = lesion_label_list[y]

    return complete_labels

def _create_text_labels(classes: List[Optional[int]], 
                        scores: List[Optional[float]], 
                        
                        class_names: List[Optional[str]], is_crowd: Optional[List[Optional[bool]]] = None):
    """
    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    # if labels is not None and is_crowd is not None:
    #     labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels