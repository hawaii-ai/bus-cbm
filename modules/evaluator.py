# Most code in this file is taken from detectron2.COCOEvaluator and pycocotools COCOeval, adapted for our purposes
# All credit for unmodified code and structure goes to the original authors.  
from collections import OrderedDict
from iopath.common.file_io import file_lock
import io, os, time, shutil, json, copy, datetime, itertools, contextlib, logging

import torch
import numpy as np

from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval

from confidenceinterval import roc_auc_score

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode,  RotatedBoxes, PolygonMasks, Instances

COCOeval_opt = COCOeval
logger = logging.getLogger(__name__)

# for minimal -> sigmoid(0.05) = 0.5125 (1), sigmoid(-0.05) = 0.4875 (0)
# for maximal - sigmoid(4.625) = 0.9903 (1), sigmoid(-4.625) = 0.0097 (0)

def instances_to_coco_json(instances: Instances, img_id: int):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []
    
    # get boxes and convert them to the form COCO expects 
    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results

def convert_to_coco_dict(dataset_name, concepts=None):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into COCO json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {
            v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
        }
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(
                        f"bbox has to be 1-dimensional. Got shape={bbox.shape}."
                    )
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                if isinstance(segmentation, list):
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):  # RLE
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
            else:
                # Computing areas using bounding boxes
                if to_bbox_mode == BoxMode.XYWH_ABS:
                    bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                else:
                    area = RotatedBoxes([bbox]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            coco_annotation["category_id"] = int(
                reverse_id_mapper(annotation["category_id"])
            )

            if concepts is not None:
                for concept_gt_name in concepts.keys():
                    concept_name_coco = 'region_' + concept_gt_name
                    coco_annotation[concept_name_coco] = int(annotation.get(concept_name_coco, -1))

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                seg = coco_annotation["segmentation"] = annotation["segmentation"]
                if isinstance(seg, dict):  # RLE
                    counts = seg["counts"]
                    if not isinstance(counts, str):
                        # make it json-serializable
                        seg["counts"] = counts.decode("ascii")

            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "categories": categories,
        "licenses": None,
    }
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict

def convert_to_coco_json(dataset_name, output_file, concepts=None, allow_cached=False):

    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(
                f"Converting annotations of dataset '{dataset_name}' to COCO format ...)"
            )
            coco_dict = convert_to_coco_dict(dataset_name, concepts)

            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)

class COCOevalMaxDets(COCOeval):
    """
    Modified version of COCOeval for evaluating AP with a custom
    maxDets (by default for COCO, maxDets is 100)
    """

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results given
        a custom value for  max_dets_per_image
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            # Evaluate AP using the custom limit on maximum detections per image
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class CBMCOCOeval(COCOevalMaxDets):
    def __init__(self, cocoGt=None, cocoDt=None, confThresh=None, concepts=None, iouType='segm'):
        super(COCOevalMaxDets, self).__init__(cocoGt, cocoDt, iouType)

        self.concept_mapping = concepts
        self.confThresh = confThresh

    # we only have one object category (lesion) so we only ever call this once 
    # nothing in the proposal matching logic needs to change when we're adjusting our concept predictions
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        dtm_concepts = dict()
        gtm_concepts = dict()
        for concept_gt_name in self.concept_mapping.keys():
            dtm_concepts[concept_gt_name] = np.zeros((T,D))
            gtm_concepts[concept_gt_name] = np.zeros((T,D))

        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
                    try:
                        for concept_gt_name in self.concept_mapping.keys():
                            gtm_concepts[concept_gt_name][tind,dind]  = gt[m]['region_' + concept_gt_name]
                            dtm_concepts[concept_gt_name][tind,dind]     = d['region_' + concept_gt_name]
                    except: 
                        gtm_concepts['cancer'][tind,dind]  = gt[m]['category_id']
                        dtm_concepts['cancer'][tind,dind]     = d['score']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))

        imgRes = {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtConcepts':   dtm_concepts, 
                'gtConcepts':   gtm_concepts,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
        }
 
        # store results for given image and category
        return imgRes

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        # creating arrays to hold results for each of the concept categories 
        concept_auc_results = dict()
        for concept_gt_name in self.concept_mapping.keys():
            concept_auc_results[concept_gt_name] = -np.ones((T,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        # we only have one category, so we only go through this part once 
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue

                    # get the concepts for the regions in dtm 
                    dtm_concepts = dict()
                    gtm_concepts = dict()

                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
                    # imgs = np.concatenate([e['image']*maxDet for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]
                    
                    try:
                        for concept_gt_name in self.concept_mapping.keys():
                            dtm_concepts[concept_gt_name]  = np.concatenate([e['dtConcepts'][concept_gt_name][:, 0:maxDet] for e in E], axis=1)[:,inds]
                            gtm_concepts[concept_gt_name]  = np.concatenate([e['gtConcepts'][concept_gt_name][:, 0:maxDet] for e in E], axis=1)[:,inds]
                    except: 
                        dtm_concepts['cancer']  = np.concatenate([e['dtConcepts']['cancer'][:, 0:maxDet] for e in E], axis=1)[:,inds]
                        gtm_concepts['cancer']  = np.concatenate([e['gtConcepts']['cancer'][:, 0:maxDet] for e in E], axis=1)[:,inds]
                    
                    

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue

                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
                    
                    # only evaluate the AUROC values when we're considering all areas
                    if a0 == 0:
                        for t in range(T):
                            for concept_gt_name in self.concept_mapping.keys():
                                detections = dtm_concepts[concept_gt_name][t, :][tps[t, :]]
                                truths = gtm_concepts[concept_gt_name][t, :][tps[t, :]]
                                concept_auc_results[concept_gt_name][t,k,a,m] = self.computeAUROC(concept_gt_name, maxDet, t, np.ravel(detections), np.ravel(truths))

                                
        self.eval = {
            'params': p,
            'counts': [T, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'auroc': concept_auc_results,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results given
        a custom value for  max_dets_per_image
        """

        def _summarize(iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "AUROC" 
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            # dimension of each auc element is the same as recall: [TxKxAxM]
            for concept_gt_name in self.concept_mapping.keys():
                s = self.eval["auroc"][concept_gt_name]
                typeStr = concept_gt_name
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
                if len(s[s > -1]) == 0:
                    mean_s = -1
                else:
                    mean_s = np.mean(s[s > -1])
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((6,))
            # Evaluate AUROC using the custom limit on maximum detections per image
            stats[0] = _summarize(maxDets=self.params.maxDets[2])
            stats[1] = _summarize(iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(maxDets=self.params.maxDets[0])
            stats[4] = _summarize(maxDets=self.params.maxDets[1])
            stats[5] = _summarize(maxDets=self.params.maxDets[2])
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        self.stats = summarize()

    def __str__(self):
        self.summarize()

    def computeAUROC(self, concept_name, max_dets, thresh, dt_concepts, gt_concepts):
        """
        Compute AUROC for detection concepts based on a confidence threshold.
        """
        try:
            auc, ci = roc_auc_score(gt_concepts,
                            dt_concepts,
                            confidence_level=0.95)
            print('# of ', concept_name, ' concepts with max_det ', str(max_dets), 'at IOU thresh ', str(thresh), ': ', str(len(dt_concepts)))
            print('95% CI: ', str(ci))
        except AssertionError:
            print('# of ', concept_name, ' concepts with max_det ', str(max_dets), 'at IOU thresh ', str(thresh), ': ', str(len(dt_concepts)))
            print(dt_concepts)
            print(gt_concepts)
            auc = -1
        return auc        

class CBMCOCOEvaluator(COCOEvaluator): 
    def __init__(self,
                 dataset_name,
                 concept_mapper=None,
                 conf_thresh=0.5,
                 tasks=None,
                 max_dets_per_image=None,
                 distributed=True,
                 output_dir=None,
                 use_fast_impl=False,
                 **kwargs):
        super(CBMCOCOEvaluator, self).__init__(dataset_name, tasks, distributed, output_dir, use_fast_impl=use_fast_impl, max_dets_per_image=max_dets_per_image)
        # dictionary which maps the name of the concept to the name of the label which is 
        # found in the COCO file 
        self.concept_mapping = concept_mapper
        # object minimum probability to consider an instance to compute AUROC
        self.confThresh = conf_thresh
        convert_to_coco_json(dataset_name, self._metadata.json_file, concepts=self.concept_mapping, allow_cached=False)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
    
    def instances_to_coco_json(self, instances, img_id):
        """
        Dump an "Instances" object to a COCO-format json that's used for evaluation.

        Args:
            instances (Instances):
            img_id (int): the image id

        Returns:
            list[dict]: list of json annotations in COCO format.
        """
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        
        if self.concept_mapping is not None:
            concepts = dict()
            try:
                for concept_gt_name, concept_score_name in self.concept_mapping.items():
                    concepts['region_' + concept_gt_name] = instances.get(concept_score_name + '_scores').tolist()
            except: pass

        has_mask = instances.has("pred_masks")
        if has_mask:
            # use RLE to encode the masks, because they are too large and takes memory
            # since this evaluator stores outputs of the entire dataset
            rles = [
                mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                for mask in instances.pred_masks
            ]
            for rle in rles:
                # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                # json writer which always produces strings cannot serialize a bytestream
                # unless you decode it. Thankfully, utf-8 works out (which is also what
                # the pycocotools/_mask.pyx does).
                rle["counts"] = rle["counts"].decode("utf-8")

        has_keypoints = instances.has("pred_keypoints")
        if has_keypoints:
            keypoints = instances.pred_keypoints

        results = []
        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }

            if self.concept_mapping is not None:
                for concept, scores in concepts.items():
                    result[concept] = scores[k]
                     
            if has_mask:
                result["segmentation"] = rles[k]
            if has_keypoints:
                # In COCO annotations,
                # keypoints coordinates are pixel indices.
                # However our predictions are floating point coordinates.
                # Therefore we subtract 0.5 to be consistent with the annotation format.
                # This is the inverse of data loading logic in `datasets/coco.py`.
                keypoints[k][:, :2] -= 0.5
                result["keypoints"] = keypoints[k].flatten().tolist()
            results.append(result)
        return results

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if self.concept_mapping is not None:
                prediction["concept_instances"] = self.instances_to_coco_json(instances, input["image_id"])
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids, eval_concepts=False)
        # run AUROC evaluation for concepts 
        if self.concept_mapping is not None:
            self._eval_predictions(predictions, img_ids=img_ids, eval_concepts=True)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions, eval_concepts=False, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        if eval_concepts:
            coco_results = list(itertools.chain(*[x["concept_instances"] for x in predictions]))
        else:
            coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        # this can stay the same for the concepts, since they're not object classes but attributes
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            if eval_concepts: task = 'segm'
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    eval_concepts=eval_concepts,
                    confThresh = self.confThresh,
                    concept_mapping=self.concept_mapping,
                    cocoeval_fn=COCOeval_opt if self._use_fast_impl else COCOeval,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

            # only evaluate concept results on the segmentation masks 
            if eval_concepts: break
    
def _evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    iou_type,
    kpt_oks_sigmas=None,
    eval_concepts=False,
    confThresh=None,
    concept_mapping=None,
    cocoeval_fn=COCOeval_opt,
    img_ids=None,
    max_dets_per_image=None,
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = cocoeval_fn(coco_gt, coco_dt, iou_type)
    
    # For COCO, the default max_dets_per_image is [1, 10, 100].
    if max_dets_per_image is None:
        max_dets_per_image = [1, 10, 100]  # Default from COCOEval
        if eval_concepts:
            max_dets_per_image = [1, 10, 4] 
            coco_eval = CBMCOCOeval(coco_gt, coco_dt, confThresh=confThresh, concepts=concept_mapping, iouType=iou_type)
    else:
        assert (
            len(max_dets_per_image) >= 3
        ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
        # In the case that user supplies a custom input for max_dets_per_image,
        # apply COCOevalMaxDets to evaluate AP with the custom input.
        if max_dets_per_image[2] != 100:
            max_dets_per_image = [1, 10, 4] 
            if eval_concepts:
                coco_eval = CBMCOCOeval(coco_gt, coco_dt, confThresh=confThresh, concepts=concept_mapping, iouType=iou_type)
            else:
                coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)
    if iou_type != "keypoints":
        coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    if iou_type == "keypoints":
        # Use the COCO default keypoint OKS sigmas unless overrides are specified
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params, "kpt_oks_sigmas"), "pycocotools is too old!"
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        # COCOAPI requires every detection and every gt to have keypoints, so
        # we just take the first entry from both
        num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
        num_keypoints_gt = len(next(iter(coco_gt.anns.values()))["keypoints"]) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
            f"[COCOEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
            f"Ground truth contains {num_keypoints_gt} keypoints. "
            f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
            "They have to agree with each other. For meaning of OKS, please refer to "
            "http://cocodataset.org/#keypoints-eval."
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
