from detectron2.config import CfgNode as CN 

def add_cbm_config(cfg):
    cfg.MODEL.CBM  = CN()
    cfg.MODEL.CBM.CONCEPTS = ("shape", "margin", "orientation", "echo", "posterior")
    cfg.MODEL.CBM.USE_SIGMOID = False
    cfg.MODEL.CBM.CANCER_ON = True 
    cfg.MODEL.CBM.SIDE_CHANNEL = False # Is optionally true when CANCER_ON = True
    cfg.MODEL.CBM.LINEAR = False # is optionally true when CANCER_ON = True, SIDE_CHANNEL must be False when this is True

    cfg.SEED = 1170
    cfg.MODEL.WEIGHTS = "R-101.pkl"
    
    cfg.MODEL.ROI_HEADS.NAME = "CBMStandardROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    cfg.MODEL.ROI_CANCER_HEAD = CN()
    cfg.MODEL.ROI_CANCER_HEAD.NAME = "CBMCancerHead"
    cfg.MODEL.ROI_CANCER_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_CANCER_HEAD.NUM_CONV = (256, 32) # number of filters in conv layers
    cfg.MODEL.ROI_CANCER_HEAD.NUM_FC = 512 # number of neurons in fc layers
 
    
def add_uhcc_config(cfg):
    cfg.DATASETS.TRAIN = ("training", )
    cfg.DATASETS.TEST = ()
    cfg.DATASETS.VAL = ("validation",)

    cfg.MODEL.PIXEL_MEAN = [55.99382249859551, 51.79007112271128, 49.38587209524078]
    cfg.MODEL.PIXEL_STD = [34.39853303014893, 32.49610343564593, 31.714134849812442]
    
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
