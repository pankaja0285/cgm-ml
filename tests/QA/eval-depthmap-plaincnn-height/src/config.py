class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


CONFIG = dotdict(dict(
    SPLIT_SEED=0,
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    BATCH_SIZE=256,
    NORMALIZATION_VALUE=7.5,
    EVAL_EXPERIMENT_NAME = "QA-pipeline",
    EVAL_DATASET_NAME = 'anon-depthmap-testset',
    MODEL_PATH = 'best_model.h5',
    EVAL_MODEL_NAME = 'q3-depthmap-plaincnn-height-100-95k',
    EVAL_RUN_NO ='_front_run_03',

    FAST_RUN = True,
    SMALL_EVAL_SIZE = 20,
    CSV_OUT_PATH = 'result2.csv',

    
    # Parameters for dataset generation.
    TARGET_INDEXES=[0]  # 0 is height, 1 is weight.
))
