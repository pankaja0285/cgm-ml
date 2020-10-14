import os
import pickle
import random
import pandas as pd
import numpy as np
import glob2 as glob
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
#from tensorflow.keras import callbacks

from config import CONFIG
from constants import REPO_DIR
from tensorflow.keras.models import load_model

import constants
import utils


# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
    def py_load_pickle(path, max_value):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = utils.preprocess_depthmap(depthmap)
        depthmap = depthmap / max_value
        depthmap = tf.image.resize(depthmap, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
        targets = utils.preprocess_targets(targets, CONFIG.TARGET_INDEXES)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1))
    targets.set_shape((len(CONFIG.TARGET_INDEXES,)))
    return depthmap, targets

def get_height_prediction(MODEL_PATH, dataset_evaluation):
    model = load_model(MODEL_PATH)
    predictions = model.predict(dataset_evaluation.batch(CONFIG.BATCH_SIZE))
    prediction_list = np.squeeze(predictions)
    return prediction_list


if __name__ == "__main__":

    # Make experiment reproducible
    tf.random.set_seed(CONFIG.SPLIT_SEED)
    random.seed(CONFIG.SPLIT_SEED)

    # Get the current run.
    run = Run.get_context()

    # Offline run. Download the sample dataset and run locally. Still push results to Azure.
    if(run.id.startswith("OfflineRun")):
        print("Running in offline mode...")

        # Access workspace.
        print("Accessing workspace...")
        workspace = Workspace.from_config()
        experiment = Experiment(workspace, CONFIG.EVAL_EXPERIMENT_NAME)
        run = experiment.start_logging(outputs=None, snapshot_directory=None)

        # Get dataset.
        print("Accessing dataset...")
        dataset_name = CONFIG.EVAL_DATASET_NAME
        dataset_path = str(REPO_DIR / "data" / dataset_name)
        if not os.path.exists(dataset_path):
            dataset = workspace.datasets[dataset_name]
            dataset.download(target_path=dataset_path, overwrite=False)

    # Online run. Use dataset provided by training notebook.
    else:
        print("Running in online mode...")
        experiment = run.experiment
        workspace = experiment.workspace
        dataset_path = run.input_datasets["dataset"]

    # Get the QR-code paths.
    dataset_path = os.path.join(dataset_path, "scans")
    print("Dataset path:", dataset_path)
    #print(glob.glob(os.path.join(dataset_path, "*"))) # Debug
    print("Getting QR-code paths...")
    qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
    print("qrcode_paths: ", len(qrcode_paths))
    assert len(qrcode_paths) != 0

    if CONFIG.FAST_RUN and len(qrcode_paths) > CONFIG.SMALL_EVAL_SIZE:
        qrcode_paths = qrcode_paths[:CONFIG.SMALL_EVAL_SIZE]
        print("Executing on {} qrcodes for FAST RUN".format(CONFIG.SMALL_EVAL_SIZE))

    print("Paths for evaluation:")
    print("\t" + "\n\t".join(qrcode_paths))

    print(len(qrcode_paths))

    # Get the pointclouds.
    print("Getting depthmap paths...")
    paths_evaluation = utils.get_depthmap_files(qrcode_paths)
    del qrcode_paths

    print("Using {} artifact files for evaluation.".format(len(paths_evaluation)))

    # Create dataset for training.
    paths = paths_evaluation
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
    dataset_norm = dataset_norm.cache()
    dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
    dataset_evaluation = dataset_norm
    del dataset_norm

    prediction_list1 = get_height_prediction(CONFIG.MODEL_PATH, dataset_evaluation)

    print(prediction_list1)

    qrcode_list, scantype_list, artifact_list, prediction_list, target_list = utils.get_column_list(
        paths_evaluation, 
        prediction_list1)

    df = pd.DataFrame({
        'qrcode':qrcode_list,
        'artifact':artifact_list,
        'scantype':scantype_list,
        'GT': target_list,
        'predicted':prediction_list
        }, columns = constants.COLUMNS)

    df['GT'] = df['GT'].astype('float64')
    df['predicted'] = df['predicted'].astype('float64')

    MAE = df.groupby(['qrcode', 'scantype']).mean()
    print(MAE)

    MAE['error'] = MAE.apply(utils.avgerror, axis=1)
    #MAE

    complete_name = CONFIG.EVAL_MODEL_NAME + CONFIG.EVAL_RUN_NO
    print(complete_name)

    print("Saving the results")
    utils.calculate_and_save_results(MAE, complete_name, CONFIG.CSV_OUT_PATH)

    # Done.
    run.complete()
