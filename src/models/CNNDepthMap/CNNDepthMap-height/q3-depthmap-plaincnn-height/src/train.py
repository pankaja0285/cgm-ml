import os
import pickle
import random

import glob2 as glob
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras import callbacks

from config import CONFIG
from constants import REPO_DIR
from model import create_cnn
from preprocessing import preprocess_depthmap, preprocess_targets

#initializing neptune
import neptune
neptune.init(
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNjVmMWFjYmUtNzI4Yy00NTJiLTk3MjItMzJlZWE3ZjFjYmJlIn0=",
    project_qualified_name="cgm-ml/height"  # You can change to your project-height,weight
)
#add a detailed description of the experiment
description = "plaincnn for height prediction on depthmap"

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
    experiment = Experiment(workspace, "training-junkyard")
    run = experiment.start_logging(outputs=None, snapshot_directory=None)

    # Get dataset.
    print("Accessing dataset...")
    dataset_name = "anon-depthmap-mini"
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

# Shuffle and split into train and validate.
random.shuffle(qrcode_paths)
split_index = int(len(qrcode_paths) * 0.8)
qrcode_paths_training = qrcode_paths[:split_index]
qrcode_paths_validate = qrcode_paths[split_index:]
qrcode_paths_activation = random.choice(qrcode_paths_validate)
qrcode_paths_activation = [qrcode_paths_activation]

del qrcode_paths

# Show split.
print("Paths for training:")
print("\t" + "\n\t".join(qrcode_paths_training))
print("Paths for validation:")
print("\t" + "\n\t".join(qrcode_paths_validate))
print("Paths for activation:")
print("\t" + "\n\t".join(qrcode_paths_activation))

print(len(qrcode_paths_training))
print(len(qrcode_paths_validate))

assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0


def get_depthmap_files(paths):
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
    return pickle_paths


# Get the pointclouds.
print("Getting depthmap paths...")
paths_training = get_depthmap_files(qrcode_paths_training)
paths_validate = get_depthmap_files(qrcode_paths_validate)
paths_activate = get_depthmap_files(qrcode_paths_activation)

del qrcode_paths_training
del qrcode_paths_validate
del qrcode_paths_activation

print("Using {} files for training.".format(len(paths_training)))
print("Using {} files for validation.".format(len(paths_validate)))
print("Using {} files for validation.".format(len(paths_activate)))


# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
    def py_load_pickle(path, max_value):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = preprocess_depthmap(depthmap)
        depthmap = depthmap / max_value
        depthmap = tf.image.resize(depthmap, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
        targets = preprocess_targets(targets, targets_indices)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1))
    targets.set_shape((len(targets_indices,)))
    return depthmap, targets


def tf_flip(image):
    image = tf.image.random_flip_left_right(image)
    return image


# Parameters for dataset generation.
targets_indices = [0]  # 0 is height, 1 is weight.

# Create dataset for training.
paths = paths_training
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_norm = dataset_norm.shuffle(CONFIG.SHUFFLE_BUFFER_SIZE)
dataset_training = dataset_norm
del dataset_norm

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validation = dataset_norm
del dataset_norm

# Create dataset for activation
paths = paths_activate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_activation = dataset_norm
del dataset_norm

# Note: Now the datasets are prepared.

#parameters info to log into neptune
training_params = CONFIG

# Create the model.
input_shape = (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1)
model = create_cnn(input_shape, dropout=True)
model.summary()
model_params = {
    "architecture": "plaincnn",
    "activation": "relu",
    "n_conv_layers": "12",
    "n_dense_layers": "1024+128"}

# Get ready to add callbacks.
training_callbacks = []

# Pushes metrics and losses into the run on AzureML.


class AzureLogCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                run.log(key, value)


training_callbacks.append(AzureLogCallback())

#creating the neptune experiment
neptune.create_experiment(
    params={**model_params, **training_params},
    # dataset type, dataset version,quarter,user story
    tags=['depthmap', 'anon-mini-95k', 'Q4', 'US'],
    description=description
)
#logging the model to the neptune
model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

# Pushes metrics and losses into the run on Neptune


class NeptuneLogger(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'epoch_{log_name}', log_value)


training_callbacks.append(NeptuneLogger())

# Add TensorBoard callback.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=0,
    write_graph=True,
    write_grads=False,
    write_images=True,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None,
    embeddings_data=None,
    update_freq="epoch"
)
training_callbacks.append(tensorboard_callback)

# Add checkpoint callback.
best_model_path = str(REPO_DIR / 'data/outputs/best_model.h5')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
training_callbacks.append(checkpoint_callback)

optimizer = tf.keras.optimizers.Nadam(learning_rate=CONFIG.LEARNING_RATE)
neptune.append_tags("optimizer:Nadam")

# Compile the model.
model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)
neptune.append_tags("loss:mse")

# Train the model.
model.fit(
    dataset_training.batch(CONFIG.BATCH_SIZE),
    validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
    epochs=CONFIG.EPOCHS,
    callbacks=training_callbacks,
    verbose=2
)
if os.path.exists(best_model_path):
    neptune.log_artifact(best_model_path)

# Done.
run.complete()
