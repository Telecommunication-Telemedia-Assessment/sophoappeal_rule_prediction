#!/usr/bin/env python3
"""
the following code is partially based on https://keras.io/guides/transfer_learning/
"""
import os
import argparse
import json
import sys
from pprint import pprint

import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from tqdm.keras import TqdmCallback
from keras.callbacks import ModelCheckpoint

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# this enables dynamic growing of gpu memory
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


models = {
    "Xception": {
        "__init__": keras.applications.Xception,
        "preprocess": tf.keras.applications.xception.preprocess_input,
    },
    "DenseNet201": {
        "__init__": keras.applications.DenseNet201,
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "VGG19": {
        "__init__": keras.applications.VGG19,
        "preprocess": tf.keras.applications.vgg19.preprocess_input,
    },
    "VGG16": {
        "__init__": keras.applications.VGG16,
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
    },
    "MobileNetV2": {
        "__init__": keras.applications.MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "MobileNet": {
        "__init__": keras.applications.MobileNet,
        "preprocess": tf.keras.applications.mobilenet.preprocess_input,
    },
    "ResNet50": {
        "__init__": keras.applications.ResNet50,
        "preprocess": tf.keras.applications.resnet.preprocess_input,
    },
    "ResNet101": {
        "__init__": keras.applications.ResNet101,
        "preprocess": tf.keras.applications.resnet.preprocess_input,
    },
    "ResNet152": {
        "__init__": keras.applications.ResNet152,
        "preprocess": tf.keras.applications.resnet.preprocess_input,
    },
    "ResNet50V2": {
        "__init__": keras.applications.ResNet50V2,
        "preprocess": tf.keras.applications.resnet_v2.preprocess_input,
    },
    "ResNet101V2": {
        "__init__": keras.applications.ResNet101V2,
        "preprocess": tf.keras.applications.resnet_v2.preprocess_input,
    },
    "ResNet152V2": {
        "__init__": keras.applications.ResNet152V2,
        "preprocess": tf.keras.applications.resnet_v2.preprocess_input,
    },
    "InceptionV3": {
        "__init__": keras.applications.InceptionV3,
        "preprocess": tf.keras.applications.inception_v3.preprocess_input,
    },
    "InceptionResNetV2": {
        "__init__": keras.applications.InceptionResNetV2,
        "preprocess": tf.keras.applications.inception_resnet_v2.preprocess_input,
    },
    "DenseNet121": {
        "__init__": keras.applications.DenseNet121,
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "DenseNet169": {
        "__init__": keras.applications.DenseNet169,
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "DenseNet201": {
        "__init__": keras.applications.DenseNet201,
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "NASNetMobile": {
        "__init__": keras.applications.NASNetMobile,
        "preprocess": tf.keras.applications.nasnet.preprocess_input,
    },
    "NASNetLarge": {
        "__init__": keras.applications.NASNetLarge,
        "preprocess": tf.keras.applications.nasnet.preprocess_input,
    },
}


def build_transfer_learning_model(modelname):

    assert modelname in models
    # get model
    model = models[modelname]

    input_shape = (224, 224, 3)

    base_model = model["__init__"](
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=input_shape,
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=input_shape)

    # we need to use the DNN specific preprocessing of the image
    x = model["preprocess"](inputs)

    x = base_model(x, training=False)

    # x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1, activation="sigmoid")(
        x
    )  # based on https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/

    model = keras.Model(inputs, outputs, name="cust_" + modelname)

    return model


def check_model_creation():
    # check model access
    print(build_transfer_learning_model("DenseNet201").summary())


def read_images(directory, subset, input_shape, batch_size=32):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=input_shape[0:2],
        shuffle=True,
        seed=42,
        validation_split=0.1,  # 10% validation
        subset=subset,
        interpolation="bilinear",
    )


def train_and_evaluate_model(modelname, training, validation, results_folder, models_folder, epochs=100):
    model = build_transfer_learning_model(modelname)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    checkpoint = ModelCheckpoint(
        os.path.join(models_folder, modelname + "_best_model.hdf5"),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        mode="auto",
        save_freq="epoch",
    )

    res = model.fit(
        training, epochs=epochs, validation_data=validation, verbose=0, callbacks=[TqdmCallback(verbose=2), checkpoint]
    )

    y_pred = [float(x) for x in res.model.predict(validation).flatten()]

    y_truth = []
    for y, l in validation:
        y_truth.extend(l.numpy())
    y_truth = [int(x) for x in np.array(y_truth).flatten()]

    result = res.history
    result["model"] = modelname
    result["pred"] = y_pred
    result["truth"] = y_truth
    result["validation_files"] = validation.file_paths

    with open(os.path.join(results_folder, modelname + ".json"), "w") as xfp:
        json.dump(result, xfp, sort_keys=True, indent=4)


def main(_):
    # argument parsing
    parser = argparse.ArgumentParser(
        description="dnn evaluation", epilog="stg7 2021", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", default="data/rule_of_thirds/", type=str, help="data to be used")
    parser.add_argument("--models_folder", default="models", type=str, help="folder to store best models")
    parser.add_argument("--results_folder", default="results", type=str, help="folder to store results of best models")

    a = vars(parser.parse_args())

    # check if gpu is accessible
    print(tf.config.list_physical_devices("GPU"))

    os.makedirs(a["storage_folder"], exist_ok=True)

    directory = "data/rule_of_thirds/"
    batch_size = 32
    input_shape = (224, 224, 3)

    training = read_images(a["data"], "training", input_shape, batch_size)

    validation = read_images(a["data"], "validation", input_shape, batch_size)
    print(f"training and validation samples: {len(training)},{len(validation)}")

    os.makedirs(a["results_folder"], exist_ok=True)
    os.makedirs(a["models_folder"], exist_ok=True)

    for modelname in models:
        print(modelname)
        train_and_evaluate_model(modelname, training, validation, a["results_folder"], a["models_folder"], epochs=1)
        break


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
