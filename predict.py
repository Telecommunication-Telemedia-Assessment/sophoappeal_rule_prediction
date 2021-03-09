#!/usr/bin/env python3
import os
import argparse
import json
import sys
import math
import glob
from pprint import pprint

import pandas as pd
import numpy as np

# deactivate GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import tensorflow as tf
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


from evaluate_models import build_transfer_learning_model


# def predict(image_batch):
#     for image_filename in sorted(validation.file_paths):
#         image = img_to_array(load_img(
#             image_filename,
#             color_mode='rgb',
#             target_size=input_shape,
#             interpolation='bilinear'
#         ))
#         y_truth.append(get_label(image_filename))
#         y_pred.append(res.model.predict(np.array([image])))
#         validation_files.append(image_filename)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(_):
    # argument parsing
    parser = argparse.ArgumentParser(description='predict',
                                     epilog="stg7 2021",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_folder", type=str, default="data/avt_image_db/jpg/", help="folder where images are stored")
    parser.add_argument("--report_name", type=str, default="prediction_rule_of_thirds.json", help="database that is used for validation")
    parser.add_argument("--model", type=str, default="rule_of_thirds/models/ResNet152_best_model.hdf5", help="model to be used")

    a = vars(parser.parse_args())

    assert(os.path.isfile(a["model"]))
    assert(os.path.isdir(a["image_folder"]))

    modelname = os.path.basename(os.path.splitext(a["model"])[0]).replace("_best_model", "")

    model = build_transfer_learning_model(modelname)

    model.load_weights(a["model"])

    print(model.summary())
    input_shape = model.layers[0].input_shape[0][1:]

    results = []
    j = 1
    images = list(glob.glob(a["image_folder"] + "/*"))
    batch_size = 64
    num_batches = math.ceil(len(images) / batch_size)

    for image_group in chunks(images, batch_size):
        print(f"predict batch {j} / {num_batches}")
        batch = []
        for image_filename in image_group:
            image = img_to_array(load_img(
                image_filename,
                color_mode='rgb',
                target_size=input_shape,
                interpolation='bilinear'
            ))

            batch.append(image)
        prediction = model.predict(np.array(batch))
        for i, image in enumerate(image_group):
            results.append({
                "image": image,
                "prediction": float(prediction[i][0])
            })
        j += 1
    print("done")
    with open(a["report_name"], "w") as xfp:
        json.dump(results, xfp, indent=4)



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
