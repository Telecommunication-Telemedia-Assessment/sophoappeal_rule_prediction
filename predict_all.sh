#!/bin/bash

#./predict.py --image_folder data/avt_image_db/jpg/ \
#    --report_name prediction_rule_of_thirds.json \
#    --mode rule_of_thirds/models/ResNet152_best_model.hdf5

./predict.py --image_folder data/avt_image_db/jpg/ \
    --report_name prediction_simplicity.json \
    --mode image_simplicity/models/DenseNet121_best_model.hdf5

