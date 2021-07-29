# Rule of Thirds and Simplicity Prediction for Photos

The following repository consists of code and data to reproduce the paper 


    Göring et al. 2021: "Rule of Thirds and Simplicity for Image Aesthetics using Deep Neural Networks"


The overall idea is that e.g. in the following papers:

* Mai et al. 2012: "Detecting Rule of Simplicity from Photos" and
* Mai et al. 2011: "Rule of Thirds Detection from Photograph"

approaches to detect simplicity and rule of thirds for images are proposed, both algorithms use saliency models to estimate the final classification decision.

In this repo, the published labeled datasets are used to re-train state-of-the-art DNNs for the same tasks.

The overall data and models can be found here: [models and data](https://zenodo.org/record/5140038#.YQAd_HUzYW0).
The included script `download.sh` will automatically add the data and models to this repository.

# Requirements 
The software is only tested on **Ubuntu 20.04** to run it you need to install:

* python3, python3-pip, python3-venv

You may create a python virtual environment and activate it with the following commands.
```bash
python3 -m venv env
source env/bin/activate
```
Afterward run: `pip3 install -r requirements.txt` to install all dependencies.

The **virtual environment is highly recommended**, because the tensorflow and keras versions changed and some incompatibilities using newer versions may occur.

Before you can use the models, you need to run `./download.sh`, you may remove the download for the images (they are only required for the training part of the experiment).

## Usage

* `evaluate_models.py` generic script to train such DNN models
* `rule_of_thirds.sh` training script for the rule of thirds prediction
* `simplicity.sh` training script for image simplicity

Both training steps are not required if you just want to use the described models, for this, you can use 

* `predict.py` and
* `predict_all.sh` that demonstrates for the best two models and for both rules the usage.

# Image Annotations
In one of the evaluation experiments the [AVT Image DB](https://github.com/Telecommunication-Telemedia-Assessment/image_compression) has been used, annotations regarding image simplicity and rule of thirds for this dataset can be found in the folder [evaluation/data/avt_image_db/annotations/](evaluation/data/avt_image_db/annotations/).


## Evaluation
The Jupyter notebooks for the evaluation figures and tables of the paper can be found in the folder `evaluation`.
To run these notebooks, you need jupyter and some more dependencies installed, e.g. scikit-learn, scipy, pandas, numpy.


## Developers
* [Steve Göring](https://github.com/stg7)

If you like this software you can also [donate me a :coffee:](https://ko-fi.com/binarys3v3n).


## Acknowledgments
If you use this software, data, or models in your research, please include a link to the repository and reference the following paper.

```
@inproceedings{goering2021rules,
  author={Steve {G{\"o}ring} and Alexander Raake},
  title="Rule of Thirds and Simplicity for Image Aesthetics using Deep Neural Networks",
  year={2021},
  booktitle={2021 IEEE 23st International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  organization={IEEE}
}
```

## License
GNU General Public License v3. See [LICENSE](LICENSE) file in this repository.

