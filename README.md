# Photo Rules

The following repository consists code and data to the paper 


    Göring et al. 2021: "Rule of Thirds and Simplicity for Image Aesthetics using Deep Neural Networks"


The overall idea is that e.g. in the following papers:

* Mai et al. 2012: "Detecting Rule of Simplicity from Photos" and
* Mai et al. 2011: "Rule of Thirds Detection from Photograph"

approaches to detect simplicity and rule of thirds for images are proposed, both algorithms use saliency models to estimate the final classification decision.

In this repo, the published labeled datasets are used to re-train state of the art DNNs for the same tasks.

The overall data and models can be found here: [models and data](https://zenodo.org/record/5140038#.YQAd_HUzYW0).
The included script `download.sh` will automatically add the data and models to this repository.

# Requirements 

* the software is only tested on Ubuntu 20.04
* python3, pip3

Afterwards run: `pip3 install -r requirements.txt` to install dependencies, you may create a python virtual environment before and activate it.

Before you can use the models, you need to run `./download.sh`, you may remove the download for the images (they are only required for the training part of the experiment).


# Usage

* `evaluate_models.py` generic script to train such DNN models
* `rule_of_thirds.sh` training script for rule of thirds prediction
* `simplicity.sh` training script for image simplicity

Both training steps are not required if you just want to use the described models, for this you can use 

* `predict.py` and
* `predict_all.sh` that demonstrates for the best two models and for both rules the usage.

# Image Annotations
In one of the evaluation experiment the AVT Image DB has been used (see [AVT Image DB](https://github.com/Telecommunication-Telemedia-Assessment/image_compression)), annotations regarding image simplicity and rule of thirds can be found in the folder [evaluation/data/avt_image_db/annotations/](evaluation/data/avt_image_db/annotations/).


## Acknowledgments
If you use this software, data or models in your research, please include a link to the repository and reference the following paper.

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




