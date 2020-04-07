# mmd

This repository contains the Pytorch implementation used for our submission

- [A Knowledge-Grounded Multimodal Search-Based Conversational Agent](https://arxiv.org/pdf/1810.11954.pdf) SCAI@EMNLP 2018
- [Improving Context Modelling in Multimodal Dialogue Generation](https://arxiv.org/pdf/1810.11955.pdf) INLG 2018

## Install

We used Python 2.7 and Pytorch 0.3 (0.3.1.post2) for our implementation.

We strongly encourage to use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or virtual environment.

The repo provides both yaml and requirements (use either of the file exported through conda/pip) to create the conda env.

```
conda create -n mmd python=2
source activate mmd
pip install -r requirements.txt
```

or

```
conda env create -n mmd -f mmd.yml
```

## Data

Download the data set from the [link](https://drive.google.com/drive/folders/1JOGHzideeAsmykMUQD3z7aGFg-M4QlE2) provided by Shah et al. (More information provided by the authors in their [webpage](https://amritasaha1812.github.io/MMD/download/) and [repo](https://github.com/amritasaha1812/MMD_Code)). Extract it in data folder

## Pre-processing

Run the bash scripts in `data_processing` folder to convert from json transcripts to the actual data for the model.
Running as `bash dialogue_data.sh` will call the python files in the same folder. Please manipulate the file paths accordingly in the shell script. Run the other shell scripts to extract KB related data for the model.

Alternatively you can also download the data for context of 2 and 5 (as well as pickled Shah et al. data) that we used for training from this [link](https://www.dropbox.com/s/yz31j1zd3vvjwrw/data.zip?dl=0).

Update: we are also providing the raw catalog, chat transcripts and metadata provided by Shah et al. [here](https://www.dropbox.com/s/s60owkgdv6glmuz/mmd_data.tar.gz?dl=0) All copyrights regarding the catalog and chat transcripts rests with them. Please contact them for further information. 

## Training

Run `bash train_and_translate.sh` for training as well as generating on the test set. This script at the end also computes the final metrics on the test set.

Structure:

```
train_and_translate.sh --> train.sh --> train.py --> modules/models.py
train_and_translate.sh --> translate.sh --> translate.py --> modules/models.py
train_and_translate.sh --> evaluation/nlg_eval.sh --> nlg_eval.py
```
This structure allows us to enter file paths only once while training and evaluating. Follow the steps on the screen to provide different parameters. My suggestion is to use Pycharm to better understand the structure and modularity.

Tune `IS_TRAIN` parameter if you have actually trained the model and want to just generate on final test set.

Hyperparameter tuning can be done by creating different config versions in `config` folder. Sample config versions are provided for reference.

## Metrics

For evaluation, we used the scripts provided by [nlg-eval](https://github.com/Maluuba/nlg-eval). (Sharma et al.)

In particular, we used their [functional api](https://github.com/Maluuba/nlg-eval#functional-api-for-the-entire-corpus) for getting evaluation metrics. Install the dependencies in the same or different conda environment.

`nlg_eval.sh` or `compute_metrics.sh` both bash script can be called depending upon the use case. See the structure above. nlg_eval.py is the main file using the funcitonal api.

## Citation

If you use this work, please cite it as
```
@inproceedings{agarwal2018improving,
  title={Improving Context Modelling in Multimodal Dialogue Generation},
  author={Agarwal, Shubham and Du{\v{s}}ek, Ond{\v{r}}ej and Konstas, Ioannis and Rieser, Verena},
  booktitle={Proceedings of the 11th International Conference on Natural Language Generation},
  pages={129--134},
  year={2018}
}

@inproceedings{agarwal2018knowledge,
  title={A Knowledge-Grounded Multimodal Search-Based Conversational Agent},
  author={Agarwal, Shubham and Du{\v{s}}ek, Ond{\v{r}}ej and Konstas, Ioannis and Rieser, Verena},
  booktitle={Proceedings of the 2018 EMNLP Workshop SCAI: The 2nd International Workshop on Search-Oriented Conversational AI},
  pages={59--66},
  year={2018}
}
```

Feel free to fork and contribute to this work. Please raise a PR or any related issues. Will be happy to help. Thanks.
