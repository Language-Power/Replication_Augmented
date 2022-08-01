[Home](./README.md) | [Data](./datasets/README.md) | [Package](./Core/README.md) | [Performance Scores](./AugmentedSocialScientist/docs/pages/saturation.md) | [Predictions](./AugmentedSocialScientist/docs/pages/train_predict.md) | [Results](./AugmentedSocialScientist/docs/pages/analysis.md)

# The Augmented Social Scientist

This repository is a replication and data file for the paper "The Augmented Social Scientist. Using Sequential Transfer Learning to Annotate Texts Faster and More Accurately" by [Salomé Do](https://sally14.github.io), [Étienne Ollion](https://ollion.cnrs.fr/english/) and [Rubing Shen](https://rubingshen.github.io).

We have also written a easy-to-use package to train BERT-family models for text classification. Check our [Google Colab tutorial](https://colab.research.google.com/drive/132_oDik-SOWve31tZ8D1VOx1Sj_Cyzn7?usp=sharing).



# 1. Data

The data we use for the paper is grouped in the folder `datasets`. It consists in newspaper articles from the French daily *Le Monde*, and of annotations for each of the two tasks, by three groups of annotators.

For more details, see [here](./datasets/README.md).

**:exclamation: The content of newspaper articles are protected by copyright. We have removed them from the public repo.**


# 2. Package

The folder `Core` is a wrapper for the [HuggingFace Transformers](https://huggingface.co/transformers/index.html) library. This wrapper implements some higher-level functions: formatting data, training models, prediction... 

## Requirements

**Software**

- Unix-based OS (OSX/Linux)
- A recent version of pip
- [Optionnal] Conda (we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html)). Conda enables installing all needed packages in a new, clean environment separated from the user's base environment. 

**Hardware**
- GPU : we use an Nvidia V100, which has 32Go of RAM. The batch sizes of our models were created so that they make full use of the graphic cards. 

## Installing the required packages

```bash
cd Core
conda create --name AugmentedSocialScientist python pip jupyter
conda activate AugmentedSocialScientist
pip install .
cd ..
```
For more details, see [here](./Core/README.md).



# 3. Preparation

## 3.1 Performance Scores

For Sample-Efficiency Curves (Figure 3) and summary tables of scores (Table 2; Tables B1, B2, B3 in the Appendix), we need to run the training script several times for each training set size. 

To obtain score for just one attempt:

 ```bash
conda activate AugmentedSocialScientist
python AugmentedSocialScientist/saturation/off/trial_off.py
python AugmentedSocialScientist/saturation/endoexo/trial_endoexo.py
 ```
As the subsampling is random, we runned 5 to 20 experiments for each task, which means we runned 5 to 20 times the above command.

For more details, see [here](./AugmentedSocialScientist/docs/pages/saturation.md).

In Table 2, we have also trained models which do not benefit from pre-training for comparison: a SVM model for the Policy/Politics task and a LSTM model for Off-the-Record task. Their performances are assessed in [AugmentedSocialScientist/train/train_predict_endoexo_SVM.ipynb](AugmentedSocialScientist/train/train_predict_endoexo_SVM.ipynb) and in [AugmentedSocialScientist/train/train_predict_off_LSTM.ipynb](AugmentedSocialScientist/train/train_predict_off_LSTM.ipynb).



## 3.2 Predictions

For the evolution of both Politics and Policy indicators (Figure 4), we need to save a trained model and use it to produce predictions on each sentence in the entire corpus. 
 
- The notebook [AugmentedSocialScientist/train/train_predict_endoexo.ipynb](AugmentedSocialScientist/train/train_predict_endoexo.ipynb) trains and saves a CamemBERT model for the Policy/Politics task.
- The notebook [AugmentedSocialScientist/train/train_predict_off.ipynb](AugmentedSocialScientist/train/train_predict_off.ipynb) trains and saves a CamemBERT for the Off-the-record task.




Once the model is trained and saved, you can use it to make predictions on the entire corpus by running:

```bash
python AugmentedSocialScientist/predict/predict_logged_models.py
```
**:exclamation: This script requires the contents of newspaper articles, which have been removed from the public repo due to copyright.**

For more details, see [here](./AugmentedSocialScientist/docs/pages/train_predict.md).

# 4. Results

The codes to produce the tables and figures in the paper are contained in the folder `AugmentedSocialScientist/analysis/`.

1. For the Table 2 and the Tables B1, B2, B2 in Appendix:
- Human performance scores (Research Assistants, Microworkers) are produced by the notebook [AugmentedSocialScientist/analysis/human_annotators_performance.ipynb](AugmentedSocialScientist/analysis/human_annotators_performance.ipynb);
- Model performance scores are produced by the notebook [AugmentedSocialScientist/analysis/saturation.ipynb](AugmentedSocialScientist/analysis/saturation.ipynb).

2. The Figure 3 is produced by the notebook [AugmentedSocialScientist/analysis/saturation.ipynb](AugmentedSocialScientist/analysis/saturation.ipynb).
3. The Figure 4 is produced by the notebook [AugmentedSocialScientist/analysis/prediction_analysis.ipynb](AugmentedSocialScientist/analysis/prediction_analysis.ipynb).
4. The Table A1 in Appendix is produced by the notebook [AugmentedSocialScientist/analysis/stats_datasets.ipynb](AugmentedSocialScientist/analysis/stats_datasets.ipynb).

