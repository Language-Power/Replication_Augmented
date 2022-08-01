[Home](./README.md) | [Data](./datasets/README.md) | [Package](./Core/README.md) | [Performance Scores](./AugmentedSocialScientist/docs/pages/saturation.md) | [Predictions](./AugmentedSocialScientist/docs/pages/train_predict.md) | [Results](./AugmentedSocialScientist/docs/pages/analysis.md)

# The Augmented Social Scientist

This repository is a replication and data file for the paper "TThe Augmented Social Scientist. Using Sequential Transfer Learning to Annotate Texts Faster and More Accurately" by [Salomé Do](https://github.com/sally14), Étienne Ollion and [Rubing Shen](https://github.com/rubingshen).

**Resubmission update**: two models without pre-training are added in the folder `AugmentedSocialScientist/train/`  

- [AugmentedSocialScientist/train/train_predict_endoexo_SVM.ipynb](AugmentedSocialScientist/train/train_predict_endoexo_SVM.ipynb) trains a SVM model for Policy vs. Politics task.

- [AugmentedSocialScientist/train/train_predict_off_LSTM.ipynb](AugmentedSocialScientist/train/train_predict_off_LSTM.ipynb) trains a LSTM model for Off-the-Record task.

# 1. Data

The data we use for the paper is grouped in the folder `datasets`. It consists in newspaper articles from the French daily *Le Monde*, and of annotations for each of the two tasks, by three groups of annotators.

For more details, see [here](./datasets/README.md).

# 2. Package

The folder `Core` is a wrapper for the [HuggingFace Transformers](https://huggingface.co/transformers/index.html) library. This wrapper implements many higher-level functions : formatting data, training models, prediction... 

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

For Sample-Efficiency Curves (Figure 3, p. 22) and summary tables of scores (Table 2, p. 20; Tables 2, 3, 4 in the appendix), we need to run the training script several times for each training set size. 

To obtain score for just one attempt:

 ```bash
conda activate AugmentedSocialScientist
python AugmentedSocialScientist/saturation/off/trial_off.py
python AugmentedSocialScientist/saturation/endoexo/trial_endoexo.py
 ```
As the subsampling is random, we runned 5 to 20 experiments for each task, which means we runned 5 to 20 times the above command.

For more details, see [here](./AugmentedSocialScientist/docs/pages/saturation.md).



## 3.2 Predictions

For the evolution of both Politics and Policy indicators (Figure 4, p. 23 in the paper), we need to save a trained model and use it to produce predictions on each sentence in the entire corpus. 
  
- The notebook [AugmentedSocialScientist/train/train_predict_endoexo.ipynb](AugmentedSocialScientist/train/train_predict_endoexo.ipynb) trains and saves one model for the Policy/Politics task.
- The notebook [AugmentedSocialScientist/train/train_predict_off.ipynb](AugmentedSocialScientist/train/train_predict_off.ipynb) trains and saves one model for the Off-the-record task.

Once the model is trained and saved, you can use it to make predictions on the entire corpus by running:

```bash
python AugmentedSocialScientist/predict/predict_logged_models.py
```

For more details, see [here](./AugmentedSocialScientist/docs/pages/train_predict.md).

# 4. Results

The codes to produce the tables and figures in the paper are contained in the folder `AugmentedSocialScientist/analysis/`.

1. For the Table 2 (p. 20) and the Table 2, 3, 4 in Appendix (p. 36-37)
- Human performance scores (Research Assistants, Microworkers) are produced by the notebook [AugmentedSocialScientist/analysis/human_annotators_performance.ipynb](AugmentedSocialScientist/analysis/human_annotators_performance.ipynb)
- Model performance scores are produced by the notebook [AugmentedSocialScientist/analysis/saturation.ipynb](AugmentedSocialScientist/analysis/saturation.ipynb)

2. The Figure 3 (p. 22) is produced by the notebook [AugmentedSocialScientist/analysis/saturation.ipynb](AugmentedSocialScientist/analysis/saturation.ipynb)
3. The Figure 4 (p. 23) is produced by the notebook [AugmentedSocialScientist/analysis/prediction_analysis.ipynb](AugmentedSocialScientist/analysis/prediction_analysis.ipynb)
4. The Table 1 in Appendix (p. 36) is produced by the notebook [AugmentedSocialScientist/analysis/stats_datasets.ipynb](AugmentedSocialScientist/analysis/stats_datasets.ipynb)

