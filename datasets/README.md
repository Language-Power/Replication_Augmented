[Home](../README.md) | [Data](./README.md) | [Package](../Core/README.md) | [Performance Scores](../AugmentedSocialScientist/docs/pages/saturation.md) | [Predictions](../AugmentedSocialScientist/docs/pages/train_predict.md) | [Results](../AugmentedSocialScientist/docs/pages/analysis.md)

# Data

This repository contains datasets for training our models on each of the two tasks (Policy/Politics and Off-the-Record) and predicting. 

All datasets are stored in the folder AugmentedSocialScientist, which contains two folders:
- all_train_and_gs: training data and gold standard data for each task (policy/politics, off), annotated by each of 3 groups of annotators (Social Scientists, Reseach Assistants, Microworkers)
- prediction_files: the entire corpus of newspaper articles from *Le Monde*, used to produce the predicted curve (Figure 4, p. 23 of the paper)

# 1. all_train_and_gs
Training data and gold standard data for each task (policy/politics, off-the-record) annotated by each of 3 groups of annotators (Social Scientist, Reseach Assistants, Microworkers).    
  
Contains 2 folders: 
- endoexo (for task1: **policy/politics**) 
- off (for task2: **off the record**)
## 1.1 endoexo
Contains training data and gold standard data for the task 1: **policy vs. politics**
  
Contains 2 folders: train (for training data) and gs (for gold standard data)
  
Each folders contains csv files, exported from the annotation tool *doccano*. All csv files have the same structure. Two columns are used:
- `text`: The newspaper articles to be annotated.
- `labels`: The annotations. Each annotation is a list of labels. Each label is a list of 3 items: beginning of the sentence, end of the sentence and the label text. For the label text, "endogène" corresponds to politics, "exogène" corresponds to policy, "autre" corresponds to other.


### 1.1.1 train
Contains 3 csv files:
- `endoexo_train_ass.csv`: training data annotated by the Social Scientist for the task policy vs. politics.
- `endoexo_train_3students.csv`: training data annotated by 3 Research Assistants for the task policy vs. politics.
- `endoexo_train_X34students.csv`: training data annotated by 34 Microworkers for the task policy vs. politics.
### 1.1.2 gs
Contains 3 csv files:
- `endoexo_gs_ass.csv`: ground truth for the task policy vs. politics, defined by the authors.
- `endoexo_gs_3students.csv`: annotation of the data that will constitute the ground truth by 3 Research Assistants (for human performance evaluation).
- `endoexo_train_X34students.csv`: annotation of the data that will constitute the ground truth by 34 Microworkers (for human performance evaluation).

## 1.2 off
Contains training data and gold standard data for the task 2: **off the record**
  
Contains 2 folders: train (for training data) and gs (for gold standard data)
  
Each folders contains csv files, exported from the annotation tool *doccano*. All csv files have the same structure. Two columns are used:
- `text`: The newspaper articles to be annotated.
- `labels`: The annotations. Each annotation is a list of labels. Each label is a list of 3 items: beginning of the sequence, end of the sequence and the label text. There is only one label text: "off". Unannotated content is identified as other (non off the record).

### 1.2.1 train
Contains 3 csv files:
- `off_train_ass.csv`: training data annotated by the Social Scientist for the task off the record.
- `off_train_3students.csv`: training data annotated by 3 Research Assistants for the task off the record.
- `off_train_34students.csv`: training data annotated by 34 Microworkers for the task off the record.
### 1.2.2 gs
Contains 3 csv files:
- `off_gs_validated_v20210716.csv`: ground truth for the task off the record, defined by the authors.
- `off_gs_3students.csv`: annotation of the data that will constitute the ground truth by 3 Research Assistants (for human performance evaluation).
- `off_gs_34students.csv`: annotation of the data that will constitute the ground truth by 34 Microworkers (for human performance evaluation).

# 2. prediction_files

Contains the entire corpus of newspaper articles from *Le Monde*, used to produce the prediction curve (Figure 4, p. 23 of the paper). 
Each csv file contains all articles for a given decade (file were split on size grounds).
