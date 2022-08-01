import os
from glob import glob
import pandas as pd
import numpy as np
from functools import reduce
from operator import add
from TransferSociologist.data import Dataset
from TransferSociologist.models import BertClassifier
from TransferSociologist.models import BertSequence
from TransferSociologist.utils import regularize_seqlab
from tqdm.auto import tqdm
import sys
sys.path.append(os.path.realpath('./AugmentedSocialScientist'))

from sklearn.metrics import precision_recall_fscore_support

from PATHS import DATAPATH, PRED_PATH, SAVED_MODELS_PATH


PRED_DATA_PATH_LIST = glob(os.path.join(PRED_PATH,'LeMonde_*.csv'))


random_seed = 2021
BS_ENDOEXO = 64
BS_OFF = 32

clf_off = BertSequence()
clf_pp = BertClassifier()

clf_pp.load(os.path.join(SAVED_MODELS_PATH, 'policy_politics_ASS/'))
clf_off.load(os.path.join(SAVED_MODELS_PATH, 'off_ASS/'))

clf_off.modelargs = {'batch_size' : BS_OFF}
clf_pp.modelargs = {'batch_size' : BS_ENDOEXO}


for pred_path in tqdm(PRED_DATA_PATH_LIST):

    dataset_pred_pp = Dataset()
    dataset_pred_pp.read(data_path=pred_path, data_type="csv")
    dataset_pred_pp.df = dataset_pred_pp.df.rename({'texte': 'text'}, axis=1)

    dataset_pred_pp.task_encode(
        task_type="sentence_classification",
        bert_model="CamemBert",
        pred_mode=True,
    )
    dataset_pred_pp.encode_torch(
        task_type="sentence_classification", bert_model="CamemBert", pred_mode=True
    )
    labels_pp, logits_pp = clf_pp.predict(dataset_pred_pp.pred)

    # predictions_off.append(dataset_pred_off.df)

    dataset_pred_pp.df["labels_pred"] = labels_pp
    #dataset_pred_pp.df["labels_pred"] = dataset_pred_pp.df["labels_pred"].apply(
    #    lambda x: inv_conv_dict[x]
    #)
    dataset_pred_pp.df["logits"] = logits_pp
    dataset_pred_pp.df["pred_labels"] = dataset_pred_pp.df.apply(
        lambda x: [[x.spans[0], x.spans[1], x.labels_pred]], axis=1
    )
    cleaned_preds = (
        dataset_pred_pp.df.groupby(dataset_pred_pp.df.text)
        .agg({"pred_labels": "sum"})
        .reset_index()
    )
    dataset_pred_pp.df = pd.merge(
        cleaned_preds,
        dataset_pred_pp.df.drop(['pred_labels'], axis=1),
        left_on="text",
        right_on="text",
    )
    dataset_pred_pp.df.to_csv(pred_path.replace('prediction_files', 'predicted_files_pp'))


    
for pred_path in tqdm(PRED_DATA_PATH_LIST):

    dataset_pred_off = Dataset()
    dataset_pred_off.read(data_path=pred_path, data_type="csv")
    dataset_pred_off.df = dataset_pred_off.df.rename({'texte': 'text'}, axis=1)[:10]
    dataset_pred_off.task_encode(
        task_type="sequence_labelling",
        bert_model="CamemBert",
        pred_mode=True,
    )
    dataset_pred_off.encode_torch(
        task_type="sequence_labelling", bert_model="CamemBert", pred_mode=True
    )
    truncated_labels_off, truncated_logits_off = clf_off.predict(dataset_pred_off.pred)


    dataset_pred_off.df['truncated_labels'] = truncated_labels_off
    dataset_pred_off.df['truncated_logits'] = truncated_logits_off
    dataset_pred_off = regularize_seqlab(dataset_pred_off, dataset_pred_off.tokenizer)

    dataset_pred_off.df.to_csv(pred_path.replace('prediction_files', 'predicted_files_off'))
