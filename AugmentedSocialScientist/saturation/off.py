import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from TransferSociologist.data import Dataset
from TransferSociologist.models import BertSequence
from TransferSociologist.utils import regularize_seqlab
from operator import add
from functools import reduce
from copy import deepcopy


def try_eval(x):
    try:
        return eval(x)
    except:
        return x


def fill_zeros(labels, zeros, conv_dict):
    try:
        labels = eval(labels)
    except:
        pass
    for l in labels:
        start_span, stop_span, lab = l
        size = len(zeros[start_span:stop_span])
        number = conv_dict[lab]
        zeros[start_span:stop_span] = [number] * size
    return zeros


def compute_TP(x, thres=0):
    nb_TP = 0
    labels = x.labels
    pred_labels = list(filter(lambda y: y[1] - y[0] > thres, x.pred_labels))

    for l_pred in pred_labels:
        for l in labels:
            is_overlap_1 = l[0] in range(l_pred[0], l_pred[1] + 1)
            is_overlap_2 = l[1] in range(l_pred[0], l_pred[1] + 1)
            is_overlap = is_overlap_1 or is_overlap_2
            if (
                is_overlap == True
            ):  # There is overlap, compute intersection etc size
                left_border = min(l_pred[0], l[0])
                right_border = max(l_pred[1], l[1])
                one_hot_pred = x.labels_pred_str[left_border:right_border]
                one_hot = x.labels_str[left_border:right_border]
                mul = np.dot(one_hot_pred, one_hot)
                intersection_length = mul.sum()
                if intersection_length > 0.25 * (l[1] - l[0]):
                    nb_TP = nb_TP + 1
                    break
    return nb_TP


def prepare_experiment(
    train_path, gs_path, drop_duplicates=False, percent_of_data=1
):
    dataset = Dataset()
    dataset.read(
        data_path=train_path, gold_standard_path=gs_path, data_type="csv"
    )
    dataset.df = dataset.df.rename({'is_control_1': 'is_control'}, axis=1)
    if drop_duplicates == True:
        if "is_control" in dataset.df.columns:
            gs = dataset.df[dataset.df.is_gold_standard == True]
            no_gs = dataset.df[dataset.df.is_gold_standard == False]
            no_gs = pd.concat(
                [
                    no_gs[no_gs.is_control == True]
                        .groupby(["text"])
                        .apply(lambda x: x.sample(1))
                        .reset_index(drop=True),
                    no_gs[no_gs.is_control != True]
                ]
            )
            dataset.df = pd.concat([no_gs, gs])
    # Now sample subset of data
    gs = dataset.df[dataset.df.is_gold_standard == True]
    no_gs = dataset.df[dataset.df.is_gold_standard == False]
    no_gs = no_gs.sample(frac=percent_of_data)
    dataset.df = pd.concat([no_gs, gs])

    dataset.task_encode(task_type="sequence_labelling", bert_model="CamemBert")
    # natural_samples = dataset.df

    dataset.encode_torch(
        task_type="sequence_labelling",
        bert_model="CamemBert",
        # test_size=0.3,
        random_seed=2018,
    )

    dataset_pred = Dataset()
    dataset_pred.read(data_path=gs_path, data_type="csv")
    dataset_pred.task_encode(
        task_type="sequence_labelling",
        bert_model="CamemBert",
        # pred_gs=True,
        pred_mode=True,
    )
    dataset_pred.encode_torch(
        task_type="sequence_labelling", bert_model="CamemBert", pred_mode=True
    )
    return dataset, dataset_pred


def run_experiment(dataset, dataset_pred, batch_size, lr, sampler, nepochs):
    clf = BertSequence()
    random_seed = np.random.randint(2021)

    perfs, best_perfs, epoch_best = clf.fit_evaluate(
        dataset.train,
        dataset.test,
        batch_size=batch_size,
        sampler=sampler,
        nepochs=nepochs,
        random_seed=random_seed,
        learning_rate=lr,
    )
    perf_dic = {
        "batch_size": batch_size,
        "lr": lr,
        "sampler": sampler,
        "nepochs": nepochs,
        "best epoch": int(epoch_best),
        "random_seed": int(random_seed),
        "train_size": len(dataset.train[0])

    }
    inv_conv_dict = {
        item: key
        for i, (key, item) in enumerate(dataset.conversion_dict.items())
    }
    for i in range(len(perfs[0])):
        j = inv_conv_dict[i]
        perf_dic[f"prec_{j}"] = float(perfs[0][i])
        perf_dic[f"rec_{j}"] = float(perfs[1][i])
        perf_dic[f"F1_{j}"] = float(perfs[2][i])
        perf_dic[f"supp_{j}"] = float(perfs[3][i])
        # perf_dic[f'prec_{j}_best_run'] = float(best_perfs[0][i])
        # perf_dic[f'rec_{j}_best_run'] = float(best_perfs[1][i])
        # perf_dic[f'F1_{j}_best_run'] = float(best_perfs[2][i])

    truncated_labels, truncated_logits = clf.predict(dataset_pred.pred)
    dataset_pred.df["truncated_labels"] = truncated_labels
    dataset_pred.df["truncated_logits"] = truncated_logits
    dataset_pred = regularize_seqlab(dataset_pred, dataset.tokenizer)

    preds = dataset_pred.df
    preds["labels_str"] = preds.sents.apply(lambda x: [0] * len(x))
    preds["labels_pred_str"] = preds.sents.apply(lambda x: [0] * len(x))
    preds["labels_str_len"] = preds["labels_str"].apply(len)
    # preds["labels_pred_str_len"] = preds["labels_pred_str"].apply(len)
    preds["labels_str"] = preds.apply(
        lambda x: fill_zeros(x.labels, x.labels_str, dataset.conversion_dict),
        axis=1,
    )
    preds["labels_pred_str"] = preds.apply(
        lambda x: fill_zeros(
            x.pred_labels, x.labels_pred_str, dataset.conversion_dict
        ),
        axis=1,
    )
    preds["labels_str_len2"] = preds["labels_str"].apply(len)
    preds["labels_pred_str_len2"] = preds["labels_pred_str"].apply(len)
    assert (preds["labels_str_len2"]==preds["labels_str_len"]).mean()==1, f'problem in true labels fill {(preds["labels_str_len2"]==preds["labels_str_len"]).mean()}'
    assert (preds["labels_pred_str_len2"]==preds["labels_str_len"]).mean()==1, f'problem in true labels fill {(preds["labels_pred_str_len2"]==preds["labels_str_len"]).mean()}'

    true = reduce(add, preds["labels_str"].values)
    pred = reduce(add, preds["labels_pred_str"].values)

    perfs_char = precision_recall_fscore_support(true, pred)
    for i in range(len(perfs_char[0])):
        j = inv_conv_dict[i]
        perf_dic[f"prec_char_{j}"] = float(perfs_char[0][i])
        perf_dic[f"rec_char_{j}"] = float(perfs_char[1][i])
        perf_dic[f"F1_char_{j}"] = float(perfs_char[2][i])
        perf_dic[f"supp_char_{j}"] = float(perfs_char[3][i])

    dataset_pred.df.pred_labels = dataset_pred.df.pred_labels.apply(try_eval)
    dataset_pred.df.labels = dataset_pred.df.labels.apply(try_eval)

    dataset_pred.df["TP"] = dataset_pred.df.apply(
        lambda x: compute_TP(x), axis=1
    )
    dataset_pred.df["TPFP"] = dataset_pred.df.pred_labels.apply(len)
    dataset_pred.df["TPFN"] = dataset_pred.df.labels.apply(len)

    dataset_pred.df["TP_thres4"] = dataset_pred.df.apply(
        lambda x: compute_TP(x, thres=4), axis=1
    )
    dataset_pred.df["TPFP_thres4"] = dataset_pred.df.pred_labels.apply(
        lambda x: len(list(filter(lambda y: y[1] - y[0] > 4, x)))
    )
    dataset_pred.df["TPFN_thres4"] = dataset_pred.df.labels.apply(len)

    perf_dic["prec_span"] = (
        dataset_pred.df["TP"].sum() / dataset_pred.df["TPFP"].sum()
    )
    perf_dic["rec_span"] = (
        dataset_pred.df["TP"].sum() / dataset_pred.df["TPFN"].sum()
    )
    perf_dic["F1_span"] = (
        2
        * perf_dic["prec_span"]
        * perf_dic["rec_span"]
        / (perf_dic["prec_span"] + perf_dic["rec_span"])
    )
    perf_dic["prec_span_T4"] = (
        dataset_pred.df["TP_thres4"].sum()
        / dataset_pred.df["TPFP_thres4"].sum()
    )
    perf_dic["rec_span_T4"] = (
        dataset_pred.df["TP_thres4"].sum()
        / dataset_pred.df["TPFN_thres4"].sum()
    )
    perf_dic["F1_span_T4"] = (
        2
        * perf_dic["prec_span"]
        * perf_dic["rec_span"]
        / (perf_dic["prec_span"] + perf_dic["rec_span"])
    )

    return perf_dic
