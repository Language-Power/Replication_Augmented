import pandas as pd
import numpy as np
from functools import reduce
from operator import add
from TransferSociologist.data import Dataset
from TransferSociologist.models import BertClassifier
from sklearn.metrics import precision_recall_fscore_support


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
            dataset.df = pd.concat([gs, no_gs])

    dataset.task_encode(
        task_type="sentence_classification", bert_model="CamemBert"
    )
    # natural_samples = dataset.df
    # Now sample subset of data
    gs = dataset.df[dataset.df.is_gold_standard == True]
    no_gs = dataset.df[dataset.df.is_gold_standard == False]
    no_gs = no_gs.sample(frac=percent_of_data)
    dataset.df = pd.concat([no_gs, gs])
    dataset.encode_torch(
        task_type="sentence_classification",
        bert_model="CamemBert",
        # test_size=0.3,
        random_seed=2018,
    )

    dataset_pred = Dataset()
    dataset_pred.read(data_path=gs_path, data_type="csv")
    dataset_pred.task_encode(
        task_type="sentence_classification",
        bert_model="CamemBert",
        # pred_gs=True,
        pred_mode=True,
    )
    dataset_pred.encode_torch(
        task_type="sentence_classification",
        bert_model="CamemBert",
        pred_mode=True,
    )
    dataset_pred.df.head()
    return dataset, dataset_pred


def run_experiment(dataset, dataset_pred, batch_size, lr, sampler, nepochs):
    clf = BertClassifier()
    random_seed = np.random.randint(2021)

    perfs, _, epoch_best = clf.fit_evaluate(
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
        perf_dic[f"prec_tok_{j}"] = float(perfs[0][i])
        perf_dic[f"rec_tok_{j}"] = float(perfs[1][i])
        perf_dic[f"F1_tok_{j}"] = float(perfs[2][i])
        perf_dic[f"supp_tok_{j}"] = float(perfs[3][i])

    # Now, predict
    labels, logits = clf.predict(dataset_pred.pred)

    dataset_pred.df["labels_pred"] = labels
    dataset_pred.df["labels_pred"] = dataset_pred.df["labels_pred"].apply(
        lambda x: inv_conv_dict[x]
    )
    dataset_pred.df["logits"] = logits
    dataset_pred.df["pred_labels"] = dataset_pred.df.apply(
        lambda x: [[x.spans[0], x.spans[1], x.labels_pred]], axis=1
    )
    cleaned_preds = (
        dataset_pred.df.groupby(dataset_pred.df.text)
        .agg({"pred_labels": "sum"})
        .reset_index()
    )
    preds = pd.merge(
        cleaned_preds,
        dataset_pred.df[["text", "labels"]].drop_duplicates(),
        left_on="text",
        right_on="text",
    )
    preds["labels_str"] = preds.text.apply(lambda x: [0] * len(x))
    preds["labels_pred_str"] = preds.text.apply(lambda x: [0] * len(x))
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
    true = reduce(add, preds["labels_str"].values)
    pred = reduce(add, preds["labels_pred_str"].values)
    perfs_char = precision_recall_fscore_support(true, pred)
    for i in range(len(perfs_char[0])):
        j = inv_conv_dict[i]
        perf_dic[f"prec_char_{j}"] = float(perfs_char[0][i])
        perf_dic[f"rec_char_{j}"] = float(perfs_char[1][i])
        perf_dic[f"F1_char_{j}"] = float(perfs_char[2][i])
        perf_dic[f"supp_char_{j}"] = float(perfs_char[3][i])
    return perf_dic


def fill_zeros(labels, zeros, conv_dict):
    try:
        labels = eval(labels)
    except:
        pass
    for l in labels:
        start_span, stop_span, lab = l
        size = stop_span - start_span
        number = conv_dict[lab]
        zeros[start_span:stop_span] = [number] * size
    return zeros
