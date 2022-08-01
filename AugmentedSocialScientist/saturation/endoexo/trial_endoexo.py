import os
import json
import logging
import sys
from torch.cuda import empty_cache
sys.path.append(os.path.realpath('./AugmentedSocialScientist'))
# sys.path.append(os.path.realpath('../../'))

from saturation.endoexo import prepare_experiment, run_experiment
from PATHS import ENDOEXO_ASS, ENDOEXO_3, ENDOEXO_34, ENDOEXO_GS
from VARS import N_EPOCHS_ENDOEXO, SAMPLER_ENDOEXO, LR_ENDOEXO, BS_ENDOEXO
from VARS import DROP_DUPLICATES, PERCENT_OF_DATA


def process(params, paths, percent_of_data=1):
    train_path, gs_path = paths
    dataset, dataset_pred = prepare_experiment(
        train_path, gs_path, params["drop_duplicates"], percent_of_data
    )
    p = run_experiment(
        dataset,
        dataset_pred,
        params["batch_size"],
        params["lr"],
        params["sampler"],
        params["nepochs"],
    )
    return p


if __name__ == "__main__":
    train_paths = [ENDOEXO_ASS, ENDOEXO_3, ENDOEXO_34]
    gs_path = ENDOEXO_GS
    params = {
        "batch_size": BS_ENDOEXO,
        "nepochs": N_EPOCHS_ENDOEXO,
        "lr": LR_ENDOEXO,
        "sampler": SAMPLER_ENDOEXO,
        "drop_duplicates": DROP_DUPLICATES,
    }
    for tpath in train_paths:
        paths = tpath, gs_path
        log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs.jsonl"
        )
        for percent in PERCENT_OF_DATA:
            empty_cache()
            exp_name = os.path.basename(tpath).replace("_train", "").replace('.csv', '')
            logging.info(
                f"Launching experiment {exp_name}, with {percent*100}% of the dataset samples."
            )
            try:
                p = process(params, paths, percent)
                p["exp_name"] = exp_name
                p["percent_of_data"] = percent
                logging.info(
                    f"Success :  experiment {exp_name}, with {percent*100}% of the dataset samples."
                )
                for pk, ik in p.items():
                    logging.info(f"\t {pk}:{ik}")

                with open(log_path, "a", encoding="utf-8") as f:
                    json.dump(p, f)
                    f.write("\n")
                empty_cache()
            except:
                pass
