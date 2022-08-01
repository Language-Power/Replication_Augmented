[Home](../../../README.md) | [Data](../../../datasets/README.md) | [Package](../../../Core/README.md) | [Performance Scores](./saturation.md) | [Predictions](./train_predict.md) | [Results](./analysis.md)


# Predictions

## Training and Saving Models

Training the final models is done in jupyter notebooks, located in ```AugmentedSocialScientist/train```. 

There are two notebooks:
- [AugmentedSocialScientist/train/train_predict_endoexo.ipynb](../../train/train_predict_endoexo.ipynb) for the Policy/Politics task.
- [AugmentedSocialScientist/train/train_predict_off.ipynb](../../train/train_predict_off.ipynb) for the Off-the-record task.

If you are working on a remote server, open a jupyter-notebook server, type the bash command below:
```bash
jupyter-notebook --no-browser --port 8887 --ip 0.0.0.0
```
Open a web navigator and go to ```localhost:8887```

Saved models are saved in ```AugmentedSocialScientist/saved_models```

## Predicting

Prediction is done through a Python script (the script takes some time to run), ```./AugmentedSocialScientist/predict/predict_logged_models.py```. This script uses the models saved during training, so make sure you run the training notebooks beforehand. 

Use:
```bash
python AugmentedSocialScientist/predict/predict_logged_models.py
```
