[Home](../README.md) | [Data](../datasets/README.md) | [Package](./README.md) | [Performance Scores](../AugmentedSocialScientist/docs/pages/saturation.md) | [Predictions](../AugmentedSocialScientist/docs/pages/train_predict.md) | [Results](../AugmentedSocialScientist/docs/pages/analysis.md)

# Package

The folder Core is a wrapper for the HuggingFace Transformers library. This wrapper implements many higher-level functions : formatting data, training models, prediction...
This makes [HuggingFace Transformers](https://huggingface.co/transformers/index.html) easier to use for people lacking a computer science background when dealing with non-standard textual data.

**WARNING** : recent GPU required. Our configuration (and subsequent batch size) makes use of a NVidia V100 (can be used for free using Google Colab for instance)

## Installing locally the repository 

### Requirements

**Software**

- Unix-based OS (OSX/Linux)
- Recent version of pip
- [Optionnal] Conda (we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html)). Conda enables to install all needed packages in a news, clean environment separated from the user's base environment. 

**Hardware**
- GPU : we use one Nvidia V100, which has 32Go of RAM. Batch sizes of our models were computed so that they make full use of the graphic cards. 

### Installing the required packages

```bash
cd Core
conda create --name AugmentedSocialScientist python pip jupyter
conda activate AugmentedSocialScientist
pip install .
cd ..
```

### Uninstall:

```bash
pip uninstall AugmentedSocialScientist
```


### Troubleshooting 

- ```conda not found``` -> Make sure that conda is properly configured with your shell (add ```export PATH="/home/username/miniconda3/bin:$PATH" ``` in your ```.bash_profile``` or try ```conda init --help```)
- ```No matching distribution found for ...``` -> Update your version of pip, or use pip3
- In case this doesn't work, you can install by yourself the packages listed in ```Core/setup.py```



