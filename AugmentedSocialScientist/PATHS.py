import os

REPO_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH =  os.path.split(REPO_PATH)[0]

DATAPATH = os.path.join(REPO_PATH, 'datasets')
PRED_PATH = os.path.join(DATAPATH, 'AugmentedSocialScientist/prediction_files')
SAVED_MODELS_PATH = os.path.join(REPO_PATH, 'AugmentedSocialScientist/saved_models')

OFF_ASS = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/off/train/off_train_ass.csv",
)
OFF_3 = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/off/train/off_train_3students.csv",
)
OFF_34 = train_path = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/off/train/off_train_34students.csv",
)

OFF_GS = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/off/gs/off_gs_validated_v20210716.csv",
)

OFF_RA_GS = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/off/gs/off_gs_3students.csv",
)

OFF_MW_GS = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/off/gs/off_gs_34students.csv",
)


ENDOEXO_ASS = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/endoexo/train/endoexo_train_ass.csv",
)
ENDOEXO_3 = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/endoexo/train/endoexo_train_3students.csv",
)
ENDOEXO_34 = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/endoexo/train/endoexo_train_X34students.csv",
)
ENDOEXO_GS = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/endoexo/gs/endoexo_gs_ass.csv",
)
ENDOEXO_RA_GS = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/endoexo/gs/endoexo_gs_assistants.csv",
)
ENDOEXO_MW_GS = os.path.join(
    DATAPATH,
    "AugmentedSocialScientist/all_train_and_gs/endoexo/gs/endoexo_gs_34students.csv",
)
