import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from functions_helper import store_strat_kfolds, KFoldReader
from global_variables import RANDOM_SEED
from functions_small_helper import create_folder_if_not_exists, get_sorted_labels_of_column



# read one dataset -> create kfolds and save them
# choose column (label col) on which the folds will be based on (in this thesis, morphology)

text_col = "tokenized_text"
label_col = "morphology"   # will be used in filename

# input data
data_folder = os.path.join("..", "data", "data_clean", "datasplits", "train120_test20_date2023-03-15_data")
data_filename = "train_valid.pkl"
data_file = os.path.join(data_folder, data_filename)

# output data
kfold_dir = os.path.join(data_folder, f"kfolds_{label_col}")
create_folder_if_not_exists(kfold_dir)
kfold_filename = f"kfolds_{data_filename[:-4]}_{label_col}.pkl"


# read data + check for correct datatypes (kfold only takes string labels)
train_valid_data = pd.read_pickle(data_file).reset_index(drop=True)
train_valid_data[label_col] = train_valid_data[label_col].astype("str")
assert type(train_valid_data[text_col][0] == list)


# double check if random seed / shuffle is needed or not
store_strat_kfolds(x_data=train_valid_data[text_col], 
                   y_data=train_valid_data[label_col],
                   unique_labels=train_valid_data[label_col].unique(),  # will be sorted! -- here they are not yet
                   out_dir=kfold_dir, 
                   out_filename=kfold_filename,
                   n_splits=6,
                   shuffle=True,
                   random_state=RANDOM_SEED)



kfolds = KFoldReader(kfold_dir=kfold_dir, kfold_filename=kfold_filename)

# plot statistics of each fold
cols = ["site", "morphology", "histology", "behavior"]
for c, col in enumerate(cols):
    figs, axs = plt.subplots(ncols=2, nrows=kfolds.n_folds)
    figs.set_figwidth(15)
    figs.set_figheight(30)
    figs.tight_layout(h_pad=6, w_pad=2)
    for i, (train_idxs, valid_idxs) in enumerate(kfolds.folds):
        # one row
        for j, (splitname, splitidxs) in enumerate([("train", train_idxs), ("valid", valid_idxs)]):
            print(splitname, len(splitidxs))
            axs[i, j] = sns.countplot(train_valid_data.loc[splitidxs], x=col, ax=axs[i, j], order=sorted(train_valid_data[col].unique()))
            axs[i, j].tick_params(axis="x", rotation=90)
            axs[i, j].bar_label(axs[i, j].containers[0], fontsize=6, rotation=0, padding=2)
            axs[i, j].set_title(f"Fold {i} {splitname} data ({len(splitidxs)} reports)")
    figs.suptitle(f"Counts per {col} for all different folds", fontsize=16)       
    plt.subplots_adjust(top=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(kfold_dir, f"kfolds_{col}_dist.png"))
    plt.close()
