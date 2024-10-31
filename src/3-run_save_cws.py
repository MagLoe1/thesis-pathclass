import os
import pandas as pd

# pytorch
import torch

# other

from functions_small_helper import create_folder_if_not_exists
from functions_helper import ConfigReader, KFoldReader, LabelMapper

from functions_training import SingleTaskManager 



text_col = "tokenized_text"

kfold_label_col = "morphology"  # kfold stratification is based on morphology
for label_col in ["morphology", "site", "histology", "behavior"]:
    class_weights_per_task = list()
    for class_weight_type in ["bl015", "bl1", "skl"]:
        # set up everything
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # input data
        data_folder = os.path.join("..", "data", "data_clean", "datasplits", "train120_test20_date2023-03-15_data")
        kfolds_dir = os.path.join(data_folder, f"kfolds_{kfold_label_col}")

        train_valid_data_filename = "train_valid.pkl"
        data_file = os.path.join(data_folder, train_valid_data_filename)
        # test_filename = "test.pkl"
        # test_file = os.path.join(data_folder, test_filename)

        kfolds = KFoldReader(kfolds_dir, f"kfolds_{train_valid_data_filename[:-4]}_{kfold_label_col}.pkl")

        train_valid_data = pd.read_pickle(data_file).reset_index(drop=True)
        # test_data = pd.read_pickle(test_file).reset_index(drop=True)

        train_valid_data[label_col] = train_valid_data[label_col].astype("str")
        # test_data[label_col] = test_data[label_col].astype("str")
        assert type(train_valid_data[text_col][0] == list)
        label_mapper = LabelMapper(train_valid_data[label_col])
        n_classes = label_mapper.n_unique_labels

        cw_output_dir = os.path.join(data_folder, "class_weights_overview")
        create_folder_if_not_exists(cw_output_dir)


        for i, (train_ind, valid_ind) in enumerate(kfolds.folds, start=0):

            # print("Current experiment output folder", one_experiment_output_folder)
            print(f"----- FOLD {i} -----")
            row = {f"cw_type+fold" : f"{class_weight_type}_fold_{i}"}
            tm = SingleTaskManager(df_train=train_valid_data.iloc[train_ind], 
                                                task_list=[label_col],  # set to list here for single-task models
                                                cw_type=class_weight_type,
                                                as_tensor=False)
            columns_labelnames = tm.complete_task_dict[label_col]["sorted_labels_alph"]
            class_weights = tm.complete_task_dict[label_col]["class_weights"]
            for j, label in enumerate(columns_labelnames):
                row[label] = class_weights[j]
            class_weights_per_task.append(row)
    df_cws_per_task = pd.DataFrame(class_weights_per_task)
    df_cws_per_task.to_csv(os.path.join(cw_output_dir, f"class_weights_{label_col}.csv"), encoding="utf-8")
