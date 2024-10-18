# this script contains helper functions that are used in other evaluation scripts
import os
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict



def load_df(csv_path, dtype="object"):
    df = pd.read_csv(csv_path, header=0, index_col=0, dtype=dtype)
    return df



def get_task_path_dict(result_collection_path, model_type, tasks):
    model_cv_dir = os.path.join(result_collection_path, model_type, f"{model_type}_cv")
    task_dirs = os.listdir(model_cv_dir)
    task_path_dict = dict()
    for task in tasks:
        for task_dir in task_dirs:
            # endswith model_cv_task
            if task_dir.endswith(task):
                task_path_dict[task] = os.path.join(model_cv_dir, task_dir)
    return task_path_dict

def get_pred_label_files(task_path_dict, pred_true_labels_filename, cw_flag):
    # get a dict of absolute paths for one file for one model and 
    # one class weight setting for all 5 folds and all specified tasks
    # ({"1": "abs-path/1-task-model-expname/filename", "2" : ....})
    pred_true_labels_all_tasks = dict()
    for task, cv_task_path in task_path_dict.items():
        per_task_data = dict()
        task_files = sorted(os.listdir(cv_task_path))
        for task_file in task_files:
            abs_path = os.path.join(cv_task_path, task_file)

            # abs path of one fold dir; e.g. 1-mor-CNN-exp_name
            if os.path.isdir(abs_path):  
                pred_true_labels_file = os.path.join(abs_path, pred_true_labels_filename)
                # store fold number
                if cw_flag and "CW" in task_file:
                    per_task_data[f"{task_file[0]}"] = pred_true_labels_file
                elif cw_flag is False and "CW" not in task_file:
                    per_task_data[f"{task_file[0]}"] = pred_true_labels_file
                else:
                    continue
        assert list(per_task_data.keys()) == ["1", "2", "3", "4", "5"]
        pred_true_labels_all_tasks[task] = per_task_data
    
    return pred_true_labels_all_tasks

def get_pred_label_files_MT(task_path_dict, pred_true_labels_filename, cw_flag):
    if pred_true_labels_filename.endswith("csv"):
        pred_true_labels_filename = pred_true_labels_filename[:-4]
    pred_true_labels_all_tasks = dict()
    for mttask, cv_task_path in task_path_dict.items():
        per_task_data = defaultdict(dict)
        task_files = sorted(os.listdir(cv_task_path))
        for task_file in task_files:
            abs_path = os.path.join(cv_task_path, task_file)

            # path of one fold dir; e.g. 1-mor-CNN-exp_name
            if os.path.isdir(abs_path):
                if mttask == "morsit":
                
                    for st_task in ["mor", "sit"]:
                        pred_true_labels_file = os.path.join(abs_path, f"{pred_true_labels_filename}_{st_task}.csv")
                        # store fold number
                        if cw_flag and "CW" in task_file:
                            per_task_data[f"{st_task}"][f"{task_file[0]}"] = pred_true_labels_file
                        elif cw_flag is False and "CW" not in task_file:
                            per_task_data[f"{st_task}"][f"{task_file[0]}"] = pred_true_labels_file
                        else:
                            continue

                else:
                    for st_task in ["beh", "his", "sit"]:
                        pred_true_labels_file = os.path.join(abs_path, f"{pred_true_labels_filename}_{st_task}.csv")
                        # store fold number
                        if cw_flag and "CW" in task_file:
                            per_task_data[f"{st_task}"][f"{task_file[0]}"] = pred_true_labels_file
                        elif cw_flag is False and "CW" not in task_file:
                            per_task_data[f"{st_task}"][f"{task_file[0]}"] = pred_true_labels_file
                        else:
                            continue

        for key, valdict in per_task_data.items():
            assert list(valdict.keys()) == ["1", "2", "3", "4", "5"]
        if mttask == "morsit":
            pred_true_labels_all_tasks[f"sit2"] = per_task_data["sit"]
            pred_true_labels_all_tasks[f"mor"] = per_task_data["mor"]
        else:
            pred_true_labels_all_tasks[f"sit3"] = per_task_data["sit"]
            pred_true_labels_all_tasks[f"his"] = per_task_data["his"]
            pred_true_labels_all_tasks[f"beh"] = per_task_data["beh"]
    return pred_true_labels_all_tasks


def get_mor_his_beh_scores(combo_df, excl_other=False):
    if excl_other:
        labels_mor = [label for label in sorted(combo_df["true_mor"].unique()) if label != "99999"]
        labels_his = [label for label in sorted(combo_df["true_his"].unique()) if label != "9999"]
        labels_beh = [label for label in sorted(combo_df["true_beh"].unique()) if label !="9"]
    else:
        labels_mor = sorted(combo_df["true_mor"].unique())
        labels_his = sorted(combo_df["true_his"].unique())
        labels_beh = sorted(combo_df["true_beh"].unique())
    if excl_other:
        assert len(labels_mor) == 15
        assert len(labels_his) == 12
        assert len(labels_beh) == 2
    else:
        assert len(labels_mor) == 16
        assert len(labels_his) == 13
        assert len(labels_beh) == 3
    
        
    
    acc_mor = accuracy_score(y_true=combo_df["true_mor"],
                        y_pred=combo_df["pred_mor"])
    acc_hisbeh = accuracy_score(y_true=combo_df["true_mor"],
                         y_pred=combo_df["pred_hisbeh"])
    
    acc_his = accuracy_score(y_true=combo_df["true_his"],
                      y_pred=combo_df["pred_his"])
    acc_his_from_mor = accuracy_score(y_true=combo_df["true_his"],
                               y_pred=combo_df["pred_his_from_mor"])
    
    acc_beh = accuracy_score(y_true=combo_df["true_beh"],
                      y_pred=combo_df["pred_beh"])
    acc_beh_from_mor = accuracy_score(y_true=combo_df["true_beh"],
                               y_pred=combo_df["pred_beh_from_mor"])
    
    
    f1_mor = f1_score(y_true=combo_df["true_mor"],
                        y_pred=combo_df["pred_mor"],
                        labels=labels_mor,
                        average="macro")

    f1_hisbeh = f1_score(y_true=combo_df["true_mor"],
                         y_pred=combo_df["pred_hisbeh"],
                         labels=labels_mor,
                         average="macro")
    f1_his = f1_score(y_true=combo_df["true_his"],
                      y_pred=combo_df["pred_his"],
                      labels=labels_his,
                      average="macro")
    f1_his_from_mor = f1_score(y_true=combo_df["true_his"],
                               y_pred=combo_df["pred_his_from_mor"],
                               labels=labels_his,
                               average="macro")
    f1_beh = f1_score(y_true=combo_df["true_beh"],
                      y_pred=combo_df["pred_beh"],
                      labels=labels_beh,
                      average="macro")
    f1_beh_from_mor = f1_score(y_true=combo_df["true_beh"],
                               y_pred=combo_df["pred_beh_from_mor"],
                               labels=labels_beh,
                               average="macro")
    
    scores = {"acc_mor" : acc_mor,
              "acc_hisbeh": acc_hisbeh,
              "acc_his_from_mor": acc_his_from_mor,
              "acc_his": acc_his,
              "acc_beh_from_mor": acc_beh_from_mor,
              "acc_beh": acc_beh,
              

              "f1_mor" : f1_mor,
              "f1_hisbeh": f1_hisbeh,
              "f1_his_from_mor": f1_his_from_mor,
              "f1_his": f1_his,
              "f1_beh_from_mor": f1_beh_from_mor,
              "f1_beh": f1_beh
              
              }
    return scores