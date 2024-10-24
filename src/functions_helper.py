import pandas as pd
import numpy as np
import torch
import json
import os
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


class ConfigReader:
    def __init__(self, path_to_configfile):
        self.config_dict = self.read_config(path_to_configfile)
        self.model_type = self.config_dict["model_type"]

        self.seed = self.config_dict["seed"]
        self.lr = self.config_dict["lr"]
        self.batch_size = self.config_dict["batch_size"]
        self.patience = self.config_dict["patience"]
        self.max_epochs = self.config_dict["max_epochs"]
        self.n_lr_iters = self.config_dict["n_lr_iters"]
        self.dropout = self.config_dict["dropout"]
        self.is_transformer = self.config_dict["is_transformer"]
        self.class_weights = self.config_dict["class_weights"]
        self.weight_decay = self.config_dict["weight_decay"]

        self.unk_threshold = self.config_dict["unk_threshold"]
        self.embedding_size = self.config_dict["embedding_size"]
        self.filter_sizes = self.config_dict["filter_sizes"]
        self.n_filters_per_size = self.config_dict["n_filters"]
        
        self.freeze_bert = self.config_dict["freeze_bert"]
        self.chunksize = self.config_dict["chunksize"]
        self.attention_heads = self.config_dict["attention_heads"]
        self.attention_dim = self.config_dict["attention_dim"]

        self.experiment_name = self.create_experiment_name()
        self.device = self.get_device()

    def read_config(self, path_to_configfile):
        with open(path_to_configfile) as f:
            config = json.load(f)
        return config
    
    def create_experiment_name(self):
        experiment_name = self.model_type
        exclude_from_experiment_name = ["is_transformer", "model_type", "filter_sizes", "chunksize", "embedding_size", "unk_threshold"]
        abbreviations = {"batch_size": "bs",
                         "lr": "lr",
                         "patience" : "p",
                         "max_epochs": "mx",
                         "n_lr_iters": "lrit",
                         "dropout" : "dr",
                         "unk_threshold": "unk",
                         "embedding_size" : "emb",
                         "class_weights" : "CW",
                         "weight_decay" : "wd",
                         "n_filters": "fs",
                         "freeze_bert" : "FB",
                         "chunksize" : "cs",
                         "attention_heads": "ah",
                         "attention_dim": "ad",
                         "seed": "s"}
        for key, value in self.config_dict.items():
            if value and key not in exclude_from_experiment_name:
                experiment_name += f"_{abbreviations[key]}{value}"
        return experiment_name
    
    def get_device(self, verbose=True):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        if verbose:
            print(f"Device: <{device}>")
        return device

class KFoldReader:
    def __init__(self, kfold_dir, kfold_filename):
        self.fold_df = pd.read_pickle(os.path.join(kfold_dir, kfold_filename))
        self.train = self.fold_df["train"]
        self.valid = self.fold_df["valid"]
        self.n_folds = len(self.fold_df)
        self.folds = [(self.train[i], self.valid[i]) for i in range(self.n_folds)]
        self.classweights_sklearn = self.fold_df["cw_sklearn"]
        self.classweights_blanchard_015 = self.fold_df["cw_blanchard_015"]
        self.classweights_blanchard_1 = self.fold_df["cw_blanchard_1"]





def compute_classweights_blanchard(classes, y_data, m, as_tensor=False):
    # map labels to idxs
    if type(classes[0]) == str:
        label_map = {label: i for i, label in enumerate(sorted(classes))}
        y_data = np.array([label_map[label] for label in y_data])
    n_samples = len(y_data)
    class_counts = np.bincount(y_data)
    class_weights = np.log(n_samples * m / class_counts)
    # classweights are restricted to be > 1 follwing the formula by Blanchard et al.
    class_weights[class_weights < 1] = 1
    print("weights check", class_weights, class_weights.shape)
    if as_tensor:
        return torch.tensor(class_weights, dtype=torch.float32)
    return class_weights


def store_strat_kfolds(x_data, y_data, unique_labels, out_dir, out_filename, random_state, n_splits, shuffle):
    stored_kfolds = defaultdict(tuple)
    strat_kfolds = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    for i, (train_idx, valid_idx) in enumerate(strat_kfolds.split(x_data, y_data)):
        # Labels will be SORTED to match label mapper! with sklearn metric
        classweights_sklearn = compute_class_weight(class_weight="balanced", classes=np.array(sorted(unique_labels)), y=y_data)
        # Labels will be SORTED to match label mapper!
        classweights_blanchard_015 = compute_classweights_blanchard(classes=np.array(sorted(unique_labels)), y_data=y_data, m=0.15)
        classweights_blanchard_1 = compute_classweights_blanchard(classes=np.array(sorted(unique_labels)), y_data=y_data, m=1)
        stored_kfolds[i] = (train_idx, valid_idx, classweights_sklearn, classweights_blanchard_015, classweights_blanchard_1)

        print("sklearn", classweights_sklearn)
        print("blanchard 0.15", classweights_blanchard_015)
        print("blanchard 1", classweights_blanchard_1)
    if not os.path.exists(out_dir):
        print(f"<{out_dir}> does not exist. New directory is created...")
        os.mkdir(out_dir)
    print(f"Folds stored in <{out_dir}>")
    kfold_df = pd.DataFrame.from_dict(stored_kfolds, orient="index", columns=["train", "valid", "cw_sklearn", "cw_blanchard_015", "cw_blanchard_1"])
    kfold_df.to_pickle(os.path.join(out_dir, out_filename))
    print(f"Kfolds stored in file {os.path.join(out_dir, out_filename)}")
    kfold_df.to_csv(os.path.join(out_dir, f"{out_filename[:-4]}.csv"))


class LabelMapper:
    def __init__(self, df_column):
        self.label_map = self.create_label_map(df_column=df_column)
        self.labels = np.array(sorted(list(self.label_map.keys())))
        self.n_unique_labels = len(self.labels)

    
    def create_label_map(self, df_column):
        label_map = sorted(list(df_column.unique()))
        return {label: i for i, label in enumerate(label_map)}


