# imports

import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

class MTNLLLoss(nn.Module):
    def __init__(self, task_dict, device):
        super(MTNLLLoss, self).__init__()
        self.task_dict = task_dict

        self.loss_funs = nn.ModuleDict()

        for task, values in self.task_dict.items():
            if type(values["class_weights"]) == torch.Tensor:
                self.loss_funs[task] = nn.NLLLoss(weight=values["class_weights"].to(device)).to(device)
            else:
                self.loss_funs[task] = nn.NLLLoss().to(device)

    def forward(self, output_probs, true_labels):
        loss = 0
        for task in self.task_dict.keys():
            task_loss = self.loss_funs[task](output_probs[task], true_labels[task])
            loss += task_loss
        return loss


# for one fold
class TaskManager:
    def __init__(self, df_train, task_list, cw_types):
        # df_train contains all columns and original labels
        self.complete_task_dict = {}
        self.sorted_task_list = sorted(task_list)
        self.n_tasks = len(self.sorted_task_list)
        
        self.n_classes_of_tasks = list()
        

        for task_name in self.sorted_task_list:
            y_data_one_task_alph = df_train[task_name]
            self.complete_task_dict[task_name] = dict()
            
            sorted_class_labels_alph = sorted(list(y_data_one_task_alph.unique()))
            sorted_class_labels_num= [i for i in range(len(sorted_class_labels_alph))]

            label_map = {label: i for i, label in enumerate(sorted_class_labels_alph)}
            label_map_num2alph = {i: label for i, label in enumerate(sorted_class_labels_alph)}
            y_data_one_task_num = [label_map[label] for label in y_data_one_task_alph]

            self.complete_task_dict[task_name]["n_classes"] = len(sorted_class_labels_alph)
            self.n_classes_of_tasks.append(len(sorted_class_labels_alph))
            self.complete_task_dict[task_name]["sorted_labels_alph"] = np.array(sorted_class_labels_alph)
            self.complete_task_dict[task_name]["sorted_labels_num"] = np.array(sorted_class_labels_num)
            self.complete_task_dict[task_name]["y_data_alph"] = np.array(y_data_one_task_alph)
            self.complete_task_dict[task_name]["y_data_num"] = np.array(y_data_one_task_num)
            self.complete_task_dict[task_name]["label_map"] = label_map
            self.complete_task_dict[task_name]["label_map_num2alph"] = label_map_num2alph
            assert list(label_map.keys()) == sorted_class_labels_alph

            if cw_types:
                self.complete_task_dict[task_name][f"class_weights"] = self.compute_classweight_type(cw_type=cw_types[task_name],
                                                                                                 sorted_class_labels_num=sorted_class_labels_num, 
                                                                                                 y_data_one_task_num=y_data_one_task_num)
            else:
                self.complete_task_dict[task_name]["class_weights"] = None
        assert len(self.n_classes_of_tasks) == self.n_tasks


    def compute_classweight_type(self, cw_type, sorted_class_labels_num, y_data_one_task_num, as_tensor=True):

        if cw_type == "bl015":
            class_counts = np.bincount(y_data_one_task_num)
            class_weights = np.log(len(y_data_one_task_num) * 0.15 / class_counts)
            class_weights[class_weights < 1] = 1

        elif cw_type == "bl1":
            class_counts = np.bincount(y_data_one_task_num)
            class_weights = np.log(len(y_data_one_task_num) * 1 / class_counts)
            class_weights[class_weights < 1] = 1
        
        elif cw_type == "skl":
            class_weights= compute_class_weight(class_weight="balanced", 
                                                classes=np.array(sorted_class_labels_num), 
                                                y=y_data_one_task_num)
        else:
            print("Incorrect Classweight argument -> Default to None")
            class_weights = None
        
        if as_tensor:
            return torch.tensor(class_weights, dtype=torch.float32)
        return class_weights