import os
import torch
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score ,f1_score
from matplotlib import pyplot as plt
from collections import defaultdict
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from torch.optim.lr_scheduler import LinearLR



def get_accuracy(y_true, y_predicted):
    assert len(y_predicted) == len(y_true)
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    return accuracy_score(y_true, y_predicted)

def get_avg_f1_macro(y_true, y_predicted):
    assert len(y_predicted) == len(y_true)
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    return f1_score(y_true, y_predicted, average="macro")

def get_avg_f1_weighted(y_true, y_predicted):
    assert len(y_predicted) == len(y_true)
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    return f1_score(y_true, y_predicted, average="weighted")

def get_metrics(y_true, y_predicted):
    accuracy = get_accuracy(y_true, y_predicted)
    f1_macro = get_avg_f1_macro(y_true, y_predicted)
    f1_weighted = get_avg_f1_weighted(y_true, y_predicted)
    return {"accuracy" : accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted}

def get_labelwise_f1(y_true, y_predicted):
    assert len(y_predicted) == len(y_true)
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    return f1_score(y_true, y_predicted, average=None)

def get_sklearn_report(y_true, y_predicted, label_list, output_dict=True):
    assert len(y_predicted) == len(y_true)
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    return classification_report(y_true=y_true, y_pred=y_predicted, 
                                 target_names=label_list, output_dict=output_dict,
                                 zero_division=0.0)



def train_one_epoch(model, train_dataloader, optimizer, loss_function, device):
    model.train()
    training_loss = 0
    labels_predicted = []
    labels_groundtruth = []
    with tqdm(train_dataloader, unit="batch") as batches:
        batches.set_description(f"Training")
        for batch in batches:
            # data, cuda
            if model.is_transformer:
                input_texts, true_labels, attention_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
                input_texts = input_texts.to(device)
                true_labels = true_labels.to(device)
                attention_mask = attention_mask.to(device)
            else:
                input_texts, true_labels = batch
                input_texts = input_texts.to(device)
                true_labels = true_labels.to(device)

            optimizer.zero_grad()

            # forward pass
            if model.is_transformer:
                output_probs = model(input_texts, attention_mask)
            else:
                output_probs = model(input_texts)

            # backward pass
            loss = loss_function(output_probs, true_labels)
            loss.backward()
            optimizer.step()

            # save loss and predictions (on cpu)
            training_loss += loss.cpu().detach()
            labels_predicted.extend(np.argmax(output_probs.cpu().detach().numpy(), axis=1))
            labels_groundtruth.extend(true_labels.cpu().numpy())
    training_metrics = get_metrics(labels_groundtruth, labels_predicted)
    labelwise_training_f1 = get_labelwise_f1(labels_groundtruth, labels_predicted)

    training_loss = training_loss
    return training_metrics, training_loss, labelwise_training_f1

def validate_one_epoch(model, valid_dataloader, loss_function, device):
    model.eval()

    validation_loss = 0
    labels_predicted = []
    labels_groundtruth = []
    with tqdm(valid_dataloader, unit="batch") as batches:
        batches.set_description(f"Validation")
        with torch.no_grad():
            for batch in batches:
                # data, cuda
                if model.is_transformer:
                    input_texts, true_labels, attention_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
                    input_texts = input_texts.to(device)
                    true_labels = true_labels.to(device)
                    attention_mask = attention_mask.to(device)
                else:
                    input_texts, true_labels = batch
                    input_texts = input_texts.to(device)
                    true_labels = true_labels.to(device)

                # forward pass
                if model.is_transformer:
                    output_probs = model(input_texts, attention_mask)
                else:
                    output_probs = model(input_texts)

                # backward pass
                loss = loss_function(output_probs, true_labels)


                # save loss and predictions (on cpu)
                validation_loss += loss.cpu().detach()
                labels_predicted.extend(np.argmax(output_probs.cpu().detach().numpy(), axis=1))
                labels_groundtruth.extend(true_labels.cpu().numpy())
    validation_metrics = get_metrics(labels_groundtruth, labels_predicted)
    labelwise_validation_f1 = get_labelwise_f1(labels_groundtruth, labels_predicted)

    validation_loss = validation_loss
    return validation_metrics, validation_loss, labelwise_validation_f1


def save_plot_train_valid_over_epochs(training_stats, validation_stats, y_metric, stopping_epoch, output_dir, experiment_name):

    # plot train and validation progress over epochs
    plt.figure(figsize=(10, 7))
    plt.plot(
        training_stats, color="green", linestyle="-", 
        label=f"Training {y_metric}"
        )
    plt.plot(
        validation_stats, color="red", linestyle="-", 
        label=f"Validation {y_metric}"
        )
    plt.xlabel("Epoch")
    plt.ylabel(y_metric)

    plt.axvline(x=stopping_epoch, ymin=0.05, ymax=0.95, color='grey', linestyle=":", 
                label=f"saved best model (epoch {stopping_epoch})")

    plt.title(experiment_name)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{y_metric}_over_epochs.png"))
    plt.close()

class LossTracker:
    def __init__(self):
        self.best_valid_loss = float("inf")
        self.best_epoch = None

    def save_best_model(self, current_valid_loss, epoch, model, optimizer, loss_function, output_dir, experiment_name):
        has_improved = None
        if current_valid_loss + 0.001 < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_epoch = epoch
            # save
            print("Improvement: Save model in epoch", epoch)
            torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_function,
                    "n_classes": model.n_classes
                    }, 
                    os.path.join(output_dir, f"best_{experiment_name}.pt"))
            has_improved = True
        else:
            has_improved = False
        return has_improved

def print_epoch_metric(print_metric, metrics, loss):
    if print_metric:
        if type(print_metric) != list:
            print_metric = [print_metric]
            assert len(print_metric) == 1
        to_return = []
        for metric in print_metric:
            if metric == "loss":
                to_return.append(("\tLoss: ", loss.item()))
            else:
                to_return.append((f"\t{metric}: ", metrics[metric]))
        for metric, value in to_return:
            print(metric, value)

def training_loop(train_dataloader, valid_dataloader, 
                  model, optimizer, loss_function, 
                  output_dir, experiment_name, device,
                  print_metric="loss",        # can be f1_macro, accuracy, loss 
                  patience=5, max_epochs=1000,
                  min_epochs=10,
                  lr_schedule=None):

    
    # stop training if no improvement after set epochs, based on valid metric
    # or if max_epochs are reached
    model = model.to(device)
    loss_tracker = LossTracker()
    if lr_schedule:
        lr_scheduler = lr_schedule
        print("Initial learning rate", lr_scheduler.get_last_lr())
    count_no_improvement = 0
    # save losses
    training_loss_over_epochs = []
    validation_loss_over_epochs = []

    # save metrics
    training_metrics_over_epochs = defaultdict(list)
    validation_metrics_over_epochs = defaultdict(list)
    training_labelwisef1_over_epochs = defaultdict(list)
    validation_labelwisef1_over_epochs = defaultdict(list)

    for epoch in range(max_epochs):
        print(f"Epoch: {epoch}")
        training_metrics, training_loss, labelwise_training_f1 = train_one_epoch(model, train_dataloader, optimizer, loss_function, device)
        print_epoch_metric(print_metric, training_metrics, training_loss)
        validation_metrics, validation_loss, labelwise_validation_f1 = validate_one_epoch(model, valid_dataloader, loss_function, device)
        print_epoch_metric(print_metric, validation_metrics, validation_loss)
        if lr_schedule:
            lr_scheduler.step()
            print("Update learning rate:", lr_scheduler.get_last_lr())
        
        # save training loss + metrics
        training_loss_over_epochs.append(training_loss)
        for metric in training_metrics.keys():
            training_metrics_over_epochs[metric].append(training_metrics[metric])
        
        for idx, labelf1 in enumerate(labelwise_training_f1):
            training_labelwisef1_over_epochs[idx].append(labelf1)

        # save valid loss + metrics
        validation_loss_over_epochs.append(validation_loss)
        for metric in validation_metrics.keys():
            validation_metrics_over_epochs[metric].append(validation_metrics[metric])

        for idx, labelf1 in enumerate(labelwise_validation_f1):
            validation_labelwisef1_over_epochs[idx].append(labelf1)
        
        
        # update patience counter
        has_improved = loss_tracker.save_best_model(validation_loss, epoch, model, optimizer, loss_function, output_dir, experiment_name)
        if has_improved:
            count_no_improvement = 0
        else:

            count_no_improvement += 1
            print("PATIENCE COUNTER: ", count_no_improvement)

        # save plots
        save_plot_train_valid_over_epochs(training_loss_over_epochs, 
                                          validation_loss_over_epochs,
                                            y_metric="loss", 
                                            stopping_epoch=loss_tracker.best_epoch, 
                                            output_dir=output_dir, 
                                            experiment_name=experiment_name)
        for metric in training_metrics.keys():
            save_plot_train_valid_over_epochs(training_metrics_over_epochs[metric], 
                                              validation_metrics_over_epochs[metric], 
                                              y_metric=metric, 
                                              stopping_epoch=loss_tracker.best_epoch, 
                                              output_dir=output_dir, 
                                              experiment_name=experiment_name)
        val_df = pd.DataFrame.from_dict(validation_metrics_over_epochs, orient="columns")
        val_df["loss"] = validation_loss_over_epochs
        val_df.to_csv(os.path.join(output_dir, "validation_metrics.csv"))

        # Early stopping:
        if count_no_improvement >= patience and epoch > patience and epoch > min_epochs:
            print(f"No validation loss improvement for {patience} consecutive epochs")
            print(f"Current epoch: {epoch}")
            print(f"Best model at Epoch {loss_tracker.best_epoch}")
            break
    return loss_tracker.best_epoch, epoch


def test(model, test_dataloader, device):
    model.eval()

    labels_predicted = []
    labels_groundtruth = []
    with tqdm(test_dataloader, unit="batch") as batches:
        batches.set_description(f"Test")
        with torch.no_grad():
            for batch in batches:
                # data, cuda
                if model.is_transformer:
                    input_texts, true_labels, attention_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
                    input_texts = input_texts.to(device)
                    true_labels = true_labels.to(device)
                    attention_mask = attention_mask.to(device)
                else:
                    input_texts, true_labels = batch
                    input_texts = input_texts.to(device)
                    true_labels = true_labels.to(device)

                # forward pass
                if model.is_transformer:
                    output_probs = model(input_texts, attention_mask)
                else:
                    output_probs = model(input_texts)

                
                labels_predicted.extend(np.argmax(output_probs.cpu().detach().numpy(), axis=1))
                labels_groundtruth.extend(true_labels.cpu().numpy())
    test_metrics = get_metrics(labels_groundtruth, labels_predicted)
    test_labelwisef1 = get_labelwise_f1(labels_groundtruth, labels_predicted)
    return test_metrics, test_labelwisef1, {"labels_true": labels_groundtruth,
                                            "labels_pred": labels_predicted}


def plot_confusion_matrix(prediction_output, label_list, out_dir, experiment_name, normalize=None, name_suffix=""):
    fig, ax = plt.subplots(figsize=(10,10))
    if normalize:
        normalize_suffix = f"_norm_{normalize}"
    else:
        normalize_suffix = ""

    conf_matrix = confusion_matrix(y_true=prediction_output["labels_true"], 
                                   y_pred=prediction_output["labels_pred"],
                                   normalize=normalize)
    if normalize:
        conf_matrix = conf_matrix * 100
        conf_matrix = conf_matrix.round(3)

    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                     display_labels=label_list)
    
    display.plot(ax=ax, include_values=True, cmap="magma", xticks_rotation="vertical")
    plt.title(f"{experiment_name}{normalize_suffix}{name_suffix}")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix{normalize_suffix}{name_suffix}.png"))
    plt.close()

    # save matrix in csv in case needed later
    df_conf_matrix = pd.DataFrame(conf_matrix, label_list, label_list)
    df_conf_matrix.to_csv(os.path.join(out_dir, f"confusion_matrix{normalize_suffix}{name_suffix}.csv"), sep=",")


def plot_labelwise_scores_1_epoch(labelwise_f1_scores, label_list, output_dir, experiment_name):

    # save scores in csv in case needed later on
    labelwise_f1_scores_df = pd.DataFrame({"labels": label_list, "labelwise_f1_scores": labelwise_f1_scores})
    labelwise_f1_scores_df.to_csv(os.path.join(output_dir, f"labelwise_f1.csv"), sep=",")

    # plot labelwise fscores
    fig, ax = plt.subplots()
    labelwise_f1_scores = [round(score, ndigits=3) for score in labelwise_f1_scores]
    ax.bar(label_list, labelwise_f1_scores)
    plt.bar_label(ax.containers[0], fontsize=6)
    plt.xticks(rotation=90, fontsize=8)
    ax.set_ylabel("F1-score", fontsize=8)
    ax.set_title(f"Labelwise F1-scores ({experiment_name})", fontsize=8)
    
    plt.savefig(os.path.join(output_dir, f"labelwise_f1.png"))
    plt.close()

def test_saved_model(model, checkpoint_path, test_dataloader, output_dir, experiment_name, device, label_list):
    print("Checkpoint path exists", os.path.exists(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_labelwisef1, prediction_output = test(model, test_dataloader, device)
    print("F1 Macro", test_metrics["f1_macro"])
    plot_confusion_matrix(prediction_output, label_list, output_dir, experiment_name)
    plot_confusion_matrix(prediction_output, label_list, output_dir, experiment_name, normalize="all")
    plot_confusion_matrix(prediction_output, label_list, output_dir, experiment_name, normalize="true")
    plot_labelwise_scores_1_epoch(test_labelwisef1, label_list, output_dir, experiment_name)
    label_map = {i : label for i, label in enumerate(label_list)}
    prediction_output_df = pd.DataFrame.from_dict(prediction_output, orient="columns")
    prediction_output_df["labels_true_alph"] = prediction_output_df["labels_true"].map(label_map)
    prediction_output_df["labels_pred_alph"] = prediction_output_df["labels_pred"].map(label_map)
    prediction_output_df.to_csv(os.path.join(output_dir, "pred_true_labels.csv"), sep=",", encoding="utf-8")
    
    class_report = get_sklearn_report(y_true=prediction_output_df["labels_true_alph"],
                                                          y_predicted=prediction_output_df["labels_pred_alph"], 
                                                          label_list=label_list, 
                                                          output_dict=False)
    with open(os.path.join(output_dir, "sklrn_report.txt"), "w") as out:
        out.writelines(class_report)

    return test_metrics, label_list


class ReadConfig:
    def __init__(self, config_path):
        self.config = json.load(config_path)


# for one fold
class SingleTaskManager:
    def __init__(self, df_train, task_list, cw_type="bl015", as_tensor=True):
        # df_train contains all columns and original labels
        self.complete_task_dict = {}
        self.sorted_task_list = sorted(task_list)
        self.n_tasks = len(self.sorted_task_list)
        self.as_tensor = as_tensor
        
        self.n_classes_of_tasks = list()
        

        for task_name in self.sorted_task_list:
            y_data_one_task_alph = df_train[task_name]
            self.complete_task_dict[task_name] = dict()
            
            sorted_class_labels_alph = sorted(list(y_data_one_task_alph.unique()))
            sorted_class_labels_num= [i for i in range(len(sorted_class_labels_alph))]

            label_map = {label: i for i, label in enumerate(sorted_class_labels_alph)}
            y_data_one_task_num = [label_map[label] for label in y_data_one_task_alph]

            self.complete_task_dict[task_name]["n_classes"] = len(sorted_class_labels_alph)
            self.n_classes_of_tasks.append(len(sorted_class_labels_alph))
            self.complete_task_dict[task_name]["sorted_labels_alph"] = np.array(sorted_class_labels_alph)
            self.complete_task_dict[task_name]["sorted_labels_num"] = np.array(sorted_class_labels_num)
            self.complete_task_dict[task_name]["y_data_alph"] = np.array(y_data_one_task_alph)
            self.complete_task_dict[task_name]["y_data_num"] = np.array(y_data_one_task_num)
            self.complete_task_dict[task_name]["label_map"] = label_map
            assert list(label_map.keys()) == sorted_class_labels_alph

            if cw_type:
                self.complete_task_dict[task_name][f"class_weights"] = self.compute_classweight_type(cw_type,
                                                                                                 sorted_class_labels_num, 
                                                                                                 y_data_one_task_num)
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
        
        if self.as_tensor:
            return torch.tensor(class_weights, dtype=torch.float32)
        return class_weights


