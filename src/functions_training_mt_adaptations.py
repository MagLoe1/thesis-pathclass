import os
import torch
from tqdm import tqdm
from datetime import datetime
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score ,f1_score
from matplotlib import pyplot as plt
from collections import defaultdict
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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
    labels_predicted = defaultdict(list)
    labels_groundtruth = defaultdict(list)
    training_metrics = dict()
    labelwise_training_f1 = dict()
    with tqdm(train_dataloader, unit="batch") as batches:
        batches.set_description(f"Training")
        for batch in batches:
            # data, cuda
            batch_cuda = {key : value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()

            # forward pass
            if model.is_transformer:
                output_probs = model(batch_cuda["input_ids"], batch_cuda["attention_mask"])
            else:
                output_probs = model(batch_cuda["text"])
            true_labels = {key: batch_cuda[key] for key in model.task_dict.keys()}


            # backward pass
            loss = loss_function(output_probs, true_labels)
            loss.backward()
            optimizer.step()

            # save loss and predictions (on cpu)
            training_loss += loss.cpu().detach()

            for task in model.task_dict.keys():
                labels_predicted[task].extend(np.argmax(output_probs[task].cpu().detach().numpy(), axis=1))
                labels_groundtruth[task].extend(true_labels[task].cpu().numpy())
    for task in model.task_dict.keys():
        training_metrics[task] = get_metrics(labels_groundtruth[task], labels_predicted[task])
        labelwise_training_f1[task] = get_labelwise_f1(labels_groundtruth[task], labels_predicted[task])

    training_loss = training_loss
    return training_metrics, training_loss, labelwise_training_f1

def validate_one_epoch(model, valid_dataloader, loss_function, device):
    model.eval()

    validation_loss = 0
    labels_predicted = defaultdict(list)
    labels_groundtruth = defaultdict(list)
    validation_metrics = dict()
    labelwise_validation_f1 = dict()

    with tqdm(valid_dataloader, unit="batch") as batches:
        batches.set_description(f"Validation")
        with torch.no_grad():
            for batch in batches:
                # data, cuda
                batch_cuda = {key : value.to(device) for key, value in batch.items()}

                # forward pass
                if model.is_transformer:
                    output_probs = model(batch_cuda["input_ids"], batch_cuda["attention_mask"])
                else:
                    output_probs = model(batch_cuda["text"])
                true_labels = {key: batch_cuda[key] for key in model.task_dict.keys()}
                # backward pass
                loss = loss_function(output_probs, true_labels)


                # save loss and predictions (on cpu)
                validation_loss += loss.cpu().detach()
                for task in model.task_dict.keys():
                    labels_predicted[task].extend(np.argmax(output_probs[task].cpu().detach().numpy(), axis=1))
                    labels_groundtruth[task].extend(true_labels[task].cpu().numpy())
    for task in model.task_dict.keys():
        validation_metrics[task] = get_metrics(labels_groundtruth[task], labels_predicted[task])
        labelwise_validation_f1[task] = get_labelwise_f1(labels_groundtruth[task], labels_predicted[task])

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


def save_MT_plots_train_valid_over_epochs(training_stats, validation_stats, y_metric, stopping_epoch, output_dir, experiment_name):
    assert training_stats.keys() == validation_stats.keys()
    num_tasks = len(list(training_stats.keys()))
    fig, axs = plt.subplots(num_tasks, 1, figsize=(10, 5 * num_tasks))
 
    fig.suptitle(f"Training and Validation for Tasks: {list(training_stats.keys())}\n{experiment_name}")

    for i, task in enumerate(training_stats.keys()):

        # plot train and validation progress over epochs

        axs[i].plot(
            training_stats[task][y_metric], color="green", linestyle="-", 
            label=f"Training {y_metric}"
            )
        axs[i].plot(
            validation_stats[task][y_metric], color="red", linestyle="-", 
            label=f"Validation {y_metric}"
            )
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel(y_metric)

        axs[i].axvline(x=stopping_epoch, ymin=0.05, ymax=0.95, color='grey', linestyle=":", 
                    label=f"saved best model (epoch {stopping_epoch})")

        axs[i].set_title(task)
        axs[i].legend()
    plt.savefig(os.path.join(output_dir, f"{y_metric}_over_epochs.png"))
    plt.close()



class LossTracker:
    def __init__(self):
        self.best_valid_loss = float("inf")
        self.best_epoch = None

    def save_best_MTmodel(self, current_valid_loss, epoch, model, optimizer, loss_function, output_dir, experiment_name):
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
                    "n_classes": model.n_classes,
                    "tasks" : model.tasks
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
                to_return.append((f"\t{metric}: ", {task: metrics[task][metric] for task in metrics.keys()}))
        for metric, value in to_return:
            print(metric, value)

def training_loop_MT(train_dataloader, valid_dataloader, 
                  model, optimizer, loss_function, 
                  output_dir, experiment_name, device,
                  print_metric="loss",        # f1_macro, accuracy, loss 
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
    
    training_metrics_over_epochs = defaultdict(lambda: defaultdict(list))
    validation_metrics_over_epochs = defaultdict(lambda: defaultdict(list))
    training_labelwisef1_over_epochs = defaultdict(lambda: defaultdict(list))
    validation_labelwisef1_over_epochs = defaultdict(lambda: defaultdict(list))

    for epoch in range(max_epochs):
        if epoch % 50 == 0 and epoch > 1:
            print("cooling break")
            time.sleep(360)
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

        for task, metric_dict in training_metrics.items():
            for metric, value in metric_dict.items():
                assert value == training_metrics[task][metric]
                training_metrics_over_epochs[task][metric].append(training_metrics[task][metric])

            for idx, labelf1 in enumerate(labelwise_training_f1[task]):
                training_labelwisef1_over_epochs[task][idx].append(labelf1)

        # save valid loss + metrics
        validation_loss_over_epochs.append(validation_loss)
  

        for task, metric_dict in validation_metrics.items():
            for metric, value in metric_dict.items():
                assert value == validation_metrics[task][metric]
                validation_metrics_over_epochs[task][metric].append(validation_metrics[task][metric])

            for idx, labelf1 in enumerate(labelwise_validation_f1[task]):
                validation_labelwisef1_over_epochs[task][idx].append(labelf1)
        
        # update patience counter
        has_improved = loss_tracker.save_best_MTmodel(validation_loss, epoch, model, optimizer, loss_function, output_dir, experiment_name)
        if has_improved:
            count_no_improvement = 0
        else:

            count_no_improvement += 1
            print("PATIENCE COUNTER: ", count_no_improvement)

        # save plots
        save_plot_train_valid_over_epochs(training_loss_over_epochs, validation_loss_over_epochs, y_metric="loss", stopping_epoch=loss_tracker.best_epoch, output_dir=output_dir, experiment_name=experiment_name)
        
        for task, metric_dict in training_metrics.items():
            save_MT_plots_train_valid_over_epochs(training_metrics_over_epochs, validation_metrics_over_epochs, y_metric="accuracy", stopping_epoch=loss_tracker.best_epoch, output_dir=output_dir, experiment_name=experiment_name)
            save_MT_plots_train_valid_over_epochs(training_metrics_over_epochs, validation_metrics_over_epochs, y_metric="f1_macro", stopping_epoch=loss_tracker.best_epoch, output_dir=output_dir, experiment_name=experiment_name)
            save_MT_plots_train_valid_over_epochs(training_metrics_over_epochs, validation_metrics_over_epochs, y_metric="f1_weighted", stopping_epoch=loss_tracker.best_epoch, output_dir=output_dir, experiment_name=experiment_name)

            val_df = pd.DataFrame.from_dict(validation_metrics_over_epochs[task], orient="columns")
            val_df["loss"] = validation_loss_over_epochs
            val_df.to_csv(os.path.join(output_dir, f"validation_{task[:3]}_metrics.csv"))

        # Early stopping:
        if count_no_improvement >= patience and epoch > patience and epoch >= min_epochs:
            print(f"No validation loss improvement for {patience} consecutive epochs")
            print(f"Current epoch: {epoch}")
            print(f"Best model at Epoch {loss_tracker.best_epoch}")
            break
    return loss_tracker.best_epoch, epoch


def test(model, test_dataloader, device):
    model.eval()

    labels_predicted = defaultdict(list)
    labels_groundtruth = defaultdict(list)
    prediction_output = dict()

    test_metrics = dict()
    labelwise_test_f1 = dict()
    with tqdm(test_dataloader, unit="batch") as batches:
        batches.set_description(f"Test")
        with torch.no_grad():
            for batch in batches:
                # data, cuda
                batch_cuda = {key : value.to(device) for key, value in batch.items()}

                # forward pass
                if model.is_transformer:
                    output_probs = model(batch_cuda["input_ids"], batch_cuda["attention_mask"])
                else:
                    output_probs = model(batch_cuda["text"])

                true_labels = {key: batch_cuda[key] for key in model.task_dict.keys()}

                for task in model.task_dict.keys():
                    labels_predicted[task].extend(np.argmax(output_probs[task].cpu().detach().numpy(), axis=1))
                    labels_groundtruth[task].extend(true_labels[task].cpu().numpy())
    for task in model.task_dict.keys():
        test_metrics[task] = get_metrics(labels_groundtruth[task], labels_predicted[task])
        labelwise_test_f1[task] = get_labelwise_f1(labels_groundtruth[task], labels_predicted[task])
        prediction_output[task] = {"labels_true": labels_groundtruth[task], "labels_pred": labels_predicted[task]}
    return test_metrics, labelwise_test_f1, prediction_output


def plot_confusion_matrix(prediction_output, task, label_list, out_dir, experiment_name, normalize=None, name_suffix=""):
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
    plt.title(f"{experiment_name} \n(norm={normalize_suffix}) \nfor {task} ")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix{normalize_suffix}{name_suffix}.png"))
    plt.close()

    # save matrix in csv in case needed later
    df_conf_matrix = pd.DataFrame(conf_matrix, label_list, label_list)
    df_conf_matrix.to_csv(os.path.join(out_dir, f"confusion_matrix{normalize_suffix}{name_suffix}.csv"), sep=",")


def plot_labelwise_scores_1_epoch(labelwise_f1_scores, task, label_list, output_dir, experiment_name, name_suffix=""):

    # save scores in csv in case needed later on
    labelwise_f1_scores_df = pd.DataFrame({"labels": label_list, "labelwise_f1_scores": labelwise_f1_scores})
    labelwise_f1_scores_df.to_csv(os.path.join(output_dir, f"labelwise_f1_{name_suffix}.csv"), sep=",")

    # plot labelwise fscores
    fig, ax = plt.subplots()
    labelwise_f1_scores = [round(score, ndigits=3) for score in labelwise_f1_scores]
    ax.bar(label_list, labelwise_f1_scores)
    plt.bar_label(ax.containers[0], fontsize=6)
    plt.xticks(rotation=90, fontsize=8)
    ax.set_ylabel("F1-score", fontsize=8)
    ax.set_title(f"Labelwise F1-scores {task} ({experiment_name})", fontsize=8)
    
    plt.savefig(os.path.join(output_dir, f"labelwise_f1_{name_suffix}.png"))
    plt.close()



def test_saved_MTmodel(model, checkpoint_path, test_dataloader, output_dir, experiment_name, device, task_dict):
    print("Checkpoint path exists", os.path.exists(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])   # check if torch.load(path?)
    test_metrics, test_labelwisef1, prediction_output = test(model, test_dataloader, device)
    for task, value_dict in task_dict.items():
        plot_confusion_matrix(prediction_output[task], task, value_dict["sorted_labels_alph"], output_dir, experiment_name, name_suffix=f"_{task[:3]}")
        plot_confusion_matrix(prediction_output[task], task, value_dict["sorted_labels_alph"], output_dir, experiment_name, normalize="all", name_suffix=f"_{task[:3]}")
        plot_confusion_matrix(prediction_output[task], task, value_dict["sorted_labels_alph"], output_dir, experiment_name, normalize="true", name_suffix=f"_{task[:3]}")
        plot_labelwise_scores_1_epoch(test_labelwisef1[task], task, value_dict["sorted_labels_alph"], output_dir, experiment_name, name_suffix=f"_{task[:3]}")
        prediction_output_df = pd.DataFrame.from_dict(prediction_output[task], orient="columns")
        label_map = value_dict["label_map_num2alph"]
        prediction_output_df["labels_true_alph"] = prediction_output_df["labels_true"].map(label_map)
        prediction_output_df["labels_pred_alph"] = prediction_output_df["labels_pred"].map(label_map)
        prediction_output_df.to_csv(os.path.join(output_dir, f"pred_true_labels_{task[:3]}.csv"), sep=",", encoding="utf-8")
        class_report = get_sklearn_report(y_true=prediction_output_df["labels_true_alph"],
                                            y_predicted=prediction_output_df["labels_pred_alph"], 
                                            label_list=value_dict["sorted_labels_alph"], 
                                            output_dict=False)
        with open(os.path.join(output_dir, f"sklrn_report_{task}.txt"), "w") as out:
            out.writelines(class_report)
    return test_metrics, task_dict


