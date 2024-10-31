import os
import numpy as np
import random
import pandas as pd
from datetime import datetime
import time

# torch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR

from transformers import AutoTokenizer
from datasets import Dataset

# custom functions

from functions_small_helper import create_folder_if_not_exists
from functions_helper import ConfigReader, KFoldReader, LabelMapper
from functions_training import training_loop, test_saved_model, SingleTaskManager
from functions_transform_data import BERTDataset
from models import KB_LongBERT



text_col = "text"

kfold_label_col = "morphology"  # kfolds are based on morph

label_col = "morphology"      # other options: "histology", "behavior", "site"


for suffix in ["_False", ""]:
###########################
# read experiment conifg + global variables

    REPLACEMENT_TOKS = ["[padnr]", "[namn]", "[pnr]", "[datum]", "[decimal]", "[time]"]
    LANG = "swedish"

    config_folder = os.path.join("..", "config_files")
    config_file = os.path.join(config_folder, f"config_bert_{label_col[:3]}{suffix}.json")
    config = ConfigReader(config_file)
    if config.freeze_bert != "partial":
        config.freeze_bert = "partial"
    if config.max_epochs > 15:  # safety measure to not it train for too long 
        config.max_epochs = 8

    ## set up everything
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    else:
        exit()  # if no GPU available, stop to prevent overheating
    # set random seeds
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)




    # load data
    # input data
    data_folder = os.path.join("..", "data", "data_clean", "datasplits", "train120_test20_date2023-03-15_data")
    kfolds_dir = os.path.join(data_folder, f"kfolds_{kfold_label_col}")
    train_valid_data_filename = "train_valid.pkl"
    data_file = os.path.join(data_folder, train_valid_data_filename)
    kfolds = KFoldReader(kfolds_dir, f"kfolds_{train_valid_data_filename[:-4]}_{kfold_label_col}.pkl")

    test_filename = "test.pkl"
    test_file = os.path.join(data_folder, test_filename)


    # output data
    datestamp = str(datetime.now().strftime("%Y%m%d"))
    datetimestamp = str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    experiments_dir = os.path.join("..", f"out_{datestamp}")
    create_folder_if_not_exists(experiments_dir)

    bert_dir = os.path.join(experiments_dir, config.model_type)
    create_folder_if_not_exists(bert_dir)

    # add cross validation folder for results:
    cv_dir = os.path.join(bert_dir, f"{config.model_type}_cv_{label_col[:3]}")
    create_folder_if_not_exists(cv_dir)

    ###
    train_valid_data = pd.read_pickle(data_file).reset_index(drop=True)
    test_data = pd.read_pickle(test_file).reset_index(drop=True)

    train_valid_data[label_col] = train_valid_data[label_col].astype("str")
    test_data[label_col] = test_data[label_col].astype("str")


    label_mapper = LabelMapper(train_valid_data[label_col])
    numeric_labels = np.array(sorted(list(label_mapper.label_map.values())))

    train_valid_with_alph_labels = train_valid_data.copy(deep=True)

    train_valid_data[label_col] = train_valid_data[label_col].map(label_mapper.label_map) # for BERT, labels are added here already (remaining models have it in dataset functions)
    test_data[label_col] = test_data[label_col].map(label_mapper.label_map)


    n_classes = label_mapper.n_unique_labels


    # BERT specific: offline model name
    model_name = "KB/bert-base-swedish-cased"
    model_path_offline = os.path.join("..", "..", "models_offline", "KB-bert-base-swedish-cased-AutoModel")
    if not os.path.exists(model_path_offline):
        print("Path to downloaded pretrained models not found.")
        raise FileNotFoundError

    ## BERT specific: max length of texts
    model_max_length = 5120

    tokenizer_long = AutoTokenizer.from_pretrained(model_path_offline,  
                                                local_files_only=True, 
                                                model_max_length=model_max_length,
                                                padding_side="right",
                                                truncation_side="left")
    # add replacement tokens to tokenizer
    vocab_size_before = len(tokenizer_long)
    tokenizer_long.add_tokens(REPLACEMENT_TOKS)
    vocab_size_after = len(tokenizer_long)
    print(vocab_size_before, "-->", vocab_size_after)
    assert vocab_size_before < vocab_size_after


    train_valid_dataset = BERTDataset(df=train_valid_data,   # input_ids, labels, attention_mask
                                        textcol_name=text_col,
                                        labelcol_name=label_col,
                                        tokenizer=tokenizer_long,
                                        truncation=True,
                                        padding=False)

    test_dataset = BERTDataset(df=test_data,   # input_ids, labels, attention_mask
                                    textcol_name=text_col,
                                    labelcol_name=label_col,
                                    tokenizer=tokenizer_long,
                                    truncation=True,
                                    padding=False)

    # result dict and log files
    cv_results = dict()
    # base_experiment_name = config.experiment_name
    # print("base experiment name", base_experiment_name)

    log_file = os.path.join(cv_dir, f"logg_{label_col[:3]}{config.experiment_name}_{datetimestamp}.txt")


                
    for i, (train_ind, valid_ind) in enumerate(kfolds.folds[1:], start=1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        one_experiment_output_folder = os.path.join(cv_dir, f"{i}-{label_col[:3]}-{config.experiment_name}")
        create_folder_if_not_exists(one_experiment_output_folder)
        print(f"----- FOLD {i} -----")



        tm = SingleTaskManager(df_train=train_valid_with_alph_labels.iloc[train_ind], 
                                task_list=[label_col],  # set to list here for ST models
                                cw_type=config.class_weights)
        print("training loader")
        train_loader = DataLoader(dataset=train_valid_dataset.dataset,
                    batch_size=config.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(train_ind),
                    drop_last=False,
                    collate_fn=train_valid_dataset.collate_fn)
        print("valid loader")
        valid_loader = DataLoader(dataset=train_valid_dataset.dataset,
                                batch_size=config.batch_size,
                                sampler=torch.utils.data.SubsetRandomSampler(valid_ind),
                                drop_last=False,
                                collate_fn=train_valid_dataset.collate_fn)
        print("test loader")
        test_loader = DataLoader(dataset=test_dataset.dataset,
                                batch_size=config.batch_size,
                                drop_last=False, 
                                collate_fn=test_dataset.collate_fn)
        
        model = KB_LongBERT(bert_model_path=model_path_offline,
                n_classes=n_classes,
                tokenizer=tokenizer_long,
                device=config.device,
                freeze_bert=config.freeze_bert,
                dropout=config.dropout,
                chunksize=config.chunksize)
        
        # handle partial freezing
        layer_components_with_grad = [param for param in model.parameters() if param.requires_grad]
        print(len(layer_components_with_grad))
        assert 50 > len(layer_components_with_grad) >=2 # at least 2 if only classification unfrozen;

        optimizer = optim.Adam(layer_components_with_grad, lr=config.lr)

        if config.n_lr_iters:
            lr_scheduler = LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.5, total_iters=config.n_lr_iters)
        else:
            lr_scheduler = None

        if config.class_weights:
            loss_function = nn.NLLLoss(weight=tm.complete_task_dict[label_col]["class_weights"].to(config.device)).to(config.device)
        else:
            loss_function = nn.NLLLoss().to(config.device)
        

        best_epoch, epoch = training_loop(train_dataloader=train_loader, 
                            valid_dataloader=valid_loader,
                            model=model, 
                            optimizer=optimizer, 
                            loss_function=loss_function, 
                            output_dir=one_experiment_output_folder, 
                            experiment_name=config.experiment_name,
                            print_metric=["f1_macro", "accuracy", "loss"],
                            device=config.device, 
                            patience=config.patience,
                            min_epochs=1,
                            max_epochs=config.max_epochs,
                            n_lr_iters=config.n_lr_iters,
                            lr_schedule=lr_scheduler)

        # test loader
        test_metrics, test_labels = test_saved_model(model=model, 
                                                    checkpoint_path=os.path.join(one_experiment_output_folder, f"best_{config.experiment_name}.pt"),
                                                    test_dataloader=test_loader, 
                                                    output_dir=one_experiment_output_folder, 
                                                    experiment_name=config.experiment_name, 
                                                    device=config.device,
                                                    label_list=label_mapper.labels)
        with open(os.path.join(one_experiment_output_folder, f"{i}_{label_col[:3]}_result_best{best_epoch}.txt"), "w") as out:
            out.write(f"{config.experiment_name} (saved (=best) ep. {best_epoch}) \t\t {test_metrics} (shown until epoch {epoch})")
        test_metrics.update({"best_epoch": best_epoch})
        cv_results[f"{i}_{config.experiment_name}_best_ep{best_epoch}"] = test_metrics

        with open(log_file, "a") as log:
            log.write(f"{i}_{label_col[:3]}_{config.experiment_name}_best_ep{best_epoch} \t\t {test_metrics}\n")
        time.sleep(120) # let cool down

    results_df = pd.DataFrame.from_dict(cv_results, orient="index")
    add_stats = {"mean": results_df.mean(),
                     "min": results_df.min(),
                     "max" : results_df.max(),
                     "std" : results_df.std()
                     }
    for stat, values in add_stats.items():
        results_df.loc[stat] = values
    results_df.to_csv(os.path.join(cv_dir, f"summary_{label_col[:3]}_{config.experiment_name}_{datetimestamp}.csv"), encoding="utf-8")
    time.sleep(120) # let cool down
