# imports


import random
import os
from datetime import datetime
import time
import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModel, AutoTokenizer

from functions_small_helper import create_folder_if_not_exists
from functions_helper import ConfigReader, KFoldReader


from mt_functions_training import TaskManager, MTNLLLoss
from mt_functions_transform import MTBERTDataset
from mt_models import MTKB_LongBERT
from functions_training_mt_adaptations import training_loop_MT, test_saved_MTmodel



for label_cols in [sorted(["histology", "behavior", "site"]),  sorted(["morphology", "site"])]:
    for suffix in ["", "_False"]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        assert type(label_cols) == list

        ###########################
        # read experiment conifg + global variables

        REPLACEMENT_TOKS = ["[padnr]", "[namn]", "[pnr]", "[datum]", "[decimal]", "[time]"]
        LANG = "swedish"

        config_folder = os.path.join("..", "config_files")
        config_file = os.path.join(config_folder, f"config_mtbert_{len(label_cols)}tasks{suffix}.json")
        config = ConfigReader(config_file)

        if config.freeze_bert != "partial":
            config.freeze_bert = "partial"

        ## set up everything
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
        else:
            exit()  # only run if GPU available
        # set random seeds
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)


        kfold_label_col = "morphology"  # kfold stratification is based on morphology


        if config.class_weights:
            config.class_weights = {"morphology" : "bl015",
                            "site": "bl015",
                            "behavior" : "bl1",
                            "histology": "bl1"}

        text_col = "text"

        if "morphology" in label_cols:
            sort_by_col = "morphology"
        elif "histology" in label_cols:
            sort_by_col = "histology"
        else:
            "Warning: no core task included. Please include histology or morphology in list"
            exit()

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

        experiments_dir = os.path.join("..", f"out_MT_{datestamp}")
        create_folder_if_not_exists(experiments_dir)

        bert_dir = os.path.join(experiments_dir, config.model_type)
        create_folder_if_not_exists(bert_dir)

        tasks_abbr = ""
        for task in label_cols:
            tasks_abbr += task[:3]
        cv_dir = os.path.join(bert_dir, f"{config.model_type}_cv_{tasks_abbr}")
        create_folder_if_not_exists(cv_dir)

        cv_results = dict()


        log_file = os.path.join(cv_dir, f"logg_{tasks_abbr}_{config.experiment_name}_{datetimestamp}.txt")

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
                                                    truncation_side="left") # in case needed, keep end as diagnosis often is noted there
        # add replacement tokens to tokenizer
        vocab_size_before = len(tokenizer_long)
        tokenizer_long.add_tokens(REPLACEMENT_TOKS)
        vocab_size_after = len(tokenizer_long)
        print(vocab_size_before, "-->", vocab_size_after)
        assert vocab_size_before < vocab_size_after


        # load data + assure correct format
        train_valid_data = pd.read_pickle(data_file).reset_index(drop=True)
        test_data = pd.read_pickle(test_file).reset_index(drop=True)

        for label_col in label_cols:
            train_valid_data[label_col] = train_valid_data[label_col].astype("str")
            test_data[label_col] = test_data[label_col].astype("str")


        for i, (train_ind, valid_ind) in enumerate(kfolds.folds[1:], start=1):

            print(f"--- Experiment {config.experiment_name} FOLD {i} -----")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            one_experiment_output_folder = os.path.join(cv_dir, f"{i}-{tasks_abbr}-{config.experiment_name}")
            create_folder_if_not_exists(one_experiment_output_folder)

            tm = TaskManager(df_train=train_valid_data.iloc[train_ind], 
                            task_list=label_cols,
                            cw_types=config.class_weights)

            print(f"Number of classes for tasks {tm.sorted_task_list} in the training set: {tm.n_classes_of_tasks}")
            dataset_train = MTBERTDataset(train_valid_data.iloc[train_ind],
                                            text_col=text_col, 
                                            label_cols=label_cols,
                                            task_manager=tm.complete_task_dict,
                                            tokenizer=tokenizer_long,
                                            encode=True, truncation=True, padding=False)
            
            train_loader = DataLoader(dataset=dataset_train,
                                    batch_size=config.batch_size,
                                    collate_fn=dataset_train._collate_fn,
                                    drop_last=False)
            
            dataset_valid = MTBERTDataset(train_valid_data.iloc[valid_ind],
                                            text_col=text_col, 
                                            label_cols=label_cols,
                                            task_manager=tm.complete_task_dict,
                                            tokenizer=tokenizer_long,
                                            encode=True, truncation=True, padding=False)
            valid_loader = DataLoader(dataset=dataset_valid,
                                    batch_size=config.batch_size,
                                    collate_fn=dataset_valid._collate_fn,
                                    drop_last=False)
            
            dataset_test = MTBERTDataset(test_data,
                                            text_col=text_col, 
                                            label_cols=label_cols,
                                            task_manager=tm.complete_task_dict,
                                            tokenizer=tokenizer_long,
                                            encode=True, truncation=True, padding=False)
            test_loader = DataLoader(dataset=dataset_test,
                                    batch_size=config.batch_size,
                                    collate_fn=dataset_test._collate_fn,
                                    drop_last=False)
            
            model = MTKB_LongBERT(bert_model_path=model_path_offline,
                                    task_dict=tm.complete_task_dict,
                                    tokenizer=tokenizer_long,
                                    device=config.device,
                                    freeze_bert=config.freeze_bert,
                                    dropout=config.dropout,
                                    chunksize=config.chunksize)
            layer_components_with_grad = [param for param in model.parameters() if param.requires_grad]

            assert 50 > len(layer_components_with_grad) >=2 # at least 2 if only final classification layers are unfrozen;

            optimizer = optim.Adam(layer_components_with_grad, lr=config.lr)

            if config.n_lr_iters:
                lr_scheduler = LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.5, total_iters=config.n_lr_iters)
            else:
                lr_scheduler = None
            

            loss_function = MTNLLLoss(task_dict=tm.complete_task_dict,
                                                device=config.device)
            best_epoch, epoch = training_loop_MT(train_dataloader=train_loader, 
                                        valid_dataloader=valid_loader, 
                                        model=model, 
                                        optimizer=optimizer, 
                                        loss_function=loss_function, 
                                        output_dir=one_experiment_output_folder, 
                                        experiment_name=config.experiment_name,   # with kfold number
                                        print_metric=["f1_macro", "loss"],
                                        device=config.device, 
                                        patience=config.patience, 
                                        min_epochs=2,
                                        max_epochs=config.max_epochs,
                                        n_lr_iters=config.n_lr_iters,
                                        lr_schedule=lr_scheduler)
            test_metrics, label_list = test_saved_MTmodel(model=model, 
                                                            checkpoint_path=os.path.join(one_experiment_output_folder, f"best_{config.experiment_name}.pt"),    #with kfold number
                                                            test_dataloader=test_loader, 
                                                            output_dir=one_experiment_output_folder, 
                                                            experiment_name=config.experiment_name, # with kfold number
                                                            device=config.device,
                                                            task_dict=tm.complete_task_dict)
            with open(os.path.join(one_experiment_output_folder, f"{i}_{tasks_abbr}_result_best{best_epoch}.txt"), "w") as out:
                out.write(f"{config.experiment_name} (saved (=best) ep. {best_epoch}) \n\n {test_metrics} \n\n (plots shown until epoch {epoch})")
            
            cv_results[f"{i}_{config.experiment_name}_best_ep{best_epoch}"] = test_metrics
            with open(log_file, "a") as log:
                log.write(f"{i}_{label_col[:3]}_{config.experiment_name}_best_ep{best_epoch} \t\t {test_metrics}\n")
            time.sleep(120)


        if cv_results:
            final_results = []
            for experiment, task_results in cv_results.items():
                row = {"configuration": experiment}
                for task, metrics_dict in task_results.items():
                    for metric, value in metrics_dict.items():
                        col_name = f"{task[:3]}_{metric}"
                        row[col_name] = value
                final_results.append(row)
            results_df = pd.DataFrame(final_results)
            results_df.set_index("configuration", inplace=True)

        add_stats = {"mean": results_df.mean(),
                    "min": results_df.min(),
                    "max" : results_df.max(),
                    "std" : results_df.std()
                    }
        for stat, values in add_stats.items():
            results_df.loc[stat] = values
        results_df.to_csv(os.path.join(cv_dir, f"summary_{tasks_abbr}_{config.experiment_name}_{datetimestamp}.csv"), encoding="utf-8")
        time.sleep(360)
    time.sleep(360)