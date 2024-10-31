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

from functions_small_helper import create_folder_if_not_exists
from functions_helper import ConfigReader, KFoldReader
from functions_transform_data import TrainVocabulary

from mt_functions_training import TaskManager, MTNLLLoss
from mt_functions_transform import MTHiSAN_SentDataset
from mt_models import MTHiSAN
from functions_training_mt_adaptations import training_loop_MT, test_saved_MTmodel






for label_cols in [sorted(["histology", "behavior", "site"]), sorted(["morphology", "site"])]:
    for suffix in ["_False", ""]:

        # read configs and set output directory name
        config_folder = os.path.join("..", "config_files")
        config_file = os.path.join(config_folder, f"config_mthisan_{len(label_cols)}tasks{suffix}.json")
        config = ConfigReader(config_file)

        ## set up everything
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
        # set random seeds
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if config.class_weights:
            config.class_weights = {"morphology" : "bl015",
                                    "site": "bl015",
                                    "behavior" : "bl1",
                                    "histology": "bl015"}

        dataset_type = "sent" # else: "split" or ""

        if dataset_type == "sent":
            # actual line splits
            text_col = "chunked_sent_tokenized_text"
            max_words_per_line = 50
            max_lines = 150
        else:
            # naive splits
            text_col = "tokenized_text"
            max_words_per_line = 20
            max_lines = 256
        if "morphology" in label_cols:
            sort_by_col = "morphology"
        elif "histology" in label_cols:
            sort_by_col = "histology"
        else:
            "Warning: no core task included. Please include histology or morphology in list"
            exit()

        kfold_label_col = "morphology"  # kfold stratification is based on morphology

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

        hisan_dir = os.path.join(experiments_dir, config.model_type)
        create_folder_if_not_exists(hisan_dir)


        assert type(label_cols) == list

        tasks_abbr = ""
        for task in label_cols:
            tasks_abbr += task[:3]
        cv_dir = os.path.join(hisan_dir, f"{config.model_type}_cv_{tasks_abbr}")
        create_folder_if_not_exists(cv_dir)

        cv_results = dict()


        log_file = os.path.join(cv_dir, f"logg_{tasks_abbr}_{config.experiment_name}_{datetimestamp}.txt")



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
            
            voc = TrainVocabulary(train_valid_data["tokenized_text"][train_ind], unk_threshold=config.unk_threshold)
            print(f"Training vocabulary size: {len(voc)}")
            
            print(f"Number of classes for tasks {tm.sorted_task_list} in the training set: {tm.n_classes_of_tasks}")

            emb_matrix = voc.create_word2vec(train_valid_data["tokenized_text"][train_ind], vector_size=config.embedding_size)

            # train
            dataset_train = MTHiSAN_SentDataset(df_split=train_valid_data.iloc[train_ind],
                                            text_col=text_col, 
                                            label_cols=label_cols,
                                            vocab=voc,
                                            task_manager=tm.complete_task_dict,
                                            max_lines=max_lines,
                                            max_words_per_line=max_words_per_line,
                                            replace_oov=True,
                                            encode=True)
            train_loader = DataLoader(dataset=dataset_train,
                                    batch_size=config.batch_size,
                                    collate_fn=dataset_train.collate_fn,
                                    drop_last=False)
            # valid
            dataset_valid = MTHiSAN_SentDataset(df_split=train_valid_data.iloc[valid_ind],
                                            text_col=text_col, 
                                            label_cols=label_cols,
                                            vocab=voc,
                                            task_manager=tm.complete_task_dict,
                                            max_lines=max_lines,
                                            max_words_per_line=max_words_per_line,
                                            replace_oov=True,
                                            encode=True)
            valid_loader = DataLoader(dataset=dataset_valid,
                                    batch_size=config.batch_size,
                                    collate_fn=dataset_valid.collate_fn,
                                    drop_last=False)
            # test
            dataset_test = MTHiSAN_SentDataset(df_split=test_data,
                                            text_col=text_col, 
                                            label_cols=label_cols,
                                            vocab=voc,
                                            task_manager=tm.complete_task_dict,
                                            max_lines=max_lines,
                                            max_words_per_line=max_words_per_line,
                                            replace_oov=True,
                                            encode=True)
            test_loader = DataLoader(dataset=dataset_test,
                                    batch_size=config.batch_size,
                                    collate_fn=dataset_test.collate_fn,
                                    drop_last=False)


            model = MTHiSAN(embedding_matrix=emb_matrix,
                            task_dict=tm.complete_task_dict, 
                            max_words_per_line=max_words_per_line,
                            max_lines=max_lines,
                            att_dim_per_head=config.attention_dim,
                            att_heads=config.attention_heads,
                            device=config.device, 
                                )

            optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
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
                                min_epochs=6,
                                max_epochs=config.max_epochs,
                                n_lr_iters=config.n_lr_iters,
                                lr_schedule=None)
            
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
        time.sleep(180)
