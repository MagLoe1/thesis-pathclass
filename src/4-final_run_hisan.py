import numpy as np
import os
import random
import pandas as pd
from datetime import date, datetime
import time

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# other
# from collections import Counter
# from gensim.models import Word2Vec

from functions_small_helper import create_folder_if_not_exists
from functions_helper import ConfigReader, KFoldReader, LabelMapper
from functions_transform_data import TrainVocabulary, HiSANDataset, HiSAN_SentDataset
from functions_training import training_loop, test_saved_model, SingleTaskManager
from models import HiSAN

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

kfold_label_col = "morphology"  # kfold stratification is based on morphology


for label_col in ["morphology", "histology", "behavior", "site"]:
    for suffix in ["_False", ""]:

        # read configs and set output directory name
        config_folder = os.path.join("..", "config_files")
        config_file = os.path.join(config_folder, f"config_hisan_{label_col[:3]}{suffix}.json")
        config = ConfigReader(config_file)

        ## set up everything
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
        else:
            exit()
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

        hisan_dir = os.path.join(experiments_dir, config.model_type)
        create_folder_if_not_exists(hisan_dir)




        # add cross validation loop:
        cv_dir = os.path.join(hisan_dir, f"{config.model_type}_cv_{label_col[:3]}")
        create_folder_if_not_exists(cv_dir)

        train_valid_data = pd.read_pickle(data_file).reset_index(drop=True)
        test_data = pd.read_pickle(test_file).reset_index(drop=True)

        train_valid_data[label_col] = train_valid_data[label_col].astype("str")
        test_data[label_col] = test_data[label_col].astype("str")
        assert type(train_valid_data[text_col][0] == list)
        label_mapper = LabelMapper(train_valid_data[label_col])
        n_classes = label_mapper.n_unique_labels

        cv_results = dict()


        log_file = os.path.join(cv_dir, f"logg_{label_col[:3]}_{config.experiment_name}{dataset_type}_{datetimestamp}.txt")


        for i, (train_ind, valid_ind) in enumerate(kfolds.folds[1:], start=1):

            one_experiment_output_folder = os.path.join(cv_dir, f"{i}-{label_col[:3]}-{config.experiment_name}")
            create_folder_if_not_exists(one_experiment_output_folder)
            print(f"--- Experiment {config.experiment_name} FOLD {i} -----")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            tm = SingleTaskManager(df_train=train_valid_data.iloc[train_ind], 
                                    task_list=[label_col],  # set to list here for ST models
                                    cw_type=config.class_weights)
            # Important for HiSAN: change here to tokenized text
            voc = TrainVocabulary(train_valid_data["tokenized_text"][train_ind], unk_threshold=config.unk_threshold)
            print(f"Training vocabulary size: {len(voc)}")
            
            print(f"Number of classes in the training set for '{label_col}': {n_classes}")
            assert label_mapper.label_map == tm.complete_task_dict[label_col]["label_map"]
            print(f"Labels: {tm.complete_task_dict[label_col]['sorted_labels_alph']}")
            print(f"Class weights: {tm.complete_task_dict[label_col]['class_weights']}")

            # Important for HiSAN: change here to tokenized text
            emb_matrix = voc.create_word2vec(train_valid_data["tokenized_text"][train_ind], vector_size=config.embedding_size)
            if dataset_type == "sent":
                assert max_words_per_line == 50
                dataset_train = HiSAN_SentDataset(train_valid_data[text_col][train_ind], 
                                        train_valid_data[label_col][train_ind], 
                                        voc, 
                                        label_map=label_mapper.label_map,
                                        replace_oov=True,
                                        encode=True,
                                        max_words_per_line=max_words_per_line,
                                        max_lines=max_lines)
                dataset_val = HiSAN_SentDataset(train_valid_data[text_col][valid_ind], 
                                            train_valid_data[label_col][valid_ind], 
                                            voc, 
                                            label_map=label_mapper.label_map,
                                            replace_oov=True,
                                            encode=True,
                                            max_words_per_line=max_words_per_line,
                                            max_lines=max_lines)
                dataset_test = HiSAN_SentDataset(text_col=test_data[text_col],
                                                 label_col=test_data[label_col],
                                                 vocab=voc,
                                                 label_map=label_mapper.label_map,
                                                 replace_oov=True,
                                                 encode=True,
                                                 max_words_per_line=max_words_per_line,
                                                 max_lines=max_lines)
            else:
                dataset_train = HiSANDataset(train_valid_data[text_col][train_ind], 
                                            train_valid_data[label_col][train_ind], 
                                            voc, 
                                            label_map=label_mapper.label_map,
                                            replace_oov=True,
                                            encode=True,
                                            max_words_per_line=max_words_per_line,
                                            max_lines=max_lines)
                dataset_val = HiSANDataset(train_valid_data[text_col][valid_ind], 
                                            train_valid_data[label_col][valid_ind], 
                                            voc, 
                                            label_map=label_mapper.label_map,
                                            replace_oov=True,
                                            encode=True,
                                            max_words_per_line=max_words_per_line,
                                            max_lines=max_lines)
                dataset_test = HiSANDataset(text_col=test_data[text_col],
                                            label_col=test_data[label_col],
                                            vocab=voc,
                                            label_map=label_mapper.label_map,
                                            replace_oov=True,
                                            encode=True,
                                            max_words_per_line=max_words_per_line,
                                            max_lines=max_lines)

            train_loader = DataLoader(dataset=dataset_train,
                                    batch_size=config.batch_size,
                                    collate_fn=dataset_train.collate_fn,
                                    drop_last=False)
            
            valid_loader = DataLoader(dataset=dataset_val,
                                    batch_size=config.batch_size,
                                    collate_fn=dataset_val.collate_fn,
                                    drop_last=False)
            test_loader = DataLoader(dataset=dataset_test,
                                     batch_size=config.batch_size,
                                     collate_fn=dataset_test.collate_fn,
                                     drop_last=False)

            model = HiSAN(embedding_matrix=emb_matrix,  
                        max_words_per_line=max_words_per_line,
                        max_lines=max_lines,
                        att_dim_per_head=config.attention_dim,
                        att_heads=config.attention_heads,
                        device=config.device, 
                        n_classes=n_classes)

            optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
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
                                experiment_name=config.experiment_name,   # with kfold number
                                print_metric=["f1_macro", "loss"],
                                device=config.device, 
                                patience=config.patience,
                                min_epochs=5,
                                max_epochs=config.max_epochs,
                                n_lr_iters=config.n_lr_iters,
                                lr_schedule=None)

            
            # DONE: changed to testloader for final runs
            test_metrics, label_list = test_saved_model(model=model, 
                                                        checkpoint_path=os.path.join(one_experiment_output_folder, f"best_{config.experiment_name}.pt"),    #with kfold number
                                                        test_dataloader=test_loader, 
                                                        output_dir=one_experiment_output_folder, 
                                                        experiment_name=config.experiment_name, # with kfold number
                                                        device=config.device,
                                                        label_list=label_mapper.labels)
            
            with open(os.path.join(one_experiment_output_folder, f"{i}_{label_col[:3]}_result_best{best_epoch}.txt"), "w") as out:
                out.write(f"{config.experiment_name} (saved (=best) ep. {best_epoch}) \t\t {test_metrics} (shown until epoch {epoch})")
            test_metrics.update({"best_epoch": best_epoch})
            cv_results[f"{i}_{config.experiment_name}_best_ep{best_epoch}"] = test_metrics

            with open(log_file, "a") as log:
                log.write(f"{i}_{label_col[:3]}_{config.experiment_name}_best_ep{best_epoch} \t\t {test_metrics}\n")
        results_df = pd.DataFrame.from_dict(cv_results, orient="index")
        add_stats = {"mean": results_df.mean(),
                     "min": results_df.min(),
                     "max" : results_df.max(),
                     "std" : results_df.std()
                     }
        for stat, values in add_stats.items():
            results_df.loc[stat] = values
        results_df.to_csv(os.path.join(cv_dir, f"summary_{label_col[:3]}_{config.experiment_name}_{datetimestamp}.csv"), encoding="utf-8")
