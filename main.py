from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging
from simpletransformers.ner import NERModel, NERArgs
import json
import pandas as pd
import time

import evaluate
from datetime import datetime
import gc
import torch

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
model = AutoModelForMaskedLM.from_pretrained("EMBEDDIA/sloberta")

datasets = [
    "datasets/SUK.CoNLL-U/elexiswsd.ud.conllu_extracted.json",
    "datasets/SUK.CoNLL-U/senticoref.ud.conllu_extracted.json",
    "datasets/SUK.CoNLL-U/ssj500k-syn.ud.conllu_extracted.json",
    "datasets/Janes-Tag.3.0.CoNLL-U/janes-tag.ud.conllu_extracted.json"
]

# Do a loop through the datasets

for dataset_path in datasets:
    # Load the json file
    with open(dataset_path, "r") as file:
        json_dict = json.load(file)

    # Open the train, eval and test dictionaries as DataFrames
    train_df = pd.DataFrame(json_dict["train"])
    test_df = pd.DataFrame(json_dict["test"])
    dev_df = pd.DataFrame(json_dict["dev"])

    # Change the sentence_ids to numbers
    test_df['sentence_id'] = pd.factorize(test_df['sentence_id'])[0]
    train_df['sentence_id'] = pd.factorize(train_df['sentence_id'])[0]
    dev_df['sentence_id'] = pd.factorize(dev_df['sentence_id'])[0]

    # Define the labels
    LABELS = json_dict["labels"]
    print(LABELS)

    print(train_df.shape, test_df.shape, dev_df.shape)
    print(train_df.head())


    def train_and_test(model, train_df, test_df, dataset_path, LABELS):

        # Define the model arguments - these arguments hold for all models,
        # we just change the epoch number based on hyperparameter search
        model_args = {"overwrite_output_dir": True,
                      "num_train_epochs": 5,
                      "labels_list": LABELS,
                      "learning_rate": 4e-05,
                      "train_batch_size": 32,
                      "logging_dir": "./logs",
                      "logging_steps": 50,
                      # Comment out no_cache and no_save if you want to save the model
                      "no_cache": True,
                      "no_save": True,
                      "max_seq_length": 256,
                      "save_steps": -1,
                      "silent": False,
                      }

        # Define the type of dataset we are using
        # - when we extend the code for SL, change this
        dataset_type = "standard_sl"

        model_args["num_train_epochs"] = 7

        # Model type - a dictionary of type and model name.
        model_type_dict = {
            "sloberta": ["camembert", "EMBEDDIA/sloberta"],
            "csebert": ["bert", "EMBEDDIA/crosloengual-bert"],
            "xlm-r-base": ["xlmroberta", "xlm-roberta-base"],
            "xlm-r-large": ["xlmroberta", "xlm-roberta-large"],
            "bertic": ["electra", "classla/bcms-bertic"],
        }

        # Define the model
        current_model = NERModel(
            model_type_dict[model][0],
            model_type_dict[model][1],
            labels=LABELS,
            use_cuda=True,
            args=model_args)

        print("Training started. Current model: {}, no. of epochs: {}".format(model, model_args["num_train_epochs"]))
        start_time = time.time()

        # Fine-tune the model
        current_model.train_model(train_df)

        print("Training completed.")

        training_time = round((time.time() - start_time) / 60, 2)

        print("It took {} minutes for {} instances.".format(training_time, train_df.shape[0]))

        # Clean cache
        gc.collect()
        torch.cuda.empty_cache()

        start_evaluation_time = time.time()

        # Evaluate the model
        results = current_model.eval_model(test_df)

        print("Evaluation completed.")

        evaluation_time = round((time.time() - start_evaluation_time) / 60, 2)

        print("It took {} minutes for {} instances.".format(evaluation_time, test_df.shape[0]))

        # Get predictions
        preds = results[1]

        # Create a list with predictions
        preds_list = []

        for sentence in preds:
            for word in sentence:
                current_word = []
                for element in word:
                    # Find prediction with the highest value
                    highest_index = element.index(max(element))
                    # Transform the index to label
                    current_pred = current_model.config.id2label[highest_index]
                    # Append to the list
                    current_word.append(current_pred)
                # Segmentation can result in multiple predictions for one word - use the first prediction only
                preds_list.append(current_word[0])

        # Get y_true
        y_true = list(test_df.labels)

        run_name = "{}-{}".format(dataset_path, model)

        # Evaluate predictions
        metrics = evaluate.testing(y_true, preds_list, list(test_df.labels.unique()), run_name, show_matrix=True)

        # Add y_pred and y_true to the metrics dict
        metrics["y_true"] = y_true
        metrics["y_pred"] = preds_list

        # Let's also add entire results
        metrics["results_output"] = results

        # The function returns a dict with accuracy, micro f1, macro f1, y_true and y_pred
        return metrics


    # For each model, repeat training and testing 3 times
    model_list = ["sloberta"]

    for model in model_list:
        # for run in list(range(2)):
        # Let's do two more runs
        # for run in [0,1,2]:
        for run in [0]:
            current_results_dict = train_and_test(model, train_df, test_df, dataset_path, LABELS)

            # Add to the dict model name, dataset name and run
            current_results_dict["model"] = model
            current_results_dict["run"] = "{}-{}".format(model, run)
            current_results_dict["dataset"] = dataset_path

            # Add to the file with results all important information
            with open("ner-results.txt", "a") as file:
                file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                                 current_results_dict["model"],
                                                                 current_results_dict["run"],
                                                                 current_results_dict["dataset"],
                                                                 current_results_dict["micro F1"],
                                                                 current_results_dict["macro F1"],
                                                                 current_results_dict["label-report"]))

            # Add to the original test_df y_preds
            test_df["y_pred_{}_{}".format(model, run)] = current_results_dict["y_pred"]

            # Save also y_pred and y_true
            with open("logs/{}-{}-{}-true-and-pred-backlog.txt".format(dataset_path, model, run), "w") as backlog:
                backlog.write("y-true\ty-pred\toutputs\n")
                backlog.write("{}\t{}\t{}\n".format(current_results_dict["y_true"], current_results_dict["y_pred"],
                                                    current_results_dict["results_output"]))

            print("Run {} finished.".format(run))

    # At the end, save the test_df with all predictions
    test_df.to_csv("{}-test_df-with-predictions.csv".format(dataset_path))

    # At the end, create a csv table with a summary of results

    results = pd.read_csv("ner-results.txt", sep="\t")

    results["Macro F1"] = results["Macro F1"].round(2)

    # Pivot the DataFrame to rearrange columns into rows
    pivot_df = results.pivot(index='Run', columns='Dataset', values='Macro F1')

    # Reset the index to have 'Model' as a column
    pivot_df.reset_index(inplace=True)

    # Save the summary results
    pivot_df.to_csv("ner-results-summary-table.csv")